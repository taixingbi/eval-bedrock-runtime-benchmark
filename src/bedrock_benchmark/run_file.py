"""Run one experiment YAML against one model end to end -- bind, sweep,
print progress and recommendations, write the raw JSONL +
capacity-profile.yaml into <results_dir>/<model name>/. Shared by
scripts/run.py and scripts/run_all.py, so both produce identical
output and artifacts.
"""
from __future__ import annotations

import asyncio
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Collection, List, Optional

import yaml

from .analysis.capacity import SweepPoint
from .analysis.metrics import DEFAULT_CONFIDENCE, min_samples_to_resolve_rate
from .client import BedrockConverseTarget
from .experiments.executor import ExperimentReport, _slo_kwargs, run_experiment
from .constraints import DEFAULT_SLO_FILE
from .experiments.schema import ExperimentSpec, load_experiment
from .models import ModelConfig
from .workload import DEFAULT_WORKLOADS_FILE
from .recommendation import admission_envelope
from .report import build_capacity_profile
from .storage import write_jsonl

# Builds the Bedrock target for a spec -- injectable so tests can
# substitute a fake client; None means the real boto3 target.
TargetFactory = Callable[[ExperimentSpec], BedrockConverseTarget]


@dataclass
class RunOutcome:
    spec: ExperimentSpec
    report: ExperimentReport
    capacity_profile: dict
    jsonl_path: Path
    profile_path: Path
    elapsed_s: float


def estimated_duration_s(spec: ExperimentSpec) -> float:
    """Wall-time estimate: every sweep point runs warmup + window per
    repetition, per sweep subject (each workload, or one mix), plus --
    with confirmation -- per subject:

    the repetitions needed to reach the FIRST pre-planned look at each
    likely candidate, i.e. assuming it PASSes there -- 0 if that look is
    unreachable within the caps (skipped unspent). A rate sweep's
    candidates are its highest rates at or below the provider ceiling;
    a concurrency sweep's request rate isn't known up front, so its
    candidates are assumed to run at the ceiling (a non-failing point
    can't sustain much more). Capped at min(max_duration_s, candidates
    x max_repetitions x (warmup + window)) -- max_duration_s alone when
    there's no repetition cap (`auto`: uncapped -- the time the first look
    needs at that rate); without a known ceiling, the cap itself. Plus `cooldown_s` once per subject that confirms.

    Real runs take longer when a look is spent on a stray bad event
    (up to the caps) and shorter on an early FAIL. Drain time on top
    depends on real latency, so it isn't counted."""
    if spec.history_protocol is not None:
        h = spec.history_protocol
        probe = spec.recovery_probe.duration_s if h.recovery_mode == "verified" else 0
        delays = h.recovery_delays_s or [h.recovery_s]
        return sum(len(spec.sweep_values(n)) for n in spec.subject_names) * spec.repetitions * (
            (1 + len(delays)) * (h.idle_s + spec.duration_s + probe)
            + len(delays) * (h.overload_duration_s + probe) + sum(delays))
    per_run = spec.warmup_s + spec.duration_s
    refine = spec.sweep.refinement.max_points if spec.sweep.refinement is not None else 0
    discovery = refine * spec.repetitions * per_run
    if spec.sweep.refinement is not None:
        discovery += spec.sweep.refinement.max_points * spec.sweep.refinement.cooldown_s  # before each point
    probe = spec.recovery_probe.duration_s if spec.recovery_probe else 0.0
    if spec.sweep.refinement is not None:
        discovery += spec.sweep.refinement.max_points * probe  # a recovery probe before each refinement point
    total = 0.0
    for index, subject in enumerate(spec.subject_names):
        subject_probe = probe
        if spec.recovery_probe and spec.recovery_probe.baseline_fraction is not None:
            rp = spec.recovery_probe
            rate = spec.provider_ceilings[subject].rps * rp.baseline_fraction
            subject_probe += spec.warmup_s + max(rp.baseline_duration_s, 1.25 * rp.baseline_min_requests / rate)
        if spec.burst_protocol:
            total += len(spec.sweep_values(subject)) * (subject_probe + spec.recovery_probe.retry_cooldown_s)
        total += subject_probe  # at the start of each subject (healthy probes; unhealthy ones retry)
        if index > 0:
            total += spec.inter_subject_cooldown_s
        confirm = _confirmation_estimate_s(spec, subject, per_run)
        if confirm > 0:  # cooldown + probe + conditioning for the (highest) candidate, assumed to PASS
            confirm += spec.confirmation.cooldown_s + subject_probe + spec.confirmation.warmup_s
        total += discovery + len(spec.sweep_values(subject)) * spec.repetitions * per_run + confirm
    return total


# Quota-relative sweep values are rounded to 4 decimals (schema.sweep_values),
# so the 1.0x point (e.g. 6.6667 rps) sits a hair above the unrounded
# ceiling (6.66666...). Without this tolerance the 1.0x point -- the most
# useful candidate -- would never be confirmed.
_ROUNDING_TOLERANCE = 1e-4


def _confirmation_estimate_s(spec: ExperimentSpec, subject: str, per_run: float) -> float:
    c = spec.confirmation
    if c is None:
        return 0.0
    rep_cap = c.max_repetitions if c.max_repetitions is not None else math.inf
    auto_duration = c.max_duration_s == "auto"
    cap = min(math.inf if auto_duration else c.max_duration_s, c.candidates * rep_cap * per_run)
    ceiling = spec.provider_ceilings.get(subject)
    if ceiling is None or not ceiling.rps:
        return cap if cap < math.inf else 0.0
    from .analysis.confirmation import limits_for, plan_looks
    if spec.mix is not None:
        total_weight = sum(spec.mix.weights.values())
        shares = {n: w / total_weight for n, w in spec.mix.weights.items()}
        gate = _slo_kwargs(spec.slo, latency=False)
        class_gate = {n: _slo_kwargs(spec.slo_for(n)) for n in shares}
    else:
        shares, class_gate = None, None
        gate = _slo_kwargs(spec.slo_for(subject))
    if spec.sweep.type == "rate":
        eligible = sorted(v for v in spec.sweep_values(subject) if v <= ceiling.rps + _ROUNDING_TOLERANCE)
        if not eligible:
            return 0.0  # The executor cannot confirm any rate above the nominal ceiling.
        k = min(c.candidates, len(eligible))
        top_rps = eligible[-1] if eligible else 0.0
    else:
        k = min(c.candidates, len(spec.sweep_values(subject)))
        top_rps = ceiling.rps
    # alpha is split over the K candidates; they're tested highest-first,
    # and the estimate assumes the highest PASSes at its first look.
    plan = plan_looks(limits_for(gate, class_gate, shares), confidence=gate["confidence"],
                      max_looks=c.max_looks, max_repetitions=c.max_repetitions, max_requests=c.max_requests,
                      max_duration_s=c.max_duration_s, candidates=max(1, k),
                      min_steady_state_duration_s=c.min_steady_state_duration_s)
    duration = max(spec.duration_s, c.min_steady_state_duration_s) if c.continuous else spec.duration_s
    per_run = spec.warmup_s + duration
    per_rep = top_rps * duration
    reps = math.ceil(plan.look_schedule[0] / per_rep) if per_rep > 0 else math.inf
    reps = max(reps, math.ceil(c.min_steady_state_duration_s / duration))
    max_requests = plan.look_schedule[-1] if c.max_requests == "auto" else c.max_requests
    if reps > rep_cap or plan.look_schedule[0] > max_requests:
        return 0.0
    # `auto` duration: no fixed cap -- the first look simply takes as long
    # as it takes at this rate (a low-RPM model shows up here as hours).
    return min(reps * per_run, cap)


def describe_sweep(spec: ExperimentSpec) -> str:
    """e.g. "concurrency [1, 2, 4]" or "rate 0.25x-2.5x of ceiling:
    short=6.67rps(rpm)" -- a quota-relative sweep's rps differ per
    subject, so the ceiling each resolves against is shown."""
    if spec.sweep.values_by_workload is not None:
        return 'rate ' + ', '.join(f'{n}={spec.sweep_values(n)}' for n in spec.subject_names)
    if spec.sweep.quota_fractions is None:
        stop = f" until {spec.sweep.stop_after_fails} FAILs" if spec.sweep.stop_after_fails else ""
        return f"{spec.sweep.type} {spec.sweep.values}{stop}"
    f = spec.sweep.quota_fractions
    ceilings = ", ".join(
        f"{name}={c.rps:.4g}rps({c.binding})" for name, c in spec.provider_ceilings.items()
    )
    return f"rate {min(f):g}x-{max(f):g}x of ceiling: {ceilings}"


def recommendation_summary(report: ExperimentReport) -> List[str]:
    """One line per sweep subject -- used both after a single run and in
    run_all's final table."""
    lines = []
    for profile_report in report.profiles:
        if report.spec.purpose == 'characterization':
            arms = profile_report.history_comparison
            invalid = (profile_report.measurement_validity or {}).get('status') == 'invalid'
            incomplete = arms is not None and (
                len(arms) != len(report.spec.sweep_values(profile_report.workload_name)) * report.spec.repetitions *
                (1 + len(report.spec.history_protocol.recovery_delays_s or [report.spec.history_protocol.recovery_s]))
                or any(a.get('status') != 'observed' for a in arms))
            status = 'incomplete or invalid' if invalid or incomplete else 'complete'
            lines.append(f'{profile_report.workload_name}: characterization {status}; '
                         'no admission envelope produced by diagnostic experiment')
            continue
        rec = profile_report.recommendation
        if rec is None:
            lines.append(f"{profile_report.workload_name}: NO swept value met the configured SLO")
            continue
        value = rec.point.concurrency if rec.point.concurrency is not None else rec.point.rps
        sat = None
        if rec.saturation_point is not None:
            sat = rec.saturation_point.concurrency if rec.saturation_point.concurrency is not None else rec.saturation_point.rps
        confirmed = rec.confirmed_point
        confirmed_value = None if confirmed is None else (
            confirmed.concurrency if confirmed.concurrency is not None else confirmed.rps)
        spec = report.spec
        if spec.purpose == "admission_calibration":
            policy = ("calibration point (confirmed, no headroom) -- for gateway admission-class derivation"
                      if confirmed_value is not None else "no calibration point (nothing statistically confirmed)")
        elif spec.purpose != "reference":
            policy = "no recommendation (characterization experiment)"
        elif confirmed_value is None:
            policy = "no recommendation (nothing statistically confirmed)"
        else:
            ceiling = spec.provider_ceilings.get(profile_report.workload_name)
            envelope = admission_envelope(
                spec.sweep.type, confirmed_value, headroom=spec.provider_headroom,
                quota_headroom=spec.quota_headroom, provider_ceiling_rps=ceiling.rps if ceiling else None,
            )["admission_envelope"]
            policy = "no recommendation (confirmed point too small for the headroom)" if envelope is None else (
                f"recommendation: max_inflight={envelope['max_inflight']}" if spec.sweep.type == "concurrency"
                else f"recommendation: sustained_rps={envelope['sustained_rps']}")
        lines.append(
            f"{profile_report.workload_name}: measured observed_nonfailing={value} [{rec.verdict.verdict}] "
            f"statistically_confirmed={confirmed_value} saturation={sat} | {policy}"
        )
    return lines


def _make_progress_printer():
    start = time.perf_counter()

    def printer(workload_name: str, sweep_value: float, point: SweepPoint) -> None:
        elapsed = time.perf_counter() - start
        m = point.metrics
        ttft = f"{m.ttft_p95_ms}ms" if m.ttft_p95_ms is not None else "n/a"
        goodput = f"{m.slo_goodput_rps}" if m.slo_goodput_rps is not None else "n/a"
        print(
            f"  [{elapsed:6.0f}s] {workload_name:<12} value={sweep_value:<6} "
            f"n={m.n:<5} success={m.success_rate:.3f} throttle={m.throttle_rate:.4f} "
            f"(<={m.throttle_rate_upper}) "
            f"ttft_p95={ttft:<10} tpot_p95={m.tpot_p95_ms}ms latency_p95={m.latency_p95_ms}ms slo_goodput={goodput}"
        )

    return printer


def _warn_if_throttle_slo_unresolvable(spec: ExperimentSpec) -> None:
    """Rate sweeps know their expected sample size up front -- say so
    before spending real Bedrock calls if the throttle SLO can't be
    statistically demonstrated at that size."""
    confidence = spec.slo.confidence or DEFAULT_CONFIDENCE
    needed = min_samples_to_resolve_rate(spec.slo.throttle_rate_max, confidence=confidence)
    if spec.sweep.type != "rate":
        print(f"note: resolving throttle_rate_max={spec.slo.throttle_rate_max} at {confidence:.0%} "
              f"needs >= {needed} measured requests per point")
        return
    short = sorted({
        v for name in spec.subject_names for v in spec.sweep_values(name)
        if v * spec.duration_s * spec.repetitions < needed
    })
    if short:
        gate = "these points will FAIL the SLO gate" if spec.slo.confidence is not None else \
            "a 0-throttle pass at these points is not statistically meaningful"
        print(f"warning: rates {short} expect fewer than {needed} measured requests "
              f"(duration_s x repetitions too short to resolve throttle_rate_max="
              f"{spec.slo.throttle_rate_max} at {confidence:.0%}) -- {gate}")


def run_file(
    path: str, model: ModelConfig, *, results_dir: str = "results", target_factory: Optional[TargetFactory] = None,
    slo_file: str = DEFAULT_SLO_FILE, workloads_file: str = DEFAULT_WORKLOADS_FILE,
    only_slo_profiles: Optional[Collection[str]] = None, mix: Optional[str] = None, retest: Optional[dict] = None, run_metadata: Optional[dict] = None,
) -> RunOutcome:
    spec = load_experiment(path, model, slo_file=slo_file, workloads_file=workloads_file,
                           only_slo_profiles=only_slo_profiles, mix=mix, retest=retest)
    if run_metadata and "workload_rotation_index" in run_metadata:
        spec.workload_rotation_index = run_metadata["workload_rotation_index"]
    print(f"running experiment: {spec.name} on {model.name} ({model.model_id})")
    print(f"sweep: {describe_sweep(spec)}")
    for name in spec.subject_names:
        if spec.sweep.quota_fractions is not None:
            print(f"  {name}: {spec.sweep_values(name)} rps")
    if spec.description:
        print(spec.description.strip())
    _warn_if_throttle_slo_unresolvable(spec)

    start = time.perf_counter()
    target = target_factory(spec) if target_factory is not None else None
    run_id = str(uuid.uuid4())[:8]
    out_dir = Path(results_dir) / model.name
    checkpoints = None
    options = {}
    if spec.history_protocol is not None:
        from .history_checkpoint import HistoryCheckpoints
        checkpoints = HistoryCheckpoints(
            out_dir / f"{spec.name}-{run_id}-checkpoints", spec,
            {**(run_metadata or {}), "run_id": run_id})
        options["on_history_arm"] = checkpoints.save_arm
        print(f"local progress: {checkpoints.directory / 'manifest.yaml'}", flush=True)
    try:
        report = asyncio.run(run_experiment(spec, on_progress=_make_progress_printer(), target=target, **options))
    except BaseException as exc:
        if checkpoints is not None:
            try:
                checkpoints.finish("interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed")
            except OSError:
                pass  # Keep the original error; previously published artifacts remain readable.
        raise
    if checkpoints is not None:
        checkpoints.finish()
    elapsed_s = time.perf_counter() - start

    print("\n-- input-token calibration --")
    for name, c in report.calibrations.items():
        if c.method == "estimate":
            print(f"  {name}: estimate, 4 chars/token ({c.note})")
        else:
            status = "converged" if c.converged else "closest, not within tolerance"
            print(f"  {name}: {c.method} -> {c.counted_input_tokens} tokens ({status}, {c.iterations} steps)"
                  + (f" [{c.note}]" if c.note else ""))

    print("\n-- recommendations --")
    for line in recommendation_summary(report):
        print(f"  {line}")
    for profile_report in report.profiles:
        rec = profile_report.recommendation
        if rec is None:
            continue
        for class_name, m in rec.point.class_metrics.items():
            print(f"    {class_name}: n={m.n} ttft_p95={m.ttft_p95_ms}ms tpot_p95={m.tpot_p95_ms}ms "
                  f"latency_p95={m.latency_p95_ms}ms "
                  f"slo_goodput={m.slo_goodput_rps}")

    jsonl_path = out_dir / f"{spec.name}-{run_id}.jsonl"
    profile_path = out_dir / f"{spec.name}-{run_id}-capacity-profile.yaml"

    write_jsonl(report.all_results, str(jsonl_path))
    capacity_profile = build_capacity_profile(report, {"run_id": run_id, **(run_metadata or {})})
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(yaml.safe_dump(capacity_profile, sort_keys=False))

    for name, entry in capacity_profile["workload_classes"].items():
        v = entry["workload_validation"]
        for side, why in (("input", f"padding sized by {v['token_counting']['method']} missed"),
                          ("output", "the model stopped well short of max_tokens")):
            c = v[side]
            if c["valid"] is False:
                print(f"\nwarning: {name} {side} p50 {c['observed_p50']} tokens vs target {c['target']} "
                      f"({c['deviation_pct']}%, tolerance {c['tolerance_pct']}%) -- {why}; "
                      f"its envelope describes a different workload shape")
    for subject, entry in {**capacity_profile["workload_classes"], **capacity_profile.get("mixed_workloads", {})}.items():
        if entry.get("client_limited_points"):
            print(f"\nwarning: {subject} points {entry['client_limited_points']} queued for client threads "
                  f"(peak outstanding > executor_workers={capacity_profile['transport']['executor_workers']}) "
                  f"-- excluded from the recommendation; raise transport.max_connections")
        generator = entry.get("load_generator")
        if generator and not generator["valid"]:
            print(f"\nwarning: {subject} load generator lagged (p99 {generator['worst_point']['lag_p99_ms']} ms at "
                  f"{generator['worst_point']['value']} > {generator['limit_p99_ms']} ms) -- arrivals started late, "
                  f"so the offered load wasn't what was scheduled; this run measured the client too")

    print(f"\nraw results:      {jsonl_path}")
    print(f"capacity profile: {profile_path}")
    return RunOutcome(
        spec=spec, report=report, capacity_profile=capacity_profile,
        jsonl_path=jsonl_path, profile_path=profile_path, elapsed_s=elapsed_s,
    )
