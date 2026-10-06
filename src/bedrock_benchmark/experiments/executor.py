"""Runs an ExperimentSpec's full sweep:

1. for each sweep subject -- each workload in isolation, or the one
   WorkloadMix when `mix:` is set -- for each swept concurrency/rate
   value, run `repetitions` measurement windows and compute pooled
   RunMetrics (plus per-class metrics for a mix);
2. one Recommendation per subject (see analysis/capacity.py).
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Union

from ..analysis.capacity import (
    FAIL, INCONCLUSIVE, PASS, Recommendation, SweepAnalysis, SweepPoint, Verdict, analyze_sweep, point_verdict, recommend,
)
from ..analysis.confirmation import (
    ConfirmationPlan, ConfirmationResult, candidate_caps, highest_confirmed, limits_for, look_sample, plan_looks,
    reachable, severe_throttling, step,
)
from ..analysis.metrics import DEFAULT_CONFIDENCE, compute_run_metrics
from ..calibration import CalibrationResult, calibrate_profile, estimate_profile, resolve_counter
from ..client import BedrockConverseTarget
from ..results import RequestResult
from ..runners.concurrency import ConcurrencyRunner
from ..runners.rate import RateRunner
from ..workload import WorkloadMix, WorkloadProfile
from .schema import ExperimentSpec

# (subject_name, sweep_value, point) -- called once per completed
# sweep point, so a long multi-point sweep isn't silent until it's
# entirely done.
ProgressCallback = Callable[[str, float, SweepPoint], None]


@dataclass
class ProfileReport:
    # A workload name, or the mix name for a mixed-workload sweep.
    workload_name: str
    points: List[SweepPoint] = field(default_factory=list)
    recommendation: Optional[Recommendation] = None
    # Normalized class shares when this subject is a WorkloadMix.
    mix_shares: Optional[Dict[str, float]] = None
    # Always set -- also when there's no recommendation, so the report
    # can say WHY (never passed vs non-monotonic from the first point).
    analysis: Optional[SweepAnalysis] = None
    # PASS / FAIL / INCONCLUSIVE per DISCOVERY point, aligned with `points`.
    verdicts: List[Verdict] = field(default_factory=list)
    # Confirmation phase (None/empty when not configured or nothing to confirm).
    confirmation_plan: Optional["ConfirmationPlan"] = None
    confirmations: List["ConfirmationResult"] = field(default_factory=list)
    # Provider-state validity of this subject's measurement: status
    # valid | suspect_* | invalid, the suspect events and every
    # recovery probe (see ExperimentSpec.recovery_probe).
    measurement_validity: Optional[dict] = None
    history_comparison: Optional[list] = None


@dataclass
class ExperimentReport:
    spec: ExperimentSpec
    profiles: List[ProfileReport] = field(default_factory=list)
    all_results: List[RequestResult] = field(default_factory=list)
    calibrations: Dict[str, CalibrationResult] = field(default_factory=dict)
    # Preserve even a measurement window with no arrivals.
    measurement_windows: List[dict] = field(default_factory=list)


def calibrate_workloads(spec: ExperimentSpec, target: BedrockConverseTarget) -> Dict[str, CalibrationResult]:
    """Resolve the model's token counter once (CountTokens > Converse
    usage > estimate), then size every workload's padding with it."""
    method, count_fn, notes = resolve_counter(spec.token_counting, [
        ("count_tokens", target.count_tokens),
        ("converse_usage", target.usage_input_tokens),
    ])
    if method is None:
        note = "; ".join(notes) or "no provider token counter available"
        return {w.name: estimate_profile(w, note) for w in spec.workloads}
    out = {}
    for w in spec.workloads:
        result = calibrate_profile(w, count_fn, method, tolerance_pct=spec.calibration_tolerance_pct)
        if notes and not result.note:
            result.note = "; ".join(notes)
        out[w.name] = result
    return out


# Quota-relative sweep values are rounded to 4 decimals (schema.sweep_values),
# so the 1.0x point (e.g. 6.6667 rps) sits a hair above the unrounded
# ceiling (6.66666...). Without this tolerance the 1.0x point -- the most
# useful candidate -- would never be confirmed.
_ROUNDING_TOLERANCE = 1e-4


def _value(point: SweepPoint) -> float:
    return point.concurrency if point.concurrency is not None else point.rps


def _candidates(points: List[SweepPoint], rec: Recommendation, spec: ExperimentSpec, subject: str,
                how_many: int) -> List[SweepPoint]:
    """The `how_many` highest points in discovery's leading non-failing
    run, returned ascending (the executor tests them highest-first).

    Rate sweep: only offered rps at or below the provider ceiling --
    production sustained_rps is capped at the quota anyway, so confirming
    above it adds nothing to the recommendation.

    Concurrency sweep: the provider ceiling is a PRIORITY rule, not a hard
    reject. Whether a point is safe is decided by independent confirmation
    (cooldown -> conditioning -> fixed-N looks), not by a heuristic: a
    point served above the ceiling may hold, or may have passed discovery
    on burst allowance -- confirmation finds out. But if none of the chosen
    points is at or below the ceiling (ceiling_ratio <= 1 + tolerance),
    the lowest one is swapped for the highest one that is, so a FAIL on
    burst-assisted points still leaves a sustainable fallback to confirm.
    """
    limit = rec.analysis.stable_pass_max
    eligible = [p for p in points if limit is not None and _value(p) <= limit]
    ceiling = spec.provider_ceilings.get(subject)
    if spec.sweep.type == "rate" and ceiling is not None and ceiling.rps:
        eligible = [p for p in eligible if _value(p) <= ceiling.rps + _ROUNDING_TOLERANCE]
    eligible.sort(key=_value)
    chosen = eligible[-how_many:]
    if spec.sweep.type == "concurrency" and ceiling is not None and ceiling.rps and how_many > 1 and chosen:
        sustainable = [p for p in eligible if not above_ceiling(p, ceiling.rps)]
        if sustainable and not any(p in sustainable for p in chosen):
            chosen = sorted(chosen[1:] + [sustainable[-1]], key=_value)
    return chosen


# Burst screen. provider_ceiling_rps is the NOMINAL sustainable quota
# ceiling, not an instantaneous hard wall: a point served at the ceiling
# reads a few % over it in a short window (short_chat C=4: 7.16 rps
# attempted vs 6.67 in discovery, then confirmed at 6.69 rps with 0
# throttles; long_context_short_answer C=5 confirmed at 6.74 rps, 1.01x).
# A point SERVED more than 10% above it may be running on burst
# allowance (those read 1.2-1.4x): it bounds refinement and loses
# candidate PRIORITY, but confirmation -- not this screen -- decides.
CEILING_RATE_TOLERANCE = 0.10


def point_rates(point: SweepPoint) -> Dict[str, Optional[float]]:
    """What a closed-loop point produced, kept apart -- under overload,
    fast 429s inflate the attempted rate far above anything served:
      attempted_rps    requests sent per second of window
      successful_rps   successes completed in the window per second (served)
      throttled_rps    429s per second of window
      slo_goodput_rps  successes that also met their latency SLO"""
    m = point.metrics
    d = m.measured_duration_s
    return {
        "attempted_rps": round(m.n / d, 4) if d else None,
        "successful_rps": m.request_throughput_rps,
        "throttled_rps": round(m.n_throttled / d, 4) if d else None,
        "slo_goodput_rps": m.slo_goodput_rps,
    }


def ceiling_ratio(point: SweepPoint, ceiling_rps: Optional[float]) -> Optional[float]:
    """successful_rps / provider_ceiling_rps -- how close to the nominal
    quota ceiling the point was SERVED."""
    served = point.metrics.request_throughput_rps
    return round(served / ceiling_rps, 4) if ceiling_rps and served is not None else None


# Provider-state suspect: throttled this hard while SERVED this far below
# the nominal ceiling -- the point's own load can't produce that, so it
# points at provider state (a preceding overload, other traffic on the
# account). With a recovery probe configured this is a CONTROL signal:
# the point's data is discarded, the provider is recovered, the point is
# re-measured.
SUSPECT_THROTTLE_RATE = 0.10
SUSPECT_CEILING_RATIO = 0.50


def suspect_point(point: SweepPoint, ceiling_rps: Optional[float]) -> bool:
    if point is None or not ceiling_rps:
        return False
    ratio = ceiling_ratio(point, ceiling_rps)
    return (point.metrics.throttle_rate >= SUSPECT_THROTTLE_RATE
            and ratio is not None and ratio < SUSPECT_CEILING_RATIO)


def above_ceiling(point: SweepPoint, ceiling_rps: float) -> bool:
    ratio = ceiling_ratio(point, ceiling_rps)
    return ratio is not None and ratio > 1 + CEILING_RATE_TOLERANCE


# Distinct seeds per phase, so no phase replays another's arrival pattern.
_SEED_OFFSET = {"discovery": 0, "refinement": 20_000, "confirmation": 10_000, "conditioning": 30_000,
                "recovery_probe": 40_000}


def _slo_kwargs(slo, *, latency: bool = True) -> dict:
    # Rate gates are always judged three-way at `confidence` (default 95%):
    # lower bound beyond the limit -> FAIL, upper bound within -> PASS, else INCONCLUSIVE.
    kwargs = dict(success_rate_min=slo.success_rate_min, throttle_rate_max=slo.throttle_rate_max,
                  confidence=slo.confidence or DEFAULT_CONFIDENCE)
    if latency:
        kwargs.update(ttft_p95_slo_ms=slo.ttft_p95_ms, latency_p95_slo_ms=slo.latency_p95_ms,
                      tpot_p95_slo_ms=slo.tpot_p95_ms)
    return kwargs


async def run_experiment(
    spec: ExperimentSpec, *, on_progress: Optional[ProgressCallback] = None,
    target: Optional[BedrockConverseTarget] = None,
    on_history_arm: Optional[Callable] = None,
) -> ExperimentReport:
    # `target` is injectable for tests (a fake client) -- never set by the CLI.
    owns_target = target is None
    if target is None:
        target = BedrockConverseTarget(
            model_id=spec.target.model_id, region=spec.target.region, transport=spec.transport,
        )
    try:
        return await _run(spec, target, on_progress, on_history_arm)
    finally:
        if owns_target:
            target.close()


async def _run(spec: ExperimentSpec, target: BedrockConverseTarget, on_progress: Optional[ProgressCallback],
               on_history_arm: Optional[Callable] = None) -> ExperimentReport:
    report = ExperimentReport(spec=spec)
    if spec.sweep.type not in ("concurrency", "rate"):
        raise ValueError(f"unknown sweep type: {spec.sweep.type!r} (use 'concurrency' or 'rate')")

    report.calibrations = calibrate_workloads(spec, target)
    profiles = {name: c.profile for name, c in report.calibrations.items()}
    subjects: List[Union[WorkloadProfile, WorkloadMix]]
    if spec.mix is not None:
        subjects = [WorkloadMix(
            name=spec.mix.name, entries=[(profiles[n], w) for n, w in spec.mix.weights.items()],
            assignment=spec.mix.assignment,
        )]
    else:
        subjects = [profiles[w.name] for w in spec.workloads]

    if subjects:
        offset = spec.workload_rotation_index % len(subjects)
        subjects = subjects[offset:] + subjects[:offset]

    for index, subject in enumerate(subjects):
        is_mix = isinstance(subject, WorkloadMix)
        shares = subject.shares if is_mix else None
        # SLOs: an isolated workload uses its own profile. A mix judges
        # each class against ITS profile -- per request for goodput, per
        # class for the gate; the blend is reported, never gated
        # (gate_kwargs is only the fallback for classes without an SLO).
        if is_mix:
            class_slos = {n: spec.slo_for(n) for n in shares}
            blend_slo = spec.slo
            metric_slo = dict(slo_by_workload={
                n: (c.ttft_p95_ms, c.latency_p95_ms, c.tpot_p95_ms) for n, c in class_slos.items()
            })
            gate_kwargs = _slo_kwargs(blend_slo, latency=False)
            class_gate = {n: _slo_kwargs(c) for n, c in class_slos.items()}
        else:
            blend_slo = spec.slo_for(subject.name)
            metric_slo = dict(ttft_slo_ms=blend_slo.ttft_p95_ms, latency_slo_ms=blend_slo.latency_p95_ms,
                              tpot_slo_ms=blend_slo.tpot_p95_ms)
            gate_kwargs = _slo_kwargs(blend_slo)
            class_gate = None
        confidence = blend_slo.confidence or DEFAULT_CONFIDENCE

        # Per sweep value, per phase: every repetition's results + window.
        # Discovery and confirmation data are kept strictly apart -- see
        # analysis/confirmation.py on why they're never pooled.
        acc: Dict[str, Dict[float, dict]] = {"discovery": {}, "refinement": {}, "confirmation": {}}

        async def measure(value: float, reps: int, phase: str) -> dict:
            state = acc[phase].setdefault(value, {"results": [], "windows": [], "per_rep": [], "peak": 0})
            offered_rps = value if spec.sweep.type == "rate" else None
            duration = spec.duration_s
            if phase == "confirmation" and spec.confirmation.continuous:
                duration = max(duration, spec.confirmation.min_steady_state_duration_s)
            for _ in range(reps):
                rep = len(state["windows"])
                # A distinct seed per repetition -- the same seed would
                # replay one identical arrival pattern R times, which
                # isn't R independent samples. Confirmation seeds are
                # offset so they never replay a discovery pattern.
                seed = None if spec.seed is None else spec.seed + rep + _SEED_OFFSET[phase]
                if spec.sweep.type == "concurrency":
                    runner = ConcurrencyRunner(
                        target, subject, concurrency=int(value), duration_s=duration,
                        warmup_s=spec.warmup_s, stream=spec.stream, seed=seed,
                        throttle_pause_s=spec.throttle_pause_s,
                    )
                else:
                    runner = RateRunner(
                        target, subject, rps=value, duration_s=duration, warmup_s=spec.warmup_s,
                        stream=spec.stream, seed=seed,
                    )
                target.reset_peak()
                results = await runner.run()
                state["peak"] = max(state["peak"], target.peak_outstanding)
                if spec.sweep.type == "concurrency" and target.peak_sdk_inflight > int(value):
                    validity["status"] = "invalid"
                    validity["events"].append({
                        "outcome": "concurrency_invariant_violated", "phase": phase,
                        "configured_concurrency": int(value),
                        "peak_sdk_inflight": target.peak_sdk_inflight,
                    })
                window = runner.window
                report.measurement_windows.append({
                    "subject": subject.name, "phase": phase, "value": value, "repetition": rep,
                    "start": window.start, "end": window.end,
                    "peak_sdk_inflight": target.peak_sdk_inflight,
                })
                for r in results:
                    r.tags.update({
                        # The runner already tagged the drawn class;
                        # `subject` differs from it only for a mix.
                        "subject": subject.name, "sweep_type": spec.sweep.type, "sweep_value": value,
                        "repetition": rep, "phase": phase,
                        # Persisted so the JSONL alone is enough to
                        # re-derive every metric with the same window.
                        "window_start": window.start, "window_end": window.end,
                        "measured": window.contains(r.scheduled_at),
                    })
                state["results"].extend(results)
                state["windows"].append(window)
                state["per_rep"].append(compute_run_metrics(
                    results, windows=[window], offered_rps=offered_rps, confidence=confidence, **metric_slo,
                ))
                report.all_results.extend(results)
                if validity["status"] == "invalid":
                    break
            return state

        async def condition(value: float, seconds: float) -> None:
            """Load at `value` for `seconds`, DISCARDED -- tagged
            phase=conditioning, measured=False, never in any metric."""
            seed = None if spec.seed is None else spec.seed + _SEED_OFFSET["conditioning"]
            if spec.sweep.type == "concurrency":
                runner = ConcurrencyRunner(target, subject, concurrency=int(value), duration_s=seconds, warmup_s=0.0,
                                           stream=spec.stream, seed=seed, throttle_pause_s=spec.throttle_pause_s)
            else:
                runner = RateRunner(target, subject, rps=value, duration_s=seconds, warmup_s=0.0,
                                    stream=spec.stream, seed=seed)
            for r in await runner.run():
                r.tags.update({"subject": subject.name, "sweep_type": spec.sweep.type, "sweep_value": value,
                               "phase": "conditioning", "measured": False})
                report.all_results.append(r)

        sub_ceiling = spec.provider_ceilings.get(subject.name)
        ceiling_rps = sub_ceiling.rps if sub_ceiling is not None else None
        rp = spec.recovery_probe
        validity: Dict = {"status": "valid", "events": [], "recovery_probes": []}
        baseline_ttft: Dict[str, Optional[float]] = {"ms": None}

        async def probe(reason: str, attempt: int) -> bool:
            """Short low-load probe (data discarded): is the provider healthy?"""
            seed = None if spec.seed is None else spec.seed + _SEED_OFFSET["recovery_probe"] + attempt
            runner = ConcurrencyRunner(target, subject, concurrency=rp.concurrency, duration_s=rp.duration_s,
                                       warmup_s=0.0, stream=spec.stream, seed=seed)
            results = await runner.run()
            for r in results:
                r.tags.update({"subject": subject.name, "sweep_type": spec.sweep.type, "phase": "recovery_probe",
                               "measured": False, "probe_reason": reason})
                report.all_results.append(r)
            m = compute_run_metrics(results, windows=[runner.window], confidence=confidence)
            ttft = m.ttft_p50_ms if m.ttft_p50_ms is not None else m.latency_p50_ms
            healthy = m.n > 0 and m.throttle_rate <= rp.max_throttle_rate and m.success_rate >= rp.min_success_rate
            if healthy and ttft and baseline_ttft["ms"] and ttft > baseline_ttft["ms"] * rp.max_ttft_ratio:
                healthy = False
            if healthy and ttft and baseline_ttft["ms"] is None:
                baseline_ttft["ms"] = ttft
            validity["recovery_probes"].append({
                "kind": "liveness", "reason": reason, "attempt": attempt, "n": m.n, "throttle_rate": m.throttle_rate,
                "success_rate": m.success_rate, "ttft_p50_ms": ttft, "healthy": healthy})
            return healthy

        baseline_capacity = None

        async def capacity_probe(reason: str, attempt: int) -> bool:
            nonlocal baseline_capacity
            if rp.baseline_fraction is None:
                return True
            if ceiling_rps is None or ceiling_rps <= 0:
                return False
            rate = ceiling_rps * rp.baseline_fraction
            duration = max(rp.baseline_duration_s, 1.25 * rp.baseline_min_requests / rate)
            runner = RateRunner(target, subject, rps=rate, duration_s=duration,
                                warmup_s=spec.warmup_s, stream=spec.stream,
                                seed=None if spec.seed is None else spec.seed + 50000 + attempt)
            rows = await runner.run()
            for r in rows:
                r.tags.update({"subject": subject.name, "phase": "baseline_capacity_probe",
                               "measured": False, "probe_reason": reason})
            report.all_results.extend(rows)
            m = compute_run_metrics(rows, windows=[runner.window], confidence=confidence, **metric_slo)
            healthy = (m.n >= rp.baseline_min_requests and m.throttle_rate <= rp.max_throttle_rate
                       and m.success_rate >= rp.min_success_rate
                       and m.slo_goodput_rps is not None
                       and m.slo_goodput_rps >= rate * rp.baseline_goodput_ratio)
            latency = m.ttft_p95_ms if spec.stream else m.latency_p95_ms
            limit = blend_slo.ttft_p95_ms if spec.stream else blend_slo.latency_p95_ms
            healthy = healthy and latency is not None and (limit is None or latency <= limit)
            if baseline_capacity is not None:
                goodput, previous_latency = baseline_capacity
                healthy = healthy and m.slo_goodput_rps is not None and m.slo_goodput_rps >= goodput * rp.baseline_goodput_ratio
                healthy = healthy and latency is not None and latency <= previous_latency * rp.max_ttft_ratio
            if healthy and baseline_capacity is None:
                baseline_capacity = (m.slo_goodput_rps, latency)
            validity["recovery_probes"].append({"kind": "baseline_capacity", "reason": reason,
                "attempt": attempt, "offered_rps": rate, "duration_s": duration, "n": m.n,
                "goodput_rps": m.slo_goodput_rps, "latency_p95_ms": latency,
                "throttle_rate": m.throttle_rate, "healthy": healthy,
                "scope": "recovery control; does not certify candidate capacity"})
            return healthy

        async def recover(cooldown_s: float, reason: str) -> bool:
            """Fixed recovery interval, then -- with a recovery probe --
            VERIFY recovery: probe, and if unhealthy wait and re-probe up to
            max_attempts. False = the provider never looked healthy."""
            if cooldown_s > 0:
                await asyncio.sleep(cooldown_s)
            if rp is None:
                return True
            consecutive_healthy = 0
            for attempt in range(1, rp.max_attempts + 1):
                if attempt > 1 and rp.retry_cooldown_s > 0:
                    await asyncio.sleep(rp.retry_cooldown_s)
                if await probe(reason, attempt) and await capacity_probe(reason, attempt):
                    consecutive_healthy += 1
                    if consecutive_healthy >= rp.required_consecutive_healthy:
                        return True
                else:
                    consecutive_healthy = 0
            validity["status"] = "invalid"
            validity["events"].append({"reason": reason, "outcome": "provider_unrecovered",
                                       "probes": rp.max_attempts})
            return False

        def invalidate(phase: str, value: float) -> None:
            """Drop a phase's data at `value`: it was measured in a suspect
            provider state. Kept in the raw JSONL, tagged, never measured."""
            state = acc[phase].pop(value, None)
            for w in report.measurement_windows:
                if w["subject"] == subject.name and w["phase"] == phase and w["value"] == value:
                    w["phase"] = "invalidated"
            for r in (state or {}).get("results", []):
                r.tags.update({"phase": "invalidated", "invalidated_phase": phase, "measured": False})

        def note_suspect(phase: str, value: float, point: SweepPoint, outcome: str) -> None:
            validity["events"].append({
                "phase": phase, "value": value, "throttle_rate": point.metrics.throttle_rate,
                "ceiling_ratio": ceiling_ratio(point, ceiling_rps), "outcome": outcome})
            if outcome == "reproduced_after_recovery" and validity["status"] in (
                    "valid", "suspect_steady_state", "suspect_non_monotonic"):
                validity["status"] = "suspect_reproduced"

        async def measure_valid(value: float, phase: str) -> Optional[SweepPoint]:
            """measure + build, with the suspect signature as a control
            signal: discard, recover, re-measure once. None = the provider
            never recovered (the subject's measurement is invalid)."""
            await measure(value, spec.repetitions, phase)
            if validity["status"] == "invalid":
                return None
            point = build(value, phase)
            if spec.sweep.stop_after_clear_fail and (point_verdict(point, class_gate, **gate_kwargs).verdict == FAIL
                                                     or severe_throttling(point.metrics)):
                return point
            if rp is None or not suspect_point(point, ceiling_rps):
                return point
            if not await recover(0.0, f"suspect {phase} point {value:g}"):
                note_suspect(phase, value, point, "provider_unrecovered")
                return None
            invalidate(phase, value)
            await measure(value, spec.repetitions, phase)
            if validity["status"] == "invalid":
                return None
            again = build(value, phase)
            note_suspect(phase, value, point,
                         "reproduced_after_recovery" if suspect_point(again, ceiling_rps) else "cleared_after_recovery")
            return again

        def build(value: float, phase: str, conf: Optional[float] = None,
                  subset: Optional[List[RequestResult]] = None) -> SweepPoint:
            """Metrics from ONE phase's data; bounds at `conf` (default:
            the SLO's confidence; confirmation uses the per-test one).
            `subset` replaces the phase's results -- a confirmation look's
            exact first-N sample."""
            state = dict(acc[phase][value])
            if subset is not None:
                state["results"] = subset
            point_conf = conf if conf is not None else confidence
            offered_rps = value if spec.sweep.type == "rate" else None
            class_metrics = {}
            if shares is not None:
                for class_name, share in shares.items():
                    own = [r for r in state["results"] if r.tags.get("workload") == class_name]
                    c = class_slos[class_name]
                    class_metrics[class_name] = compute_run_metrics(
                        own, windows=state["windows"], offered_rps=None if offered_rps is None else offered_rps * share,
                        ttft_slo_ms=c.ttft_p95_ms, latency_slo_ms=c.latency_p95_ms, tpot_slo_ms=c.tpot_p95_ms,
                        confidence=conf if conf is not None else (c.confidence or DEFAULT_CONFIDENCE),
                    )
            return SweepPoint(
                concurrency=int(value) if spec.sweep.type == "concurrency" else None,
                rps=value if spec.sweep.type == "rate" else None,
                metrics=compute_run_metrics(
                    state["results"], windows=state["windows"], offered_rps=offered_rps, confidence=point_conf,
                    **metric_slo,
                ),
                repetitions=state["per_rep"],
                class_metrics=class_metrics,
                peak_outstanding=state["peak"],
                client_limited=state["peak"] > target.executor_workers,
                phase=phase,
            )

        if spec.history_protocol is not None:
            from .history import run_history_comparison
            history = await run_history_comparison(
                spec, target, subject, report.all_results, recover, on_progress,
                **({"on_history_arm": on_history_arm} if on_history_arm is not None else {}))
            report.profiles.append(ProfileReport(workload_name=subject.name, measurement_validity=validity,
                                                 history_comparison=history))
            continue

        # Start of the subject: a recovery interval after the previous
        # subject (all share one model's quota), then -- with a probe --
        # a verified healthy baseline before any discovery point.
        healthy = await recover(spec.inter_subject_cooldown_s if index > 0 else 0.0,
                                "start of subject" if index == 0 else "after previous subject")

        # Phase 1 -- discovery: every value, spec.repetitions each.
        values = spec.sweep_values(subject.name) if healthy else []
        points: List[SweepPoint] = []
        consecutive_fails = 0
        for value in values:
            if spec.burst_protocol:
                pulse_started = time.perf_counter()
                await measure(value, 1, "discovery")
                point = build(value, "discovery")
                state = acc["discovery"][value]
                window = state["windows"][-1]
                throttles = [r.started_at for r in state["results"]
                             if r.throttled and window.contains(r.scheduled_at)]
                recovered = await recover(rp.retry_cooldown_s, f"after burst {value:g}")
                validity.setdefault("burst_results", []).append({
                    "offered_rps": value, "ceiling_multiple": value / ceiling_rps,
                    "burst_duration_s": window.end - window.start,
                    "throttle_onset_s": max(0, min(throttles) - window.start) if throttles else None,
                    "throttle_onset_censored": not bool(throttles),
                    "recovery_observed_after_s": max(0.0, time.perf_counter() - pulse_started - spec.warmup_s - spec.duration_s),
                    "recovery_time_censored": not recovered, "recovered": recovered,
                    "recovery_definition": "first successful liveness and baseline-capacity probe; includes drain, cooldown and probe time",
                })
                points.append(point)
                if on_progress is not None:
                    on_progress(subject.name, value, point)
                if not recovered:
                    break
                continue
            point = await measure_valid(value, "discovery")
            if point is None:
                break  # provider never recovered: stop -- no conclusion from this subject
            points.append(point)
            if on_progress is not None:
                on_progress(subject.name, value, point)
            consecutive_fails = consecutive_fails + 1 if point_verdict(point, class_gate, **gate_kwargs).verdict == FAIL else 0
            if spec.sweep.stop_after_clear_fail and (consecutive_fails or severe_throttling(point.metrics)):
                validity["events"].append({"phase": "discovery", "value": value,
                    "outcome": "stopped_clear_fail" if consecutive_fails else "stopped_severe_throttling"})
                break
            if spec.sweep.stop_after_fails is not None and consecutive_fails >= spec.sweep.stop_after_fails:
                break  # saturation seen stop_after_fails times in a row -- the rest is past it

        # Phase 1b -- boundary refinement (concurrency only): bisect
        # between the last non-failing point L and the first FAIL F, so
        # the candidate is the real edge rather than the coarse grid
        # point below it. Still discovery-class data: selects, never confirms.
        if spec.sweep.refinement is not None and validity["status"] != "invalid":
            # Upper bound: the first FAIL -- or, with a known ceiling, the
            # first point whose achieved rate is above it (burst, not
            # sustainable). Lower bound: the highest point below that.

            def upper(p: SweepPoint) -> bool:
                return (point_verdict(p, class_gate, **gate_kwargs).verdict == FAIL
                        or (ceiling_rps is not None and above_ceiling(p, ceiling_rps)))

            lo = hi = None
            for p in sorted(points, key=_value):
                if upper(p):
                    hi = _value(p)
                    break
                lo = _value(p)
            for _ in range(spec.sweep.refinement.max_points):
                if lo is None or hi is None:
                    break
                if spec.sweep.refinement.stop_when_adjacent and hi - lo <= 1:
                    break
                if (int(lo) + int(hi)) // 2 in (lo, hi):
                    break  # no integer strictly between the bounds -- nothing new to test
                # Recovery before EVERY refinement point: the first follows
                # the coarse sweep's overload, each later one may follow a
                # refinement point that just FAILed -- and refinement exists
                # to locate the boundary precisely.
                mid = (int(lo) + int(hi)) // 2
                if not await recover(spec.sweep.refinement.cooldown_s, f"before refinement point {mid}"):
                    break
                point = await measure_valid(mid, "refinement")
                if point is None:
                    break
                points.append(point)
                if on_progress is not None:
                    on_progress(subject.name, mid, point)
                if upper(point):
                    hi = mid
                else:
                    lo = mid
            points.sort(key=_value)

        # Phase 2 -- confirmation (analysis/confirmation.py): fresh,
        # independent repetitions at candidates chosen from discovery,
        # PASS and FAIL only at pre-planned looks, a severe-throttling
        # early stop and caps -> INCONCLUSIVE. Discovery data is not reused here.
        recommendation = recommend(points, class_gate, **gate_kwargs)
        plan = None
        confirmations: List[ConfirmationResult] = []
        if spec.confirmation is not None and recommendation is not None and validity["status"] != "invalid":
            cfg = spec.confirmation
            # Candidates come from discovery ALONE, before any confirmation
            # data -- so alpha can be split over exactly these K.
            candidates = _candidates(points, recommendation, spec, subject.name, cfg.candidates)
            if cfg.values_by_workload is not None:
                requested = cfg.values_by_workload.get(subject.name, [])
                candidates = sorted([p for p in points if _value(p) in requested
                                     and recommendation.analysis.stable_pass_max is not None
                                     and _value(p) <= recommendation.analysis.stable_pass_max], key=_value)
            plan = plan_looks(
                limits_for(gate_kwargs, class_gate, shares), confidence=gate_kwargs["confidence"],
                max_looks=cfg.max_looks, max_repetitions=cfg.max_repetitions,
                max_requests=cfg.max_requests, max_duration_s=cfg.max_duration_s,
                candidates=len(candidates), min_steady_state_duration_s=cfg.min_steady_state_duration_s,
            )
            plan.order = cfg.order
            gate_look = {**gate_kwargs, "confidence": plan.per_test_confidence}
            class_look = None if class_gate is None else {
                n: {**kw, "confidence": plan.per_test_confidence} for n, kw in class_gate.items()
            }
            confirmation_duration = max(spec.duration_s, cfg.min_steady_state_duration_s) if cfg.continuous else spec.duration_s
            per_rep_s = spec.warmup_s + confirmation_duration
            started = time.perf_counter()
            stopped = False
            for disc in (candidates if cfg.order == "ascending" else reversed(candidates)):
                value = _value(disc)
                if stopped:
                    confirmations.append(ConfirmationResult(value, INCONCLUSIVE, "not_tested"))
                    continue
                est_per_rep = (value * confirmation_duration if spec.sweep.type == "rate"
                               else disc.metrics.n / max(1, len(disc.repetitions)) * confirmation_duration / spec.duration_s)
                result = None
                looks_used = 0
                passed_look = None
                last_look = None
                # Every tested candidate: recovery (interval + verified
                # healthy probe) -> conditioning -> looks, so none inherits
                # what ran before it and conditioning never starts on a
                # throttled provider. None of it counts against
                # max_duration_s: shift its clock.
                recovery_start = len(validity["recovery_probes"])
                candidate_state = {"status": "unverified", "recovery_checks": []}
                paused_from = time.perf_counter()
                if not await recover(cfg.cooldown_s, f"before candidate {value:g}"):
                    candidate_state.update(status="unrecovered", recovery_checks=list(validity["recovery_probes"][recovery_start:]))
                    confirmations.append(ConfirmationResult(value, INCONCLUSIVE, "provider_state_invalid",
                                                            provider_state=candidate_state))
                    stopped = True  # the provider never recovered: no further candidate is meaningful
                    continue
                candidate_state.update(
                    status="healthy_observed" if rp is not None and rp.baseline_fraction is not None else "unverified",
                    recovery_checks=list(validity["recovery_probes"][recovery_start:]),
                    checked_at_unix_s=time.time(),
                )
                if cfg.warmup_s > 0:
                    await condition(value, cfg.warmup_s)  # discarded: steady state before the looks
                started += time.perf_counter() - paused_from
                caps = candidate_caps(plan, cfg.max_requests, cfg.max_duration_s, est_requests_per_rep=est_per_rep,
                                      per_rep_s=per_rep_s, measured_per_rep_s=confirmation_duration)
                if caps["per_candidate"]:
                    started = time.perf_counter()  # an `auto` budget is this candidate's own
                requests_cap, duration_cap = caps["max_requests"], caps["max_duration_s"]
                suspect_retries = 1 if rp is not None else 0
                if not reachable(replace(plan, max_requests=requests_cap), est_requests_per_rep=est_per_rep,
                                 per_rep_s=per_rep_s,
                                 remaining_duration_s=duration_cap - (time.perf_counter() - started)):
                    result = ConfirmationResult(value, INCONCLUSIVE, "unreachable_within_caps",
                                                next_look_n=plan.look_schedule[0], caps=caps)
                while result is None:
                    state = await measure(value, 1, "confirmation")
                    if validity["status"] == "invalid":
                        result = ConfirmationResult(value, INCONCLUSIVE, "provider_state_invalid")
                        break
                    point = build(value, "confirmation", conf=plan.per_test_confidence)
                    if on_progress is not None:
                        on_progress(subject.name, value, point)
                    if suspect_retries and suspect_point(point, ceiling_rps):
                        # Control signal: throttled far below the ceiling --
                        # possible provider-state anomaly; cause remains unresolved.
                        # Discard its confirmation data, recover, restart.
                        suspect_retries -= 1
                        paused_from = time.perf_counter()
                        ok = await recover(0.0, f"suspect confirmation at {value:g}")
                        note_suspect("confirmation", value, point,
                                     "restarted_after_recovery" if ok else "provider_unrecovered")
                        invalidate("confirmation", value)
                        candidate_state.update(
                            status="anomaly_observed" if ok else "unrecovered",
                            recovery_checks=list(validity["recovery_probes"][recovery_start:]),
                            checked_at_unix_s=time.time(),
                        )
                        if not ok:
                            result = ConfirmationResult(value, INCONCLUSIVE, "provider_state_invalid")
                            break
                        if cfg.warmup_s > 0:
                            await condition(value, cfg.warmup_s)
                        started += time.perf_counter() - paused_from
                        looks_used = 0
                        passed_look = last_look = None
                        continue
                    verdict = point_verdict(point, class_look, **gate_look)
                    n, reps = point.metrics.n, len(state["windows"])
                    class_n = {name: m.n for name, m in point.class_metrics.items()} if is_mix else None
                    looked: Dict[int, tuple] = {}

                    def look_verdict(j: int) -> Verdict:
                        # Fixed-count look: EXACTLY the first N_j measured
                        # requests, however many this repetition produced.
                        measured = [r for r in state["results"] if r.tags.get("measured")]
                        sample = look_sample(measured, plan.look_sizes(j))
                        look_point = build(value, "confirmation", conf=plan.per_test_confidence, subset=sample)
                        v_look = point_verdict(look_point, class_look, **gate_look)
                        looked[j] = (look_point, v_look, len(sample))
                        return v_look

                    # A fixed-count PASS is retained while the candidate finishes
                    # its minimum exposure. No repeated statistical tests on a
                    # growing sample: the additional checks can only veto PASS.
                    decision = None if passed_look is not None else step(
                        verdict, n, looks_used, plan, class_n, look_verdict)
                    if looked:
                        looks_used = max(looked) + 1
                        last_look = looked[max(looked)]
                    v, reason = INCONCLUSIVE, None
                    if decision is not None:
                        v, reason, looks_used = decision
                        if v == PASS:
                            passed_look = last_look
                    measured_s = point.metrics.measured_duration_s
                    sanity = []
                    if passed_look is not None:
                        measured = [r for r in state["results"] if r.tags.get("measured")]
                        sample_ids = {id(r) for r in look_sample(measured, plan.look_sizes(looks_used - 1))}
                        subsets = {
                            "all_collected": measured,
                            "latest_window": [r for r in measured if r.tags["repetition"] == reps - 1],
                            "post_look": [r for r in measured if id(r) not in sample_ids],
                        }
                        for scope, rows in subsets.items():
                            if not rows:
                                continue
                            check_point = build(value, "confirmation", conf=plan.per_test_confidence, subset=rows)
                            # An absent class in a tail/window is not evidence
                            # of a violation; its full fixed-count look still gates PASS.
                            check_point.class_metrics = {name: m for name, m in check_point.class_metrics.items()
                                                         if m.n > 0}
                            check = point_verdict(check_point, class_look, **gate_look)
                            severe = any(severe_throttling(m) for m in
                                         [check_point.metrics, *check_point.class_metrics.values()])
                            if check.verdict == FAIL or severe:
                                sanity.append({"scope": scope, "n": check_point.metrics.n,
                                               "n_throttled": check_point.metrics.n_throttled,
                                               "throttle_rate": check_point.metrics.throttle_rate,
                                               "failed_checks": [c.name for c in check.checks if c.verdict == FAIL],
                                               "severe_throttling": severe})
                        if sanity:
                            v, reason = INCONCLUSIVE, "post_look_violation"
                            validity["events"].append({"phase": "confirmation", "value": value,
                                                       "outcome": reason, "violations": sanity})
                            if validity["status"] == "valid":
                                validity["status"] = "suspect_steady_state"
                        elif measured_s + 1e-9 >= cfg.min_steady_state_duration_s:
                            v, reason = PASS, "confirmed"
                        else:
                            v, reason = INCONCLUSIVE, None
                    if reason is None:
                        remaining = duration_cap - (time.perf_counter() - started)
                        rep_cap = cfg.max_repetitions if cfg.max_repetitions is not None else math.inf
                        if severe_throttling(point.metrics):
                            reason = "stopped_severe_throttling"
                        elif reps >= rep_cap:
                            reason = "max_repetitions"
                        elif n >= requests_cap:
                            reason = "max_requests"
                        elif remaining < per_rep_s:
                            reason = "max_duration"
                        elif passed_look is None:
                            more_reps = min(rep_cap - reps, int(remaining // per_rep_s))
                            max_n = min(requests_cap, n + (n / reps) * more_reps)
                            if max_n < plan.look_schedule[looks_used]:
                                reason = "unreachable_within_caps"
                    if reason is not None:
                        if rp is not None and reason in ("violation_demonstrated", "stopped_severe_throttling", "post_look_violation") \
                                and suspect_point(point, ceiling_rps):
                            # Still throttled far below the ceiling after a
                            # verified recovery: accepted as a real result.
                            note_suspect("confirmation", value, point, "reproduced_after_recovery")
                        result = ConfirmationResult(
                            value, v, reason, repetitions=reps, n=n, looks_used=looks_used,
                            next_look_n=plan.look_schedule[looks_used] if looks_used < plan.max_looks else None,
                            # A look's verdict when a look decided; the all-data one for a stop / cap.
                            detail=last_look[1] if last_look and reason in (
                                "confirmed", "violation_demonstrated", "looks_exhausted") else verdict,
                            point=point,
                            decision_n=last_look[2] if last_look else None,
                            decision_metrics=last_look[0].metrics if last_look else None,
                            caps=caps,
                            steady_state={
                                "required_duration_s": cfg.min_steady_state_duration_s,
                                "continuous": cfg.continuous,
                                "longest_continuous_window_s": max(w.end - w.start for w in state["windows"]),
                                "measured_duration_s": measured_s,
                                "minimum_duration_met": measured_s + 1e-9 >= cfg.min_steady_state_duration_s,
                                "statistical_look_passed": passed_look is not None,
                                "violations": sanity,
                            },
                        )
                result.provider_state = candidate_state
                confirmations.append(result)
                stopped = (result.stop_reason == "provider_state_invalid"
                           or (cfg.order == "highest_first" and result.verdict == PASS)
                           or (cfg.stop_after_fail and result.verdict != PASS))
            confirmed = highest_confirmed(confirmations)
            recommendation.confirmed_point = confirmed.point if confirmed is not None else None
            recommendation.confirmation_source = "confirmation"

        analysis = analyze_sweep(points, class_gate, **gate_kwargs)
        if analysis.status == "unresolved":
            validity["events"].append({"phase": "discovery", "outcome": "non_monotonic",
                                       "unstable_region": analysis.unstable_region})
            if validity["status"] == "valid":
                validity["status"] = "suspect_non_monotonic"
        report.profiles.append(ProfileReport(
            workload_name=subject.name, points=points, mix_shares=shares,
            recommendation=recommendation,
            analysis=analysis,
            confirmation_plan=plan,
            confirmations=confirmations,
            verdicts=[point_verdict(p, class_gate, **gate_kwargs) for p in points],
            measurement_validity=validity,
        ))

    return report
