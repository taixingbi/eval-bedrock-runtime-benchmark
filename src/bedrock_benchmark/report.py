"""Builds the capacity-profile.yaml artifact (schema_version 5) -- the
one machine-readable thing this repo exists to hand to
eval-bedrock-gateway's own control-plane config review, not a
human-facing HTML report.

schema_version 2 fixes two real bugs schema_version 1 had:

1. Rate-sweep results were written into the SAME `saturation_concurrency`
   field a concurrency sweep uses, silently mislabeling an RPS value as
   a concurrency value -- and provider_headroom was only ever applied
   to `.concurrency`, so a rate sweep's recommended RPS was never
   headroom-adjusted at all. Rate and concurrency results now live in
   their own `rate`/`workload_classes.<name>.rate` and
   `.concurrency` sub-blocks with their own headroom-adjusted
   `production_rps`/`production_max`, never sharing a field name.
2. `global_max_concurrency = max(concurrencies)` across independently-
   swept workload classes doesn't mean anything: a real MIXED workload
   (some short traffic + some long traffic concurrently) can exceed
   safe backend capacity well before either class's own isolated
   measured max would predict. There is no such thing as a
   scientifically defensible "global max concurrency" derived from
   per-class isolated sweeps alone -- it needs its own dedicated
   mixed-workload experiment (not yet built). So this artifact reports
   ONLY per-class envelopes now; a gateway's own global concurrency
   config is the gateway's decision to make from these, not something
   this repo pre-packages for it.

schema_version 3 fixes a field-semantics bug schema_version 2 still had:

3. The rate block's `measured_sustainable_rps` was the best point's SLO
   GOODPUT, and `production_rps` applied headroom to that goodput --
   but a gateway admission limit is set on OFFERED load, not on the
   fraction of it that came back within SLO. Those are three different
   numbers now: `max_safe_offered_rps` (the swept rate that passed),
   `slo_goodput_rps` (what it actually delivered within SLO), and
   `production_offered_rps` (headroom applied to the OFFERED rate --
   the one a gateway config should read).

It also records whether each workload class actually measured the
shape it claims (`workload_validation`: requested vs Bedrock-reported
input tokens -- i.e. whether the 4-chars/token padding estimate held
for this model), and -- for a `mix:` experiment -- a `mixed_workloads`
envelope, the only cross-class number this artifact ever reports.

It also records how the numbers were measured (`measurement`: warmup,
window, repetitions, confidence) and per-class `evidence` (sample
size, throttle count, confidence bounds), so a reader can tell a
statistically resolved 0.1% throttle SLO from an unresolved one.

schema_version 4 adds, per sweep subject: `provider_constraints` (RPM-
vs TPM-bound request ceiling, see ceiling.py) and the rps a quota-
relative sweep actually resolved to; `saturation_status` (resolved /
not_reached / unresolved -- a non-monotonic sweep claims no saturation);
input AND output token validation; per-class SLO profiles; and client
integrity evidence (`peak_outstanding`, `client_limited_points`,
`transport.executor_workers`).

schema_version 5 groups the quota snapshot and the SLO profiles under
one `constraints:` block (quota: what the provider allows; slo: what
quality we require), mirroring constraints/quota.yaml and
constraints/slo.yaml -- replacing v4's top-level quota_snapshot/slo/
slo_profiles.

See this repo's own README for the boundary this draws: this repo
outputs a safe operating envelope per workload class; it never
implements or pre-decides a gateway's global/tenant/AIMD control
policy.
"""
from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Optional

from .analysis.capacity import FAIL, INCONCLUSIVE, PASS, Recommendation
from .analysis.metrics import DEFAULT_CONFIDENCE, RunMetrics, min_samples_to_resolve_rate, percentile
from .experiments.executor import ExperimentReport, above_ceiling, ceiling_ratio, point_rates, suspect_point
from .recommendation import admission_envelope
from .results import RequestResult
from .workload import WorkloadProfile


def _observed_tokens(results: List[RequestResult]) -> dict:
    """Real, measured p50 input/output tokens for this workload class
    -- distinct from WorkloadProfile's own input_tokens/output_tokens
    (the TARGET the prompt generator aimed for). Comparing the two is
    the cheapest sanity check that a workload class actually measured
    what it claims to have measured."""
    input_tokens = [r.input_tokens for r in results if r.success and r.input_tokens is not None]
    output_tokens = [r.output_tokens for r in results if r.success and r.output_tokens is not None]
    return {
        "input_tokens_p50": round(percentile(input_tokens, 50), 1) if input_tokens else None,
        "output_tokens_p50": round(percentile(output_tokens, 50), 1) if output_tokens else None,
    }


def _saturation(rec: Recommendation) -> dict:
    """The first FAIL the DISCOVERY sweep (incl. refinement) observed --
    an observed edge, never a statistically confirmed one; the capacity
    is statistically_confirmed alone. status is discovery_resolved /
    discovery_not_reached / discovery_unresolved; for a non-monotonic
    sweep no edge is claimed and the unstable region is described
    instead (see analysis/capacity.py's SweepAnalysis)."""
    a = rec.analysis
    sat = rec.saturation_point
    out: dict = {"observed_edge": _value(sat) if sat is not None else None,
                 "phase": sat.phase if sat is not None else None,
                 "status": f"discovery_{a.status}"}
    if a.status == "unresolved":
        out.update(unstable_region=a.unstable_region, confirmed_fail_from=a.confirmed_fail_from)
    return out


def _value(point) -> float:
    return point.concurrency if point.concurrency is not None else point.rps


def _observed_fields(rec: Recommendation) -> dict:
    """What was OBSERVED: the best point with no FAIL before the first
    failure. Its verdict may be INCONCLUSIVE -- no violation seen, not
    enough requests to prove the SLO -- which is exactly why it is never
    used to derive a production value."""
    out: dict = {"observed_verdict": rec.verdict.verdict}
    if rec.verdict.inconclusive_checks:
        out["observed_inconclusive_checks"] = [c.to_dict() for c in rec.verdict.inconclusive_checks]
    return out


def _concurrency_block(rec: Recommendation, *, ceiling_rps: Optional[float] = None) -> dict:
    """MEASUREMENT only: observed_nonfailing -> statistically_confirmed.
    The policy step (headroom -> max_inflight) lives in the entry's
    `recommendation` block (recommendation.py), never here.
    observed_nonfailing_rates are what that point produced (attempted /
    successful / throttled / goodput); ceiling_ratio is successful_rps
    over the nominal provider ceiling -- well above 1 means burst."""
    confirmed = rec.confirmed_point.concurrency if rec.confirmed_point is not None else None
    out = {
        "observed_nonfailing": rec.point.concurrency,
        **_observed_fields(rec),
        "observed_nonfailing_rates": point_rates(rec.point),
        "statistically_confirmed": confirmed,
        "confirmation_source": rec.confirmation_source,
        "saturation": _saturation(rec),
        "observed_slo_goodput_rps": rec.point.metrics.slo_goodput_rps,
    }
    if ceiling_rps:
        out["provider_ceiling_rps"] = round(ceiling_rps, 4)  # nominal sustainable quota ceiling, not a hard wall
        out["observed_nonfailing_ceiling_ratio"] = ceiling_ratio(rec.point, ceiling_rps)
        out["observed_nonfailing_above_provider_ceiling"] = above_ceiling(rec.point, ceiling_rps)
        if rec.confirmed_point is not None:
            # Confirmed while SERVED above the nominal ceiling: it held for
            # the confirmation window (which can be short -- bronze's first
            # look is ~a minute), not proof it holds indefinitely; the quota
            # still caps sustained production rate (capacity-reference-rate).
            out["confirmed_ceiling_ratio"] = ceiling_ratio(rec.confirmed_point, ceiling_rps)
            out["confirmed_above_provider_ceiling"] = above_ceiling(rec.confirmed_point, ceiling_rps)
    return out


def _rate_block(rec: Recommendation, *, ceiling_rps: Optional[float]) -> dict:
    """MEASUREMENT only -- two distinct numbers, never conflated:

        observed_nonfailing_offered_rps      no FAIL observed (may be INCONCLUSIVE)
        statistically_confirmed_offered_rps  strictly PASS at the configured confidence

    A rate sweep deliberately goes above quota (to see throttling and
    burst behavior), and a short window can pass there on Bedrock's
    burst allowance -- measured_burst_ceiling_rps records that, but it
    is observed serving, not a sustainable rate. The policy step
    (headroom + quota cap -> sustained_rps) is the entry's
    `recommendation` block (recommendation.py), never here.
    """
    confirmed = rec.confirmed_point.rps if rec.confirmed_point is not None else None
    out = {
        "observed_nonfailing_offered_rps": rec.point.rps,
        **_observed_fields(rec),
        "observed_slo_goodput_rps": rec.point.metrics.slo_goodput_rps,
        "statistically_confirmed_offered_rps": confirmed,
        "confirmation_source": rec.confirmation_source,  # confirmation | discovery_fixed_sequence
        "confirmed_slo_goodput_rps": rec.confirmed_point.metrics.slo_goodput_rps if rec.confirmed_point else None,
        # Highest swept rate that didn't FAIL anywhere in the sweep --
        # observed short-window serving, possibly above quota on burst.
        "measured_burst_ceiling_rps": rec.burst_point.rps if rec.burst_point is not None else None,
        "provider_ceiling_rps": round(ceiling_rps, 4) if ceiling_rps else None,
        "saturation": _saturation(rec),
    }
    return out


def _observed_mix(counts: Dict[str, int]) -> dict:
    total = sum(counts.values())
    return {"n": total, "counts": counts,
            "shares": {name: n / total if total else None for name, n in counts.items()}}


def _sweep_points(profile_report, ceiling_rps: Optional[float] = None) -> List[dict]:
    """Every swept point's verdict -- the transition region at a glance.
    Concurrency points carry their rates -- attempted (inflated by fast
    429s under overload), successful (served), throttled, goodput -- and
    ceiling_ratio (served / nominal ceiling); one served >10% above the
    ceiling may be on burst allowance -- flagged, and left to
    confirmation to decide."""
    out = []
    for point, verdict in zip(profile_report.points, profile_report.verdicts):
        row = {"value": _value(point), "verdict": verdict.verdict, "phase": point.phase,
               "repetitions": len(point.repetitions) or 1, "n": point.metrics.n}
        if point.class_metrics:
            row["observed_mix"] = _observed_mix({name: m.n for name, m in point.class_metrics.items()})
        if point.concurrency is not None:
            row.update(point_rates(point))
            if ceiling_rps:
                row["ceiling_ratio"] = ceiling_ratio(point, ceiling_rps)
            if ceiling_rps and above_ceiling(point, ceiling_rps):
                row["above_provider_ceiling"] = True
        failed = [c.name for c in verdict.checks if c.verdict == "FAIL"]
        if failed:
            row["failed"] = failed
        inconclusive = verdict.inconclusive_checks
        if inconclusive:
            row["inconclusive"] = [f"{c.name}: n={c.n} < required_n={c.required_n}" for c in inconclusive]
        out.append(row)
    return out


_STOP_HINTS = {
    "post_look_violation": "a fixed-count look passed, but collected evidence violated the steady-state checks; "
                           "candidate not confirmed",
    "violation_demonstrated": "fresh data at a planned look demonstrated an SLO violation there (lower bound "
                              "over the limit) -- a lower candidate may confirm (candidates > 1)",
    "looks_exhausted": "every planned look was neither a demonstrated PASS nor a demonstrated FAIL",
    "stopped_severe_throttling": "stopped early: severely throttled (an operational guard -- not a statistical "
                                 "FAIL, not a saturation edge)",
    "max_repetitions": "raise the confirmation caps to collect more samples",
    "max_requests": "raise the confirmation caps to collect more samples",
    "max_duration": "raise the confirmation caps to collect more samples",
    "unreachable_within_caps": "its next look can't be reached within the confirmation caps -- raise them",
}


_LATENCY_CHECKS = ("ttft_p95", "tpot_p95", "latency_p95")


def _summary(rec: Recommendation) -> str:
    """The three numbers side by side, so an INCONCLUSIVE observed point
    between capacity and saturation reads as 'not proven' -- never as
    'unsafe', and never as the capacity."""
    confirmed = _value(rec.confirmed_point) if rec.confirmed_point is not None else None
    observed = _value(rec.point)
    parts = [f"statistically_confirmed={confirmed if confirmed is not None else 'none'} (the capacity)"]
    if confirmed is None or observed != confirmed:
        note = {"INCONCLUSIVE": "INCONCLUSIVE -- no violation seen, too few requests to prove the SLO; not shown unsafe",
                "PASS": "PASS in discovery, not confirmed on independent data"}.get(rec.verdict.verdict, rec.verdict.verdict)
        parts.append(f"observed_nonfailing={observed} ({note})")
    sat = _value(rec.saturation_point) if rec.saturation_point is not None else None
    parts.append(f"saturation={sat} (first FAIL in discovery -- an observed edge, not confirmed)" if sat is not None
                 else f"saturation=discovery_{rec.analysis.status}")
    return " | ".join(parts)


def _diagnosis(rec: Recommendation, profile_report, ceiling) -> dict:
    """What limits capacity -- read from the saturation point's FAILED
    checks, not guessed from one number:

      provider_throttling     throttling observed; quota mechanism unproven
      latency                 a latency check failed, no throttling
      provider_throttling_and_latency  both
      errors                  non-throttle failures
      not_reached / unresolved  no clean saturation point to read
    """
    latency_at_observed = {
        c.name: {"observed": c.observed, "threshold": c.threshold, "verdict": c.verdict}
        for c in rec.verdict.checks if c.name in _LATENCY_CHECKS
    }
    out: dict = {"latency_at_observed_nonfailing": latency_at_observed,
                 # None when no latency check is configured -- not vacuously healthy.
                 "latency_healthy_at_observed_nonfailing": (
                     all(c["verdict"] != FAIL for c in latency_at_observed.values()) if latency_at_observed else None)}
    out["nominal_binding_constraint"] = ceiling.binding if ceiling is not None else None
    out["attribution_basis"] = "observed symptoms; nominal quotas do not identify the cause of throttling"
    suspect = (profile_report.measurement_validity or {}).get("status", "valid") != "valid"
    sat = rec.saturation_point
    if sat is None:
        out["bottleneck"] = "unresolved" if suspect else rec.analysis.status
        return out
    verdict = next((v for p, v in zip(profile_report.points, profile_report.verdicts) if p is sat), None)
    failed = sorted({c.name.split(".")[-1] for c in (verdict.checks if verdict else []) if c.verdict == FAIL})
    m = sat.metrics
    other_errors = max(0.0, round(1.0 - m.success_rate - m.throttle_rate, 4))
    throttled = "throttle_rate" in failed or ("success_rate" in failed and m.throttle_rate > 0 and other_errors == 0)
    slow = any(name in _LATENCY_CHECKS for name in failed)
    if throttled and slow:
        bottleneck = "provider_throttling_and_latency"
    elif throttled:
        bottleneck = "provider_throttling"
    elif slow:
        bottleneck = "latency"
    else:
        bottleneck = "errors"
    attempted = round(m.n / m.measured_duration_s, 4) if m.measured_duration_s else None
    out.update({
        "bottleneck": "unresolved" if suspect else bottleneck,
        "observed_symptom": bottleneck,
        "saturation_at": _value(sat),
        "failed_checks": failed,
        "throttle_rate_at_saturation": m.throttle_rate,
        "non_throttle_error_rate_at_saturation": other_errors,
        "attempted_rps_at_saturation": attempted,       # requests sent per second
        "served_rps_at_saturation": m.request_throughput_rps,  # successes completed per second
        "provider_ceiling_rps": round(ceiling.rps, 4) if ceiling is not None and ceiling.rps else None,
    })
    return out


# Open-loop validity: an arrival scheduled at t must actually start at
# ~t. If the client's event loop or thread pool falls behind, the offered
# load isn't what was intended and the sweep measures the client, not
# Bedrock. p99 of (started_at - scheduled_at) above this -> invalid.
LOAD_GENERATOR_LAG_P99_LIMIT_MS = 50.0


def _load_generator(results: List[RequestResult]) -> Optional[dict]:
    lags = sorted(max(0.0, (r.started_at - r.scheduled_at) * 1000.0)
                  for r in results if r.tags.get("measured") and r.scheduled_at and r.started_at)
    if not lags:
        return None
    by_point: Dict[tuple, List[float]] = {}
    for r in results:
        if r.tags.get("measured") and r.scheduled_at and r.started_at:
            by_point.setdefault((r.tags.get("phase"), r.tags.get("sweep_value")), []).append(
                max(0.0, (r.started_at - r.scheduled_at) * 1000.0))
    (phase, value), worst = max(by_point.items(), key=lambda kv: percentile(sorted(kv[1]), 99))
    p99 = percentile(lags, 99)
    return {
        "scheduling_lag_p50_ms": round(percentile(lags, 50), 2),
        "scheduling_lag_p95_ms": round(percentile(lags, 95), 2),
        "scheduling_lag_p99_ms": round(p99, 2),
        "max_lag_ms": round(lags[-1], 2),
        "worst_point": {"phase": phase, "value": value, "lag_p99_ms": round(percentile(sorted(worst), 99), 2)},
        "limit_p99_ms": LOAD_GENERATOR_LAG_P99_LIMIT_MS,
        # Every point's p99, not just the pooled one: one lagging point
        # is enough to make that point's offered load untrustworthy.
        "valid": percentile(sorted(worst), 99) <= LOAD_GENERATOR_LAG_P99_LIMIT_MS,
    }


def _no_recommendation(purpose: str) -> dict:
    if purpose == "admission_calibration":
        return {"admission_envelope": None,
                "reason": "admission-calibration experiment -- see calibration_point: a statistically confirmed "
                          "capacity point for this workload shape, from which a gateway derives admission "
                          "classes or weights; no envelope or headroom is produced here"}
    return {"admission_envelope": None,
            "reason": "characterization experiment -- measures how token shape / context move the envelope; "
                      "production admission envelopes come only from reference experiments"}


def _calibration_point(spec, subject: str, rec: Optional[Recommendation], ceiling, diagnosis: Optional[dict],
                       validity: Optional[dict] = None) -> dict:
    """admission_calibration: the CONFIRMED capacity of one workload
    shape under its SLO, the quota and the measured provider environment
    -- C_safe = f(shape, SLO, quota, provider conditions) -- as an input
    for gateway policy derivation, not a config value (no headroom).
    The rates are what C and latency PRODUCED in this closed-loop run --
    observations, not a tested rate envelope (that is capacity-reference-rate's
    sustained_rps). Different workload shapes may require different
    concurrency to reach the same provider rate ceiling, so concurrency is
    not a workload cost weight. Null values when nothing was confirmed."""
    workload = next((w for w in spec.workloads if w.name == subject), None)
    confirmed = rec.confirmed_point if rec is not None else None
    point = {
        "workload_shape": {"input_tokens": workload.input_tokens, "output_tokens": workload.output_tokens}
        if workload is not None else None,
        "slo_profile": workload.slo_profile if workload is not None else None,
        "tokens_per_request": ceiling.tokens_per_request if ceiling is not None else None,
        f"statistically_confirmed_{spec.sweep.type}": _value(confirmed) if confirmed is not None else None,
        # Observed, not controlled: closed-loop C + latency produced them.
        "confirmed_rates": point_rates(confirmed) if confirmed is not None else None,
        "ceiling_ratio": ceiling_ratio(confirmed, ceiling.rps if ceiling else None) if confirmed is not None else None,
        # Held above the nominal ceiling for the confirmation window only.
        "confirmed_above_provider_ceiling": (above_ceiling(confirmed, ceiling.rps)
                                             if confirmed is not None and ceiling is not None and ceiling.rps else None),
        "observed_saturation_edge": _value(rec.saturation_point) if rec is not None and rec.saturation_point is not None else None,
        "bottleneck": (diagnosis or {}).get("bottleneck"),
        "scope": "isolated_workload_class",
        "use": "workload-specific admission evidence -> policy derivation -> mixed validation; not a config "
               "value or a cost weight, no headroom applied; any derived policy must be validated under "
               "representative mixed traffic through the deployed gateway (eval-bedrock-platform) before production",
    }
    if confirmed is None:
        point["reason"] = "nothing statistically confirmed for this shape -- see confirmation.candidates"
    status = (validity or {}).get("status")
    if status is not None:
        point["measurement_validity"] = status
    if status == "invalid":
        point["reason"] = ("measurement INVALID: the provider never passed a recovery probe -- no capacity "
                           "conclusion (confirmed or not) is drawn from this run; re-run")
    return point


def _throttled_below_ceiling(result, ceiling_rps: Optional[float]) -> bool:
    """See executor.suspect_point: throttled while served far below the
    nominal ceiling -- provider state, not the candidate's own load."""
    return suspect_point(result.point, ceiling_rps)


def _candidate_dict(result, ceiling_rps: Optional[float]) -> dict:
    out = result.to_dict()
    if result.point is not None and result.point.class_metrics:
        out["observed_mix"] = _observed_mix({name: m.n for name, m in result.point.class_metrics.items()})
    if result.point is not None and ceiling_rps:
        out["ceiling_ratio"] = ceiling_ratio(result.point, ceiling_rps)
        if _throttled_below_ceiling(result, ceiling_rps):
            out["throttled_below_ceiling"] = True
            out["capacity_interpretation"] = "provider_state_anomaly; candidate safety unresolved"
            out["anomaly"] = {"kind": "throttled_below_nominal_ceiling", "cause": "unresolved"}
    return out


def _unconfirmed_reason(profile_report, spec) -> str:
    """WHY nothing was statistically confirmed, from what actually ran --
    a confirmation phase's stop reason, or (discovery only) the point
    where the fixed-sequence test stopped and how many requests it
    lacked. Never suggests relaxing the SLO."""
    prefix = "no statistically confirmed point -- "
    if (profile_report.measurement_validity or {}).get("status") == "invalid":
        return prefix + ("measurement INVALID: the provider never passed a recovery probe, so this run supports no "
                         "capacity conclusion (not 'unsafe') -- see measurement_validity; re-run")
    if profile_report.recommendation is None:
        return prefix + "the first swept value already FAILs the SLO, so there is nothing to confirm; sweep lower values"
    if profile_report.confirmation_plan is not None:
        # Tested highest-first; every tested candidate is non-PASS here.
        tried = sorted((c for c in profile_report.confirmations if c.stop_reason != "not_tested"),
                       key=lambda c: -c.value)
        if not tried:
            return prefix + "no non-failing discovery point at or below the provider ceiling to confirm"
        parts = []
        for c in tried:
            part = f"{c.value:g}: {c.verdict} ({c.stop_reason}, n={c.n}"
            if c.next_look_n is not None:
                part += f", next look at n={c.next_look_n}"
            parts.append(part + ")")
        hint = _STOP_HINTS.get(tried[-1].stop_reason)
        text = (prefix + "confirmation (highest first) at " + "; ".join(parts) + (f"; {hint}" if hint else "")
                + " -- see `confirmation.candidates`")
        sub_ceiling = spec.provider_ceilings.get(profile_report.workload_name)
        suspect = [c for c in tried if _throttled_below_ceiling(c, sub_ceiling.rps if sub_ceiling else None)]
        if suspect:
            text += ("; NOTE: " + ", ".join(f"{c.value:g}" for c in suspect) + " throttled while served far below "
                     "the nominal ceiling (throttled_below_ceiling) -- possible provider state anomaly; cause unresolved: re-run before reading it as unsafe")
        return text
    ordered = sorted(zip(profile_report.points, profile_report.verdicts), key=lambda pv: _value(pv[0]))
    stop = next(((p, v) for p, v in ordered if v.verdict != PASS), None)
    text = prefix + ("discovery only (no `confirmation:` phase), a fixed-sequence test that stops at the first "
                     "non-PASS point")
    if stop is None:
        return text
    point, verdict = stop
    text += f": {_value(point):g} is {verdict.verdict}"
    lacking = [f"{c.name} n={c.n} < required_n={c.required_n}" for c in verdict.inconclusive_checks]
    if verdict.verdict == INCONCLUSIVE and lacking:
        text += f" ({'; '.join(lacking)}) -- add a `confirmation:` block to the experiment to collect them"
    return text


def _evidence(point) -> dict:
    m: RunMetrics = point.metrics
    out = {
        "n": m.n,
        "n_throttled": m.n_throttled,
        "measured_duration_s": m.measured_duration_s,
        "throttle_rate": m.throttle_rate,
        "throttle_rate_upper": m.throttle_rate_upper,
        "success_rate": m.success_rate,
        "success_rate_lower": m.success_rate_lower,
        "bound_confidence": m.bound_confidence,
        "ttft_p95_ms": m.ttft_p95_ms,
        "tpot_p95_ms": m.tpot_p95_ms,
        "latency_p95_ms": m.latency_p95_ms,
        "peak_outstanding": point.peak_outstanding,
    }
    if len(point.repetitions) > 1:
        out["repetition_slo_goodput_rps"] = [r.slo_goodput_rps for r in point.repetitions]
    return out


def _deviation(observed: Optional[float], target: int, tolerance_pct: float) -> dict:
    deviation_pct = valid = None
    if observed is not None and target > 0:
        deviation_pct = round((observed - target) / target * 100, 2)
        valid = abs(deviation_pct) <= tolerance_pct
    return {"target": target, "observed_p50": observed, "deviation_pct": deviation_pct,
            "tolerance_pct": tolerance_pct, "valid": valid}


def _workload_validation(workload: WorkloadProfile, results: List[RequestResult], report: ExperimentReport) -> dict:
    """Did this class actually measure the shape it claims? Compares
    requested input tokens and the output target (max_tokens) against
    what Bedrock itself reported during the run. Output matters as much
    as input: "4096 in / 512 out" that really emitted 110 tokens is a
    different workload, and its envelope would mislead a gateway config
    for long generations."""
    spec = report.spec
    observed = _observed_tokens(results)
    inp = _deviation(observed["input_tokens_p50"], workload.input_tokens, spec.workload_validation_tolerance_pct)
    out = _deviation(observed["output_tokens_p50"], workload.output_tokens, spec.output_validation_tolerance_pct)
    checks = [v for v in (inp["valid"], out["valid"]) if v is not None]
    calibration = report.calibrations.get(workload.name)
    return {
        "token_counting": calibration.to_dict() if calibration else {"method": "estimate"},
        "input": inp,
        "output": out,
        "valid": all(checks) if checks else None,
    }


def _slo_dict(slo) -> dict:
    return {
        "ttft_p95_ms": slo.ttft_p95_ms,
        "tpot_p95_ms": slo.tpot_p95_ms,
        "latency_p95_ms": slo.latency_p95_ms,
        "success_rate_min": slo.success_rate_min,
        "throttle_rate_max": slo.throttle_rate_max,
        "confidence": slo.confidence,
    }


def _measurement_windows(subject, spec, all_results, recorded_windows=()):
    """Keep each repetition/recovery attempt separate, including warmup/drain rows."""
    from .analysis.metrics import MeasurementWindow
    from .analysis.observability import describe_series

    grouped = {
        (w["start"], w["end"], w["phase"], w["value"], w["repetition"]): []
        for w in recorded_windows if w["subject"] == subject and w["phase"] != "invalidated"
    }
    for r in all_results:
        tags = r.tags
        if tags.get("subject") != subject or tags.get("phase") not in (
                "discovery", "refinement", "confirmation"):
            continue
        start, end = tags.get("window_start"), tags.get("window_end")
        if start is None or end is None:
            continue
        key = (start, end, tags.get("phase"), tags.get("sweep_value"), tags.get("repetition"))
        grouped.setdefault(key, []).append(r)
    slos = {w.name: (spec.slo_for(w.name).ttft_p95_ms, spec.slo_for(w.name).latency_p95_ms,
                     spec.slo_for(w.name).tpot_p95_ms) for w in spec.workloads}
    out = []
    for (start, end, phase, value, repetition), rows in sorted(grouped.items()):
        kwargs = dict(slo_by_workload=slos,
                      configured_concurrency=value if spec.sweep.type == "concurrency" else None,
                      configured_offered_rps=value if spec.sweep.type == "rate" else None)
        window = MeasurementWindow(start, end)
        entry = {"phase": phase, "value": value, "repetition": repetition,
                 **describe_series(rows, window, **kwargs)}
        observed_peak = next((w["peak_sdk_inflight"] for w in recorded_windows
                              if w["subject"] == subject and w["start"] == start
                              and "peak_sdk_inflight" in w), None)
        if observed_peak is not None:
            entry["peak_sdk_inflight"] = observed_peak
        if spec.mix is not None:
            entry["classes"] = {
                w.name: describe_series([r for r in rows if r.tags.get("workload") == w.name],
                                        window, slo_by_workload=slos)
                for w in spec.workloads
            }
        out.append(entry)
    return out


def _envelope(entry: dict, profile_report, spec, report_results: List[RequestResult], recorded_windows=()) -> None:
    _envelope_unchecked(entry, profile_report, spec, report_results, recorded_windows)
    reasons = []
    if entry.get("workload_validation", {}).get("valid") is False:
        reasons.append("workload_validation_failed")
    if (profile_report.measurement_validity or {}).get("status") == "invalid":
        reasons.append("measurement INVALID")
    if reasons:
        prior_reason = entry.get("recommendation", {}).get("reason")
        if prior_reason:
            reasons.append(prior_reason)
        entry["recommendation"] = {"admission_envelope": None, "reason": "; ".join(reasons)}
        if "calibration_point" in entry:
            entry["calibration_point"] = None


def _envelope_unchecked(entry: dict, profile_report, spec, report_results: List[RequestResult], recorded_windows=()) -> None:
    subject = profile_report.workload_name
    control = entry.get("role") == "reference_control"
    purpose = "characterization" if control else spec.purpose
    if control:
        entry["control_use"] = "reference control for interpreting this experiment; no admission calibration or recommendation"
    if subject in spec.provider_ceilings:
        entry["provider_constraints"] = spec.provider_ceilings[subject].to_dict()
    if spec.sweep.quota_fractions is not None:
        entry["sweep_values_rps"] = spec.sweep_values(subject)
    entry["measurement_windows"] = _measurement_windows(subject, spec, report_results, recorded_windows)
    if profile_report.history_comparison is not None:
        entry["history_comparison"] = profile_report.history_comparison
        entry["measurement_validity"] = profile_report.measurement_validity
        entry["recommendation"] = {"admission_envelope": None,
                                   "reason": "history comparison: descriptive time series, no capacity claim"}
        return
    points = profile_report.points
    limited = [p.concurrency if p.concurrency is not None else p.rps for p in points if p.client_limited]
    if limited:
        entry["client_limited_points"] = limited
    # Did the client deliver the arrivals it scheduled? (started_at -
    # scheduled_at: event-loop lag + thread-pool queueing.)
    generator = _load_generator([r for r in report_results if r.tags.get("subject") == subject])
    if generator is not None:
        entry["load_generator"] = generator
    if profile_report.verdicts:
        # Discovery only: picks candidates and shows the transition region.
        sub_ceiling = spec.provider_ceilings.get(subject)
        entry["sweep_points"] = _sweep_points(profile_report, sub_ceiling.rps if sub_ceiling else None)
    swept = {_value(p) for p in points}
    skipped = [v for v in spec.sweep_values(subject) if v not in swept]
    if skipped and (spec.sweep.stop_after_fails is not None or spec.sweep.stop_after_clear_fail):
        entry["sweep_stopped_early"] = {"after_consecutive_fails": spec.sweep.stop_after_fails,
                                        **({"stop_after_clear_fail": True} if spec.sweep.stop_after_clear_fail else {}), "skipped_values": skipped}
    if profile_report.measurement_validity is not None:
        # valid | suspect_* | invalid -- suspect measurements carry
        # explicit caveats even when a lower candidate confirms.
        entry["measurement_validity"] = profile_report.measurement_validity
    if profile_report.confirmation_plan is not None:
        # Independent data at the candidates; the only source of
        # statistically_confirmed when present.
        sub_ceiling = spec.provider_ceilings.get(subject)
        entry["confirmation"] = {
            "plan": profile_report.confirmation_plan.to_dict(),
            "candidates": [_candidate_dict(c, sub_ceiling.rps if sub_ceiling else None)
                           for c in profile_report.confirmations],
        }
    if spec.burst_protocol:
        entry["burst"] = (profile_report.measurement_validity or {}).get("burst_results", [])
        entry["recommendation"] = _no_recommendation(purpose)
        return
    rec = profile_report.recommendation
    if rec is None:
        analysis = profile_report.analysis
        if analysis is not None and analysis.status == "unresolved":
            entry["note"] = ("non-monotonic from the first swept value -- no stable passing region; "
                             "re-run with repetitions")
            entry["unstable_region"] = analysis.unstable_region
        else:
            entry["note"] = "no swept value met the configured SLO -- re-run with lower sweep values"
        entry["recommendation"] = _no_recommendation(purpose) if purpose != "reference" else admission_envelope(
            spec.sweep.type, None, headroom=spec.provider_headroom,
            unconfirmed_reason=_unconfirmed_reason(profile_report, spec),
        )
        if purpose == "admission_calibration":
            entry["calibration_point"] = _calibration_point(spec, subject, None, spec.provider_ceilings.get(subject), None,
                                                            profile_report.measurement_validity)
        return
    ceiling = spec.provider_ceilings.get(subject)
    ceiling_rps = ceiling.rps if ceiling else None
    if spec.sweep.type == "concurrency":
        entry["concurrency"] = _concurrency_block(rec, ceiling_rps=ceiling_rps)
        confirmed = rec.confirmed_point.concurrency if rec.confirmed_point is not None else None
    else:
        entry["rate"] = _rate_block(rec, ceiling_rps=ceiling_rps)
        confirmed = rec.confirmed_point.rps if rec.confirmed_point is not None else None
    block = entry["concurrency" if spec.sweep.type == "concurrency" else "rate"]
    # isolated_workload_class: this class running ALONE -- per-class
    # values are not additive across classes and are not a global limit.
    scope = "workload_mix" if profile_report.mix_shares is not None else "isolated_workload_class"
    block["scope"] = scope
    block["summary"] = _summary(rec)
    # MEASUREMENT interpretation: what limits this envelope.
    entry["diagnosis"] = _diagnosis(rec, profile_report, ceiling)
    # POLICY, kept apart from the measurement above: the confirmed point
    # after this benchmark's safety headroom (recommendation.py) -- only
    # from a reference experiment.
    if purpose == "admission_calibration":
        entry["calibration_point"] = _calibration_point(spec, subject, rec, ceiling, entry["diagnosis"],
                                                        profile_report.measurement_validity)
    entry["recommendation"] = _no_recommendation(purpose) if purpose != "reference" else admission_envelope(
        spec.sweep.type, confirmed, headroom=spec.provider_headroom, quota_headroom=spec.quota_headroom,
        provider_ceiling_rps=ceiling_rps, scope=scope,
        unconfirmed_reason=None if confirmed is not None else _unconfirmed_reason(profile_report, spec),
    )
    envelope = entry["recommendation"].get("admission_envelope")
    if envelope is not None:
        # One run is one snapshot of provider conditions -- never by itself
        # a production-safe config (see drift.build_temporal_profile).
        envelope["evidence"] = "single_run_operating_envelope"
        envelope["production_use"] = ("not production-ready alone: repeat across times / days, then use "
                                      "`bedrock-benchmark validate`'s production_capacity_input "
                                      "(temporal-capacity-profile.yaml)")
    # evidence = the observed point; confirmed_evidence = the point the
    # recommendation is derived from, when it's a different point.
    entry["evidence"] = _evidence(rec.point)
    entry["evidence"]["verdict"] = rec.verdict.to_dict()
    if rec.confirmed_point is not None and rec.confirmed_point is not rec.point:
        entry["confirmed_evidence"] = _evidence(rec.confirmed_point)


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


# The code that RUNS is the code imported when the process started -- a
# commit made during a long run must not be attributed to it. Captured
# once, at import.
_GIT_AT_START = (_git("rev-parse", "HEAD"), _git("status", "--porcelain", "--untracked-files=no"))


def _version(dist: str) -> Optional[str]:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _iso(ts: Optional[float]) -> Optional[str]:
    return None if ts is None else datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def _environment(report: ExperimentReport) -> dict:
    """Provenance: the measured envelope belongs to THIS environment at
    THIS time -- model + Bedrock serving + inference-profile routing +
    account/region quota + provider conditions then. Needed to compare
    runs over time (scripts/drift.py) and to know when a profile is stale."""
    spec = report.spec
    started = [r.started_at for r in report.all_results if r.started_at]
    completed = [r.completed_at for r in report.all_results if r.completed_at]
    commit, status = _GIT_AT_START
    return {
        "measured_at": {"start": _iso(min(started) if started else None),
                        "end": _iso(max(completed) if completed else None)},
        "account": spec.quota_account,
        "region": spec.target.region,
        "inference_profile": spec.target.model_id,
        "benchmark_version": _version("eval-bedrock-runtime-benchmark"),
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "runtime": {"python": platform.python_version(), "boto3": _version("boto3"),
                    "botocore": _version("botocore")},
    }


def build_capacity_profile(report: ExperimentReport, run_metadata: Optional[dict] = None) -> dict:
    """`run_metadata` -- who ran it and why ({run_id, owner, purpose,
    ticket, environment}), so a shared result answers "who, why, which
    run" without asking the author."""
    spec = report.spec
    by_name = {p.workload_name: p for p in report.profiles}
    workload_classes: Dict[str, dict] = {}

    # Every defined workload gets observed tokens + validation. It gets
    # an isolated envelope only when it was swept in isolation -- a
    # class measured inside a mix has no isolated envelope to report.
    for workload in spec.workloads:
        own_results = [
            r for r in report.all_results
            if r.tags.get("workload") == workload.name and r.tags.get("measured", True)
        ]
        entry: dict = {
            "slo_profile": workload.slo_profile,
            "role": workload.role,  # reference | characterization (catalog/workloads.yaml)
            "observed": _observed_tokens(own_results),
            "workload_validation": _workload_validation(workload, own_results, report),
        }
        profile_report = by_name.get(workload.name)
        if profile_report is not None and profile_report.mix_shares is None:
            _envelope(entry, profile_report, spec, report.all_results, report.measurement_windows)
            if spec.name == "capacity-reference-rate":
                entry.get("rate", {}).pop("measured_burst_ceiling_rps", None)
        if profile_report is not None and profile_report.mix_shares is None:
            confirmed = next((c for c in profile_report.confirmations if c.verdict == PASS), None)
            entry["operating_conditions"] = {
                "scope": "isolated_workload_class",
                "provider_state": confirmed.provider_state if confirmed else None,
                "measurement_status": (profile_report.measurement_validity or {}).get("status", "unverified"),
                "quota": {"rpm": spec.quota_snapshot.rpm, "tpm": spec.quota_snapshot.tpm},
                "slo": _slo_dict(spec.slo_for(workload.name)),
                "environment": _environment(report),
                "interpretation": "conditional on observed conditions; recovery probes do not prove provider reset",
                "temporal_validation_required": True,
            }
        workload_classes[workload.name] = entry

    mixed: Dict[str, dict] = {}
    for profile_report in report.profiles:
        if profile_report.mix_shares is None:
            continue
        entry = {"shares": {k: round(v, 4) for k, v in profile_report.mix_shares.items()}}
        entry["configured_mix"] = {
            "name": spec.mix.name, "weights": dict(spec.mix.weights),
            "shares": dict(profile_report.mix_shares), "assignment": spec.mix.assignment,
            "source": spec.mix.source, "observed_from": spec.mix.observed_from,
        }
        entry["observed_mix"] = _observed_mix({
            name: sum(1 for r in report.all_results
                      if r.tags.get("measured", True) and r.tags.get("workload") == name)
            for name in profile_report.mix_shares
        })
        entry["observed_mix"]["scope"] = "all_measured_requests"
        entry["workload_validation"] = {
            "valid": not any(workload_classes[n]["workload_validation"]["valid"] is False
                             for n, share in profile_report.mix_shares.items() if share > 0)
        }
        _envelope(entry, profile_report, spec, report.all_results, report.measurement_windows)
        rec = profile_report.recommendation
        if rec is not None and rec.confirmed_point is not None:
            # Per class at the CONFIRMED point -- the capacity -- not the
            # observed one.
            entry["classes_at_confirmed_point"] = {
                name: {
                    "n": m.n, "slo_goodput_rps": m.slo_goodput_rps, "throttle_rate": m.throttle_rate,
                    "ttft_p95_ms": m.ttft_p95_ms, "tpot_p95_ms": m.tpot_p95_ms, "latency_p95_ms": m.latency_p95_ms,
                }
                for name, m in rec.confirmed_point.class_metrics.items()
            }
        mixed[profile_report.workload_name] = entry

    confidence = spec.slo.confidence or DEFAULT_CONFIDENCE
    return {
        "schema_version": 23,
        "experiment": spec.name,
        "mode": spec.mode,
        "run": run_metadata or {},
        # reference: carries production admission envelopes;
        # admission_calibration: confirmed calibration_point per workload
        # shape (no envelope); characterization: measurement only.
        "purpose": spec.purpose,
        "environment": _environment(report),
        # One profile is ONE snapshot of provider conditions. Validity
        # across time comes from comparing repeated runs (bedrock-benchmark validate),
        # which reports runs / days_observed / spread per envelope.
        "validity": {
            # One run is never more than this; a stable / conservative
            # envelope needs repeated independent runs (bedrock-benchmark validate).
            "envelope": "single_run_operating_envelope",
            "repeated_runs": 1,
            "days_observed": 1,
            "scope": "single run -- a snapshot of the provider conditions at measured_at; re-measure on "
                     "other days and times, then `bedrock-benchmark validate` for a temporal-capacity-profile",
        },
        "model": {
            "name": spec.model_name,
            "provider": "bedrock",
            "model_id": spec.target.model_id,
            "region": spec.target.region,
        },
        # What every number below was judged against: the provider's
        # quota (constraints/quota.yaml) and the required SLO
        # (constraints/slo.yaml; each workload class names its profile).
        "constraints": {
            "quota": {
                "account": spec.quota_account,
                "region": spec.target.region,
                "rpm": spec.quota_snapshot.rpm,
                "tpm": spec.quota_snapshot.tpm,
                "output_burndown": spec.output_burndown,
            },
            # The profiles this experiment's workloads use (each class
            # names its own under workload_classes.<name>.slo_profile).
            # POLICY, not a result: externally supplied requirements the
            # envelope is judged against -- never derived from measurements.
            "slo": {
                "role": "policy_input",
                "ttft_selection": "configured_workload_input_tokens" if spec.ttft_budgets else "profile",
                "ttft_budgets": {
                    name: {"max_input_tokens": b.max_input_tokens, "ttft_p95_ms": b.ttft_p95_ms}
                    for name, b in spec.ttft_budgets.items()
                },
                "effective_by_workload": {
                    w.name: {"ttft_budget": spec.ttft_budget_name(w.name), **_slo_dict(spec.slo_for(w.name))}
                    for w in spec.workloads
                },
                "profiles": {
                    n: _slo_dict(spec.slo_profiles[n])
                    for n in sorted({w.slo_profile for w in spec.workloads if w.slo_profile in spec.slo_profiles})
                },
            },
            # The catalog entries (catalog/workloads.yaml) as run: shape,
            # profile, and the workload-level E2E cap -- profiles carry
            # TPOT/reliability; TTFT is resolved above by input length. The
            # effective E2E limit is the one here.
            "workloads": {
                w.name: {"input_tokens": w.input_tokens, "output_tokens": w.output_tokens,
                         "slo_profile": w.slo_profile, "latency_p95_ms": w.latency_p95_ms, "role": w.role}
                for w in spec.workloads
            },
        },
        "measurement": {
            "workload_rotation_index": spec.workload_rotation_index,
            "workload_order": [p.workload_name for p in report.profiles],
            "burst_protocol": spec.burst_protocol,
            "metrics_version": 1,
            "stream": spec.stream,
            "descriptive_bin_s": spec.history_protocol.bin_s if spec.history_protocol else 30,
            "metric_definitions": {
                "latency": "successful scheduled cohort, including drain; missing values excluded with n reported",
                "e2e_ms": "SDK invocation to response/stream completion; excludes executor queueing",
                "tpot_ms": "(stream completion - first text) / (output tokens - 1); existing SLO definition",
                "text_decode_tpot_ms": "(last text - first text) / (output tokens - 1); descriptive, no SLO gate",
                "throughput": "successful completions inside window / window seconds; tokens from provider usage",
                "attempted_rps": "SDK calls started inside window / window seconds",
                "reliability": "all requests scheduled inside window, including drain outcomes",
                "outstanding": "time-weighted mean and peak clipped to window; queue-inclusive and SDK calls separate",
                "max_inflight": "null means no explicit in-flight cap for open-loop rate runner",
                "bins": "descriptive only; no additional hypothesis tests",
            },
            **({"retest": spec.retest} if spec.retest is not None else {}),
            **({"history_protocol": vars(spec.history_protocol)} if spec.history_protocol is not None else {}),
            **({"baseline_context": spec.baseline_context} if spec.baseline_context is not None else {}),
            "warmup_s": spec.warmup_s,
            "window_s": spec.duration_s,
            "throttle_pause_s": spec.throttle_pause_s,  # concurrency sweeps: a worker's wait after a 429
            # Provider-state isolation: idle time so no phase is measured
            # conditional on the overload before it (confirmation.cooldown_s
            # separates every candidate).
            "isolation": {
                "inter_subject_cooldown_s": spec.inter_subject_cooldown_s,
                "refinement_cooldown_s": spec.sweep.refinement.cooldown_s if spec.sweep.refinement else None,
                "per_candidate_cooldown_s": spec.confirmation.cooldown_s if spec.confirmation else None,
                # Recovery is checked by this probe, not assumed (None: intervals only).
                "recovery_probe": dict(spec.recovery_probe.__dict__) if spec.recovery_probe else None,
            },
            "repetitions": spec.repetitions,
            # rates/percentiles over requests scheduled in the window
            # (drain included); throughput over completions in it.
            "window_policy": "scheduled_in_window_for_rates__completed_in_window_for_throughput",
            "clock": "monotonic_durations__wall_clock_timestamps",
            # Every check is PASS / FAIL / INCONCLUSIVE on exact
            # Clopper-Pearson bounds at this confidence: upper bound within
            # the limit -> PASS, lower bound beyond it -> FAIL (split over
            # the point's checks), otherwise INCONCLUSIVE.
            "gate": "pass_fail_inconclusive",
            "fail_rule": "exact_lower_bound_beyond_limit",
            "confidence": confidence,
            "confirmation": (
                {"max_looks": spec.confirmation.max_looks, "max_repetitions": spec.confirmation.max_repetitions,
                 "max_requests": spec.confirmation.max_requests, "max_duration_s": spec.confirmation.max_duration_s,
                 "candidates": spec.confirmation.candidates, "cooldown_s": spec.confirmation.cooldown_s,
                 "warmup_s": spec.confirmation.warmup_s,
                 "min_steady_state_duration_s": spec.confirmation.min_steady_state_duration_s,
                 "continuous": spec.confirmation.continuous}
                if spec.confirmation is not None else None
            ),
            "min_requests_to_resolve_throttle_slo": min_samples_to_resolve_rate(
                spec.slo.throttle_rate_max, confidence=confidence,
            ),
        },
        "sweep": {
            "type": spec.sweep.type,
            "stop_after_clear_fail": spec.sweep.stop_after_clear_fail,
            # Absolute values, or quota_fractions of each subject's
            # provider ceiling (resolved per class: sweep_values_rps).
            **({"quota_fractions": spec.sweep.quota_fractions, "relative_to": "provider_ceiling"}
               if spec.sweep.quota_fractions is not None else
               {"values_by_workload": {n: spec.sweep_values(n) for n in spec.subject_names}}
               if spec.sweep.values_by_workload is not None else {"values": list(spec.sweep.values)}),
        },
        "workload_classes": workload_classes,
        # Only present for a `mix:` experiment -- the one valid source
        # of a cross-class envelope, and only for THAT mix's shares.
        **({"mixed_workloads": mixed} if mixed else {}),
        # The policy applied to turn confirmed measurements into each
        # entry's `recommendation` -- configured
        # (constraints/recommendation-policy.yaml), not measured.
        "recommendation_policy": {
            "headroom_fraction": spec.provider_headroom,        # back-off from the confirmed point
            "quota_headroom_fraction": spec.quota_headroom,     # back-off from the provider ceiling
        },
        # Recorded for reproducibility -- what was actually running
        # when these numbers were measured (see client.py's
        # TransportConfig docstring on why this matters: SDK retry/
        # pooling/thread defaults can silently change what a sweep measures).
        "transport": {
            "max_connections": spec.transport.max_connections,
            "executor_workers": spec.transport.effective_executor_workers,
            "total_max_attempts": spec.transport.total_max_attempts,
            "connect_timeout_s": spec.transport.connect_timeout_s,
            "read_timeout_s": spec.transport.read_timeout_s,
        },
    }
