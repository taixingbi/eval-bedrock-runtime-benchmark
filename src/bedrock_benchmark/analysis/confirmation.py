"""Confirmation phase -- the ONLY source of a statistically confirmed
operating point. Discovery data picks candidates; fresh, independent
confirmation data decides whether a candidate meets its SLO.

Why a separate module: the statistics here are what make a PASS mean
something, so they live apart from the Bedrock I/O loop (executor.py)
and are testable on their own.

Two rules keep the false-PASS rate -- and, symmetrically, the false-FAIL
rate -- at or below alpha = 1 - confidence:

1. No double-dipping. The candidate is chosen because its DISCOVERY data
   looked good; confirming it with that same data would be biased. So
   confirmation verdicts use confirmation data only -- discovery data is
   never pooled in.

2. No unplanned looks. Re-checking a confidence bound after every repetition
   and stopping the first time it clears is optional stopping: given
   enough looks, noise alone eventually produces a PASS -- or a FAIL. So
   BOTH are declared only at L sample sizes fixed BEFORE any confirmation
   data exists (the look schedule). With K candidates (see below) there
   are L x K tests in all, each at the per-test confidence
   1 - alpha / (L x K) (Bonferroni): P(any false PASS) <= alpha, and
   likewise P(any false FAIL) <= alpha. At a look:

       every check's exact upper bound within its limit -> PASS
       any check's exact lower bound beyond its limit   -> FAIL
       (the FAIL side is also split over the point's m checks --
        capacity.fail_confidence -- since ANY check can fail it)
       otherwise                                         -> next look
   Each test is the EXACT Clopper-Pearson bound, so its error really is
   <= alpha / (L x K) (Wilson would not guarantee that at the 0-2 events
   gold operates at).
   Look j's sample size is the smallest n at which j - 1 observed bad
   events would still clear the limit -- so later looks exist to absorb
   a stray throttle, not to retry until lucky.

   In a mix each class's checks see only that class's requests, and the
   class of each arrival is random, so look j is taken when EVERY group
   has its own required count -- blend total and each class's ACTUAL n
   -- not when the total reaches required / expected share (6,147 total
   requests can hold only 3,600 short_chat ones). The class counts
   depend only on the class draws, never on outcomes, so the look times
   stay outcome-independent and the Bonferroni bound holds.

   Looks are FIXED-COUNT: look j is evaluated on exactly the first N_j
   confirmation requests (by scheduled time; in a mix, the first N_c,j
   of each class) -- never on however many a fixed-duration repetition
   happened to produce. In a closed-loop concurrency run the request
   count per 120 s depends on outcomes (fast 429s add requests, slow
   responses remove them), so evaluating "all n >= N_j" would make the
   sample size outcome-dependent; truncating to N_j keeps each look an
   exact test at a pre-declared n.

   EARLY STOP is separate from FAIL: a candidate whose confirmation data
   is severely throttled (throttle_rate >= SEVERE_THROTTLE_RATE over at
   least SEVERE_MIN_N requests) stops at once to spare the provider --
   recorded as INCONCLUSIVE with stop_reason `stopped_severe_throttling`,
   never as a statistical FAIL and never as a saturation edge (it tested
   nothing at a planned look).

Caps (repetitions / requests per candidate, wall time for the phase)
bound the cost. Reaching a cap without a PASS is INCONCLUSIVE -- the
SLO is never relaxed to reach a verdict. A candidate whose first look
can't be reached within the caps (estimated from discovery's
requests-per-repetition) is reported `unreachable_within_caps` without
spending the calls.

Several candidates are tested HIGHEST-first and stop at the first PASS:
the goal is the maximum confirmed point, and the highest candidate
usually confirms, so testing it first saves confirming a lower one too.
Each candidate is its own chance of a false PASS ("high falsely passes",
or "high fails, then low falsely passes"), so alpha is split across
candidates as well as looks: every look runs at 1 - alpha / (L x K) for
K candidates (Bonferroni) -- P(any false PASS) <= L x K x alpha / (L x K)
= alpha. K is the number of candidates discovery actually chose -- fixed
before any confirmation data exists -- so one candidate keeps
1 - alpha / L. (Lowest-first as a fixed sequence would keep alpha per
test unsplit, but confirms every lower point on the way up; with K = 2,
gold's first look is 4,380 requests instead of 2 x 3,688.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

from .capacity import FAIL, INCONCLUSIVE, LATENCY_EXCEEDANCE_MAX, PASS, SweepPoint, Verdict
from .metrics import RunMetrics, rate_upper, required_samples  # noqa: F401 -- re-exported


@dataclass
class RateLimit:
    """One rate check to plan for. share: the fraction of the candidate's
    requests this check sees (1.0 for the blend, a class's mix share)."""
    name: str
    max_bad_rate: float  # throttle_rate_max, or 1 - success_rate_min
    share: float = 1.0


@dataclass
class ConfirmationPlan:
    confidence: float           # the SLO's (family-wise) confidence, e.g. 0.95
    max_looks: int              # L
    # Confidence of EACH test (every look of every candidate):
    # 1 - (1 - confidence) / (L x K) -- corrects for looks AND candidates.
    per_test_confidence: float
    look_schedule: List[int]    # N_1 < ... < N_L total confirmation requests (expected, for a mix)
    max_repetitions: Optional[int]  # None: no repetition cap (sample-count driven)
    max_requests: Union[int, str]
    max_duration_s: Union[float, str]
    # Per look, the ACTUAL count each group needs: "total" (the blend's
    # checks) and, in a mix, each class name. A look is taken only when
    # every group has reached its count.
    look_requirements: List[Dict[str, int]] = field(default_factory=list)
    # K: the candidates alpha is split over (tested highest-first).
    candidates: int = 1
    order: str = "highest_first"
    min_steady_state_duration_s: float = 0.0

    def look_sizes(self, j: int) -> Dict[str, int]:
        """The exact sample for look j: {"total": N} or, in a mix, {class: N_c}."""
        return self.look_requirements[j] if self.look_requirements else {TOTAL: self.look_schedule[j]}

    def look_reached(self, j: int, n: int, class_n: Optional[Dict[str, int]] = None) -> bool:
        if not self.look_requirements:
            return n >= self.look_schedule[j]
        return all((n if group == TOTAL else (class_n or {}).get(group, 0)) >= need
                   for group, need in self.look_requirements[j].items())

    def to_dict(self) -> dict:
        out = {
            "confidence": self.confidence, "max_looks": self.max_looks,
            "candidates": self.candidates, "order": self.order,
            "min_steady_state_duration_s": self.min_steady_state_duration_s,
            # 1 - (1 - confidence) / (max_looks x candidates), Bonferroni
            "per_test_confidence": round(self.per_test_confidence, 6),
            "look_schedule_requests": self.look_schedule,
            "caps": {"max_repetitions": self.max_repetitions, "max_requests": self.max_requests,
                     "max_duration_s": self.max_duration_s},
        }
        if any(len(r) > 1 for r in self.look_requirements):
            out["look_requirements"] = self.look_requirements  # per class: actual n, not total x share
        return out


TOTAL = "total"  # the look-requirement group of blend (non-class) checks


def plan_looks(limits: List[RateLimit], *, confidence: float, max_looks: int, max_repetitions: Optional[int],
               max_requests: Union[int, str], max_duration_s: Union[float, str], candidates: int = 1,
               min_steady_state_duration_s: float = 0.0) -> ConfirmationPlan:
    """Look j (1-based) is where every rate check could still PASS with
    j - 1 bad events of its own -- fixed before any confirmation data
    exists. Requirements are per group (the blend's total, and each
    class's own n); look_schedule is the total request count at which
    they're EXPECTED to be met (class requirement / share), used for
    caps and time estimates only."""
    per_test = 1.0 - (1.0 - confidence) / (max_looks * max(1, candidates))
    groups: Dict[str, List[RateLimit]] = {}
    for lim in limits:
        group = lim.name.split(".", 1)[0] if "." in lim.name else TOTAL
        groups.setdefault(group, []).append(lim)
    requirements: List[Dict[str, int]] = []
    schedule: List[int] = []
    for j in range(max_looks):
        need = {g: max(required_samples(j, lim.max_bad_rate, confidence=per_test) for lim in ls)
                for g, ls in groups.items()}
        if requirements:  # strictly increasing per group
            need = {g: max(v, requirements[-1][g] + 1) for g, v in need.items()}
        requirements.append(need)
        n = max(math.ceil(need[g] / groups[g][0].share) for g in need)
        schedule.append(max(n, schedule[-1] + 1) if schedule else n)
    return ConfirmationPlan(
        confidence=confidence, max_looks=max_looks, per_test_confidence=per_test, look_schedule=schedule,
        max_repetitions=max_repetitions, max_requests=max_requests, max_duration_s=max_duration_s,
        look_requirements=requirements, candidates=max(1, candidates),
        min_steady_state_duration_s=min_steady_state_duration_s,
    )


@dataclass
class ConfirmationResult:
    value: float                 # the candidate's concurrency or rps
    verdict: str                 # PASS | FAIL | INCONCLUSIVE
    stop_reason: str             # confirmed | violation_demonstrated (both at a look) | looks_exhausted |
                                 # stopped_severe_throttling | max_repetitions | max_requests | max_duration |
                                 # unreachable_within_caps | provider_state_invalid | not_tested | post_look_violation
    repetitions: int = 0
    n: int = 0
    looks_used: int = 0
    next_look_n: Optional[int] = None  # the look it was working toward when it stopped
    detail: Optional[Verdict] = None   # checks at the last evaluation (per-test confidence)
    point: Optional[SweepPoint] = None # ALL confirmation data collected -- never discovery (descriptive)
    # The exact fixed-count sample the last LOOK was decided on (first N_j).
    decision_n: Optional[int] = None
    decision_metrics: Optional["RunMetrics"] = None
    # The caps this candidate ran under -- per candidate when `auto`.
    caps: Optional[dict] = None
    steady_state: Optional[dict] = None
    provider_state: Optional[dict] = None

    def to_dict(self) -> dict:
        out = {"value": self.value, "verdict": self.verdict, "stop_reason": self.stop_reason,
               "repetitions": self.repetitions, "n": self.n, "looks_used": self.looks_used}
        if self.next_look_n is not None and self.verdict != PASS:
            out["next_look_n"] = self.next_look_n
        if self.provider_state is not None:
            out["provider_state"] = self.provider_state
        if self.steady_state is not None:
            out["steady_state"] = self.steady_state
        if self.caps is not None:
            out["caps"] = self.caps
        if self.decision_metrics is not None:
            d = self.decision_metrics
            # What the look was DECIDED on: exactly the first decision_n requests.
            out["decision"] = {"n": self.decision_n, "n_throttled": d.n_throttled,
                               "throttle_rate_upper": d.throttle_rate_upper, "success_rate_lower": d.success_rate_lower}
        if self.detail is not None:
            out["checks_basis"] = "fixed_count_decision" if self.stop_reason in ("confirmed", "violation_demonstrated", "looks_exhausted") else "descriptive_all_collected; not the sequential verdict"
            out["checks"] = [c.to_dict() for c in self.detail.checks if c.verdict != PASS] or "all PASS"
        if self.point is not None:
            m = self.point.metrics
            out["metrics"] = {"n_throttled": m.n_throttled, "throttle_rate": m.throttle_rate,
                              "throttle_rate_upper": m.throttle_rate_upper,
                              "success_rate_lower": m.success_rate_lower, "ttft_p95_ms": m.ttft_p95_ms,
                              "tpot_p95_ms": m.tpot_p95_ms, "latency_p95_ms": m.latency_p95_ms,
                              "slo_goodput_rps": m.slo_goodput_rps}
        return out


SEVERE_THROTTLE_RATE = 0.10  # early stop: >= 10% of confirmation requests throttled ...
SEVERE_MIN_N = 100           # ... over at least this many requests


def severe_throttling(metrics: RunMetrics) -> bool:
    """The early-stop condition -- an operational guard, not a test."""
    return metrics.n >= SEVERE_MIN_N and metrics.throttle_rate >= SEVERE_THROTTLE_RATE


def step(verdict: Verdict, n: int, looks_used: int, plan: ConfirmationPlan,
         class_n: Optional[Dict[str, int]] = None,
         look_verdict: Optional[Callable[[int], Verdict]] = None) -> Optional[tuple]:
    """Decide after one confirmation repetition. Verdicts count ONLY at a
    look: taken when every group's count (n, and in a mix each class's
    `class_n`) reaches the next scheduled requirement, and decided by
    `look_verdict(j)` -- the verdict (bounds at plan.per_test_confidence)
    on EXACTLY the first N_j requests (fixed-count; `verdict`, the
    all-data one, when not given -- simulations whose n is exact
    already). PASS -> statistical sample condition met (the executor also
    requires minimum exposure and no collected-data veto),
    FAIL -> violation_demonstrated, else the
    look is spent and measuring continues. Returns (verdict, stop_reason,
    looks_used) to stop, or None to keep measuring."""
    while looks_used < plan.max_looks and plan.look_reached(looks_used, n, class_n):
        at_look = look_verdict(looks_used) if look_verdict is not None else verdict
        looks_used += 1  # this scheduled look is spent whatever it decides
        if at_look.verdict == PASS:
            return PASS, "confirmed", looks_used
        if at_look.verdict == FAIL:
            return FAIL, "violation_demonstrated", looks_used
    if looks_used >= plan.max_looks:
        return INCONCLUSIVE, "looks_exhausted", looks_used
    return None


def look_sample(results: List, sizes: Dict[str, int]) -> List:
    """Exactly the first N requests of the confirmation stream by
    scheduled time -- or, in a mix, the first N_c of each class
    (sizes = ConfirmationPlan.look_sizes(j)). `results` must be the
    measured (in-window) confirmation requests."""
    ordered = sorted(results, key=lambda r: r.scheduled_at)
    if set(sizes) == {TOTAL}:
        return ordered[:sizes[TOTAL]]
    out = []
    for cls, need in sizes.items():
        out += [r for r in ordered if r.tags.get("workload") == cls][:need]
    return out


AUTO_MARGIN = 1.25  # `auto` caps: the last look's need x this, for a stray look / rate variance


def candidate_caps(plan: ConfirmationPlan, max_requests, max_duration_s, *, est_requests_per_rep: float,
                   per_rep_s: float, measured_per_rep_s: Optional[float] = None) -> dict:
    """Numeric caps for one candidate. `auto` max_requests = the last
    look's sample size (or minimum exposure need, whichever is larger)
    x AUTO_MARGIN; `auto` max_duration_s = the time
    to collect that many at THIS candidate's request rate (offered RPS for rate sweeps; discovery's
    measured requests per repetition for concurrency sweeps) x AUTO_MARGIN -- per candidate, so
    a slow (low-RPM) candidate gets the time its looks need instead of
    being INCONCLUSIVE by construction. Fixed numbers pass through
    (max_duration_s then stays a cap on the whole subject's phase)."""
    measured_s = measured_per_rep_s if measured_per_rep_s is not None else per_rep_s
    time_reps = math.ceil(plan.min_steady_state_duration_s / measured_s) if measured_s > 0 else 0
    need = max(plan.look_schedule[-1], math.ceil(time_reps * est_requests_per_rep))
    requests = math.ceil(need * AUTO_MARGIN) if max_requests == "auto" else int(max_requests)
    if max_duration_s != "auto":
        return {"max_requests": requests, "max_duration_s": float(max_duration_s), "per_candidate": False}
    if est_requests_per_rep <= 0:
        budget = 0.0
    else:
        budget = math.ceil(min(need, requests) / est_requests_per_rep) * per_rep_s * AUTO_MARGIN
    return {"max_requests": requests, "max_duration_s": round(budget, 1), "per_candidate": True}


def reachable(plan: ConfirmationPlan, *, est_requests_per_rep: float, remaining_duration_s: float,
              per_rep_s: float, looks_used: int = 0) -> bool:
    """Can the next scheduled look be reached within the caps?"""
    if looks_used >= plan.max_looks or est_requests_per_rep <= 0:
        return False
    rep_cap = plan.max_repetitions if plan.max_repetitions is not None else math.inf
    reps_by_time = int(remaining_duration_s // per_rep_s) if per_rep_s > 0 else rep_cap
    max_n = min(plan.max_requests, est_requests_per_rep * min(rep_cap, reps_by_time))
    return max_n >= plan.look_schedule[looks_used]


def highest_confirmed(results: List[ConfirmationResult]) -> Optional[ConfirmationResult]:
    """Candidates are tested highest-first and stop at the first PASS, so
    the confirmed point is the highest PASS (there is at most one)."""
    passed = [r for r in results if r.verdict == PASS]
    return max(passed, key=lambda r: r.value) if passed else None


def limits_for(gate_kwargs: dict, class_gate: Optional[Dict[str, dict]], shares: Optional[Dict[str, float]]) -> List[RateLimit]:
    """The rate checks a candidate's verdict depends on: an isolated
    subject's own, or -- for a mix -- each class's only (seeing just its
    share of the requests); a mix's blend is reported, never gated."""
    def of(prefix: str, kw: dict, share: float) -> List[RateLimit]:
        out = []
        # Latency SLOs are exceedance proportions too: P(value > threshold)
        # <= 5% (see capacity._latency_check). They need far fewer samples
        # than a 0.1% throttle limit, but the plan must still cover them.
        for key, name in (("ttft_p95_slo_ms", "ttft_p95"), ("tpot_p95_slo_ms", "tpot_p95"),
                          ("latency_p95_slo_ms", "latency_p95")):
            if kw.get(key) is not None:
                out.append(RateLimit(f"{prefix}{name}", LATENCY_EXCEEDANCE_MAX, share))
        if kw.get("throttle_rate_max", 0) > 0:
            out.append(RateLimit(f"{prefix}throttle_rate", kw["throttle_rate_max"], share))
        if kw.get("success_rate_min", 0) < 1:
            out.append(RateLimit(f"{prefix}success_rate", 1.0 - kw["success_rate_min"], share))
        return out

    limits = [] if class_gate else of("", gate_kwargs, 1.0)
    for name, kw in (class_gate or {}).items():
        limits += of(f"{name}.", kw, (shares or {}).get(name, 1.0))
    return limits
