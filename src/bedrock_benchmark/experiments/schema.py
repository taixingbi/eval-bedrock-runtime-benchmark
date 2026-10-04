"""ExperimentSpec -- the YAML-loadable description of a sweep: one or
more workload profiles and a sweep dimension (concurrency OR rate --
see runners/ for why these are kept separate, not combined into one
experiment).

Experiment files are MODEL-AGNOSTIC: no `target:`, no quota, no model
name. `load_experiment(path, model)` binds one file to one model from
the models file (see models.py), filling in the target and quota --
so the same experiment runs unchanged against every model.

Rate sweeps are written as `quota_fractions` of the bound model's
PROVIDER CEILING for each sweep subject -- min(RPM, TPM / tokens per
request), see ceiling.py -- because quotas differ by an order of
magnitude across models and RPM vs TPM binds differently per workload
shape. Fixed rps values would be far over one model's ceiling and
nowhere near another's.

Workloads come from the catalog (catalog/workloads.yaml) and SLOs from
constraints/slo.yaml -- never from the experiment, which only LISTS
workload names. Each catalog workload binds its slo_profile explicitly
(there is no default), so the same workload is judged identically in
every experiment (see constraints.py, workload.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Collection, Dict, List, Optional, Union

import math

import yaml

from ..ceiling import ProviderCeiling, provider_ceiling
from ..client import TransportConfig
from ..constraints import DEFAULT_POLICY_FILE, DEFAULT_SLO_FILE, SloConfig, TtftBudget, ttft_budget_for, load_policy, load_slo
from ..models import ModelConfig
from ..workload import DEFAULT_WORKLOADS_FILE, WorkloadProfile, load_workloads


@dataclass
class TargetConfig:
    model_id: str
    region: str = "us-east-1"


@dataclass
class QuotaSnapshot:
    """Documented context for the capacity-profile.yaml artifact and
    for a human reading the experiment -- NOT enforced by this repo.
    Real RPM/TPM enforcement is Bedrock's own; this repo only ever
    measures what actually happens, it never simulates or caps against
    a quota number itself. Named "snapshot" (not "config") because it's
    a point-in-time fact about the account/region, not something this
    tool configures or controls."""
    rpm: Optional[float] = None
    tpm: Optional[float] = None


@dataclass
class HistoryProtocol:
    idle_s: float = 300.0
    overload_quota_fraction: float = 2.0
    overload_duration_s: float = 120.0
    recovery_s: float = 120.0
    recovery_delays_s: Optional[List[float]] = None
    recovery_mode: str = "verified"
    bin_s: float = 30.0


@dataclass
class RecoveryProbe:
    """`isolation.recovery_probe` -- turns provider recovery into a checked
    experiment state instead of an assumption. After each fixed recovery
    interval, `duration_s` of load at `concurrency` (data discarded) must
    look healthy before the next phase starts: throttle_rate <=
    max_throttle_rate, success_rate >= min_success_rate, and TTFT p50
    within max_ttft_ratio x the subject's first healthy probe. Unhealthy
    -> wait `retry_cooldown_s` and probe again, up to `max_attempts`; never
    healthy -> the subject's measurement is INVALID (no capacity
    conclusion is drawn from it)."""
    duration_s: float = 20.0
    concurrency: int = 1
    max_throttle_rate: float = 0.0
    min_success_rate: float = 0.9
    max_ttft_ratio: float = 2.0
    max_attempts: int = 5
    retry_cooldown_s: float = 120.0
    baseline_fraction: Optional[float] = None
    baseline_duration_s: float = 120.0
    baseline_min_requests: int = 100
    baseline_goodput_ratio: float = 0.9


REFINEMENT_STRATEGIES = ("integer_bisection",)


@dataclass
class RefinementConfig:
    """Boundary refinement between coarse discovery and confirmation.
    The coarse sweep brackets saturation: lower = the highest point of
    the leading non-FAIL run (PASS or INCONCLUSIVE -- discovery only
    needs non-failing), upper = the first FAIL. integer_bisection then
    tests floor((lower + upper) / 2), moving lower up on a non-FAIL and
    upper down on a FAIL: 8 / 12 -> 10 -> 11 or 9. Refinement is still
    DISCOVERY-class data: it moves the candidates, never confirms one."""
    strategy: str = "integer_bisection"
    # Stop as soon as upper - lower == 1 (the edge is resolved).
    stop_when_adjacent: bool = True
    # Extra points per sweep subject. 4 resolves a 16-wide gap
    # (32 -> 48: 40, 44, 46, 47), the widest in the shipped grids.
    max_points: int = 4
    # Recovery interval before EACH refinement point: the first follows the
    # coarse sweep's overload points (two consecutive FAILs), each later
    # one may follow a refinement point that just FAILed.
    cooldown_s: float = 0.0


@dataclass
class SweepConfig:
    type: str  # "concurrency" | "rate"
    # Absolute values: concurrency levels, or rps. Resolved from
    # quota_fractions at bind time for a quota-relative rate sweep.
    values: List[float] = field(default_factory=list)
    # Rate sweeps only: fractions of each subject's provider ceiling
    # (value_rps = fraction * ceiling_rps; see ceiling.py).
    quota_fractions: Optional[List[float]] = None
    values_by_workload: Optional[Dict[str, List[float]]] = None
    # Stop discovery after this many CONSECUTIVE FAIL points (None = sweep
    # every value). Saturation is then seen twice -- enough for the
    # non-monotonic check -- without spending windows on a throttle storm
    # far past it. Values not reached are reported as skipped.
    stop_after_fails: Optional[int] = None
    stop_after_clear_fail: bool = False
    # Concurrency sweeps only -- see RefinementConfig. None = no refinement.
    refinement: Optional["RefinementConfig"] = None

    def __post_init__(self):
        if isinstance(self.refinement, dict):
            self.refinement = RefinementConfig(**self.refinement)

    @property
    def point_count(self) -> int:
        return len(self.quota_fractions) if self.quota_fractions is not None else len(self.values)


@dataclass
class ConfirmationConfig:
    """Adaptive confirmation (analysis/confirmation.py): after discovery
    picks candidates, fresh independent repetitions at each candidate
    until a pre-planned look passes and minimum exposure/sanity checks
    permit confirmation, or a look FAILs it (exact bounds), severe
    throttling stops it early, or a cap is reached (both -> INCONCLUSIVE).
    Discovery data never counts."""
    # PASS may be declared only at this many pre-planned sample sizes;
    # each test runs at 1 - alpha / (max_looks x K) for K candidates
    # (Bonferroni over looks and candidates).
    max_looks: int = 2
    # Measured load exposure required in addition to a fixed-count PASS.
    # Excludes cooldown, conditioning, per-window warmup and drain.
    min_steady_state_duration_s: float = 0.0
    continuous: bool = False
    # Per-candidate caps. The statistics are SAMPLE-COUNT driven (fixed-
    # count looks at pre-planned N); repetitions are only how data is
    # collected. max_repetitions is an optional extra cap -- None (the
    # default) caps by max_requests and max_duration_s only.
    max_repetitions: Optional[int] = None
    # A number, or "auto": the last look's sample size x AUTO_MARGIN.
    max_requests: Union[int, str] = 8000
    # A number: wall-time cap for the whole confirmation phase of one
    # sweep subject. "auto": a budget PER CANDIDATE, the time to collect
    # the last look's samples at that candidate's own request rate (from
    # discovery) x AUTO_MARGIN -- so a low-RPM model isn't INCONCLUSIVE
    # by construction just because a fixed window is too short for it.
    max_duration_s: Union[float, str] = 1800.0
    # How many of the highest non-failing discovery points to confirm:
    # tested highest-first, stopping at the first PASS, with alpha split
    # over them (analysis/confirmation.py).
    candidates: int = 1
    # Idle seconds before EACH tested candidate, so every candidate starts
    # from the same procedure -- cooldown -> conditioning -> measurement --
    # and none inherits the overload of what ran before it (discovery's
    # saturation points, or a higher candidate that just FAILed under
    # heavy throttling). Not counted against max_duration_s.
    cooldown_s: float = 0.0
    # Conditioning before each candidate's first confirmation repetition:
    # this many seconds of load at the candidate whose data is DISCARDED
    # -- so the fixed-N looks read steady state, not the burst credit a
    # rested provider bucket hands out at first. Not counted against
    # max_duration_s.
    warmup_s: float = 0.0


AUTO = "auto"


@dataclass
class MixConfig:
    """Mixed-workload experiment: instead of sweeping each workload in
    isolation, sweep ONE offered load (rate) or concurrency over a class
    mix. The only valid source of a cross-class envelope -- isolated
    per-class maxima can't be combined into one (see report.py) -- and
    only for THIS mix: R_safe(mix), never a global R_safe.

    Weights normally come from catalog/mixes.yaml (the experiment names
    the mix; `--mix` swaps it), where each mix records its `source`
    (reference_example | production_traffic_profile | synthetic) and
    `observed_from`. `assignment` (workload.WorkloadMix): stochastic
    (independent draws -- production-like randomness) or stratified
    (exact per-block composition -- measures the configured mix
    precisely)."""
    name: str
    weights: Dict[str, float] = field(default_factory=dict)
    assignment: str = "stochastic"
    source: str = "inline"          # inline = weights written in the experiment file
    observed_from: Optional[str] = None
    description: str = ""


DEFAULT_MIXES_FILE = "catalog/mixes.yaml"
MIX_SOURCES = ("reference_example", "production_traffic_profile", "synthetic")


def load_mixes(path: str = DEFAULT_MIXES_FILE) -> Dict[str, dict]:
    """The mix catalog: name -> {source, observed_from, description, weights}."""
    if not Path(path).exists():
        return {}
    raw = yaml.safe_load(Path(path).read_text()) or {}
    mixes = raw.get("mixes") or {}
    for name, cfg in mixes.items():
        cfg = cfg or {}
        unknown = sorted(set(cfg) - {"source", "observed_from", "description", "weights"})
        if unknown:
            raise ValueError(f"{path}: mixes.{name}: unknown keys {unknown}")
        if cfg.get("source") not in MIX_SOURCES:
            raise ValueError(f"{path}: mixes.{name}.source must be one of {list(MIX_SOURCES)}")
        if cfg["source"] == "production_traffic_profile" and not cfg.get("observed_from"):
            raise ValueError(f"{path}: mixes.{name}: a production_traffic_profile needs observed_from "
                             f"(where / when the weights were measured)")
        weights = cfg.get("weights")
        if not isinstance(weights, dict) or not weights or any(not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0
                                                               for w in weights.values()):
            raise ValueError(f"{path}: mixes.{name}.weights: workload -> weight > 0")
    return mixes


def _resolve_mix(raw: dict, path: str, mixes_file: str, override: Optional[str]) -> Optional[MixConfig]:
    """The experiment's mix -- a catalog reference (`mix: {name, assignment}`),
    inline weights, or `override` (`--mix NAME`) replacing the name."""
    cfg = dict(raw.get("mix") or {})
    if override is not None:
        if not cfg:
            raise NoMatchingWorkloads(f"--mix {override}: not a mixed-workload experiment")
        cfg = {"name": override, **({"assignment": cfg["assignment"]} if "assignment" in cfg else {})}
    if not cfg:
        return None
    unknown = sorted(set(cfg) - {"name", "weights", "assignment"})
    if unknown:
        raise ValueError(f"{path}: mix takes name / weights / assignment, got {unknown}")
    catalog = load_mixes(mixes_file)
    name = cfg.get("name")
    if "weights" in cfg:
        if name in catalog:
            raise ValueError(f"{path}: mix {name!r} is defined in {mixes_file} -- don't also give weights inline")
        return MixConfig(name=name, weights=cfg["weights"], assignment=cfg.get("assignment", "stochastic"))
    if name not in catalog:
        raise ValueError(f"mix {name!r} is not in {mixes_file} (has: {sorted(catalog)})")
    entry = catalog[name]
    return MixConfig(name=name, weights=dict(entry["weights"]), assignment=cfg.get("assignment", "stochastic"),
                     source=entry["source"], observed_from=entry.get("observed_from"),
                     description=(entry.get("description") or "").strip())


@dataclass
class ExperimentSpec:
    name: str  # the experiment's name -- never contains a model name
    target: TargetConfig
    workloads: List[WorkloadProfile]
    sweep: SweepConfig
    description: str = ""
    quota_snapshot: QuotaSnapshot = field(default_factory=QuotaSnapshot)
    slo: SloConfig = field(default_factory=SloConfig)
    duration_s: float = 60.0
    # Load runs for warmup_s before the measurement window opens
    # (connection pool / TLS / cold paths) -- none of it is counted.
    warmup_s: float = 0.0
    # Each sweep point runs this many times back to back; the SLO gate
    # reads the pooled window, and per-repetition metrics are kept so
    # run-to-run spread is visible.
    repetitions: int = 1
    stream: bool = True
    # POLICY (constraints/recommendation-policy.yaml), not measurement:
    # back-off from the statistically confirmed point.
    provider_headroom: float = 0.20
    # Back-off from the provider CEILING (quota) for production: a sweep
    # that passed above quota may have ridden Bedrock's short-window
    # burst allowance, which isn't sustainable, so production rate is
    # min(statistically_confirmed x (1 - provider_headroom), ceiling x (1 - quota_headroom)).
    quota_headroom: float = 0.10
    confirmation: Optional[ConfirmationConfig] = None
    # reference: produces the production admission envelope, and may only
    # list reference workloads. admission_calibration: statistically
    # confirmed capacity points per workload shape (input / output tokens,
    # SLO), from which a gateway DERIVES admission classes or weights --
    # no envelope, no headroom. characterization: measurement only.
    purpose: str = "reference"
    # Concurrency sweeps only: a closed-loop worker that gets a 429 waits
    # this long before its next request instead of re-firing at once (a
    # 429 returns in milliseconds, so without it one throttled worker
    # becomes a retry storm that dominates the throttle count). The
    # throttle is still recorded; 0 = re-fire immediately.
    throttle_pause_s: float = 0.0
    seed: Optional[int] = None
    workload_rotation_index: int = 0
    transport: TransportConfig = field(default_factory=TransportConfig)
    mix: Optional[MixConfig] = None
    history_protocol: Optional[HistoryProtocol] = None
    baseline_context: Optional[dict] = None
    burst_protocol: bool = False
    retest: Optional[dict] = None
    # `isolation: {inter_subject_cooldown_s}` -- a fixed recovery interval
    # between sweep subjects (workloads) to reduce carry-over: all subjects
    # share one model's quota, and a subject's overload points otherwise
    # shape the next one's results -- C_safe(W_i | history) instead of
    # C_safe(W_i). A policy, not a proven reset: the provider's quota /
    # burst / routing state isn't observable.
    inter_subject_cooldown_s: float = 0.0
    # `isolation.recovery_probe` -- see RecoveryProbe. None = fixed
    # intervals only (no recovery check, no suspect-point re-measure).
    recovery_probe: Optional["RecoveryProbe"] = None
    # Post-run check: Bedrock-REPORTED input_tokens p50 vs requested.
    # Outside this, the class's workload_validation is valid: false
    # (the 4-chars/token padding estimate missed for this model).
    workload_validation_tolerance_pct: float = 10.0
    # Post-run check: Bedrock-reported output_tokens p50 vs the target
    # (max_tokens). Looser than input -- models legitimately stop a bit
    # early -- but a "512 out" class that really emits 110 is flagged.
    output_validation_tolerance_pct: float = 25.0
    # Padding calibration (calibration.py): strategy from the model entry,
    # tolerance for counted vs requested input tokens.
    token_counting: str = "auto"
    calibration_tolerance_pct: float = 2.0
    output_burndown: float = 1.0  # from constraints/quota.yaml, for the artifact
    quota_account: Optional[str] = None  # the AWS account the quota was taken for
    # Every profile from constraints/slo.yaml; each workload names one via
    # WorkloadProfile.slo_profile. For a loaded spec, `slo` above is the
    # STRICTEST success/throttle limits among the profiles its workloads
    # use (no latency) -- used for the reported sample-size figure
    # (min_requests_to_resolve_throttle_slo), never as a mix's gate: a
    # mix is gated per class only (capacity.point_verdict).
    slo_profiles: Dict[str, SloConfig] = field(default_factory=dict)
    # The models-file entry this spec is bound to (None only for specs
    # built directly in code, e.g. tests).
    model_name: Optional[str] = None
    # Per sweep subject (workload name, or the mix name): the quota's
    # theoretical request ceiling for that subject's token shape.
    provider_ceilings: Dict[str, ProviderCeiling] = field(default_factory=dict)

    ttft_budgets: Dict[str, TtftBudget] = field(default_factory=dict)

    def ttft_budget_name(self, workload_name: str) -> Optional[str]:
        if not self.ttft_budgets:
            return None
        workload = next(w for w in self.workloads if w.name == workload_name)
        return ttft_budget_for(self.ttft_budgets, workload.input_tokens)[0]

    def slo_for(self, workload_name: str) -> SloConfig:
        """Resolve input-length TTFT, profile TPOT/reliability and workload E2E.
        Legacy SLO files without ttft_budgets retain profile-based TTFT.
        """
        workload = next((w for w in self.workloads if w.name == workload_name), None)
        slo = self.slo
        if workload is not None and workload.slo_profile is not None:
            slo = self.slo_profiles[workload.slo_profile]
        if workload is not None and self.ttft_budgets:
            _, budget = ttft_budget_for(self.ttft_budgets, workload.input_tokens)
            slo = replace(slo, ttft_p95_ms=budget.ttft_p95_ms)
        if workload is not None and workload.latency_p95_ms is not None:
            slo = replace(slo, latency_p95_ms=workload.latency_p95_ms)
        return slo

    @property
    def mode(self) -> str:
        if self.retest is not None:
            return "sustain"
        return "history" if self.history_protocol is not None else "sweep"

    @property
    def subject_names(self) -> List[str]:
        return [self.mix.name] if self.mix is not None else [w.name for w in self.workloads]

    def sweep_values(self, subject_name: str) -> List[float]:
        """Absolute sweep values for one subject -- quota-relative rate
        sweeps resolve against that subject's own provider ceiling."""
        if self.sweep.values_by_workload is not None:
            return list(self.sweep.values_by_workload[subject_name])
        if self.sweep.quota_fractions is None:
            return list(self.sweep.values)
        ceiling = self.provider_ceilings[subject_name].rps
        return [round(f * ceiling, 4) for f in self.sweep.quota_fractions]


class NoMatchingWorkloads(Exception):
    """Skip when filters leave no workloads or a template awaits baseline evidence."""


_MODEL_KEYS = ("target", "quota_snapshot", "quota")
_SLO_KEYS = ("slo", "slo_profiles", "ttft_budgets")
_POLICY_KEYS = ("provider_headroom", "quota_headroom")


def load_experiment(
    path: str, model: ModelConfig, *, slo_file: str = DEFAULT_SLO_FILE, workloads_file: str = DEFAULT_WORKLOADS_FILE,
    only_slo_profiles: Optional[Collection[str]] = None, policy_file: str = DEFAULT_POLICY_FILE,
    mix: Optional[str] = None, mixes_file: str = DEFAULT_MIXES_FILE, retest: Optional[dict] = None,
) -> ExperimentSpec:
    """only_slo_profiles (e.g. {"gold"}) keeps just the workloads bound to
    those profiles. An isolated sweep keeps its matching workloads; a mix
    runs only if EVERY class matches (dropping classes would silently
    make it a different mix); with nothing left, NoMatchingWorkloads is
    raised so the caller can skip the experiment."""
    raw = yaml.safe_load(Path(path).read_text())
    if raw.get("baseline_required"):
        raise NoMatchingWorkloads(
            "Experiment B needs confirmed Experiment A results; run scripts/prepare_context_history.py first"
        )

    present = [k for k in _MODEL_KEYS if k in raw]
    if present:
        raise ValueError(
            f"{path}: experiments are model-agnostic -- remove {present}; models live in "
            f"catalog/models.yaml and quotas in constraints/quota.yaml"
        )
    present = [k for k in _SLO_KEYS if k in raw]
    if present:
        raise ValueError(
            f"{path}: remove {present} -- SLOs are defined once in {slo_file}; "
            f"give a workload `slo_profile: <name>` to use a non-default one"
        )
    present = [k for k in _POLICY_KEYS if k in raw]
    if present:
        raise ValueError(
            f"{path}: remove {present} -- headroom is recommendation POLICY, defined once in {policy_file}; "
            f"an experiment defines only what is measured"
        )
    mix_config = _resolve_mix(raw, path, mixes_file, mix)
    if mix_config is not None:
        # A mix experiment's workloads ARE its mix's classes -- also when
        # --mix swaps in a mix over other workloads.
        listed = raw.get("workloads")
        if mix is not None or listed is None:
            raw = {**raw, "workloads": list(mix_config.weights)}
        elif set(listed) != set(mix_config.weights):
            raise ValueError(f"{path}: workloads {listed} must be exactly mix {mix_config.name!r}'s classes "
                             f"{sorted(mix_config.weights)} (or omit `workloads:`)")
        raw = {**raw, "mix": {"name": mix_config.name, "weights": mix_config.weights}}
    slos = load_slo(slo_file)
    policy = load_policy(policy_file)
    sweep = raw["sweep"]
    transport = raw.get("transport") or {}
    workloads = _resolve_workloads(raw.get("workloads"), path, workloads_file)
    roles = raw.get("workload_roles") or {}
    if not isinstance(roles, dict) or set(roles) - {w.name for w in workloads}:
        raise ValueError(f"{path}: workload_roles must map selected workload names to reference_control")
    for w in workloads:
        if w.name in roles and (roles[w.name] != "reference_control" or w.role != "reference"):
            raise ValueError(f"{path}: workload_roles only permits reference -> reference_control")
    if roles and (raw.get("purpose") == "reference" or mix_config is not None):
        raise ValueError(f"{path}: reference controls require an isolated non-reference experiment")
    workloads = [replace(w, role=roles.get(w.name, w.role)) for w in workloads]

    if only_slo_profiles is not None:
        workloads = _filter_by_slo_profile(workloads, raw, set(only_slo_profiles), slos.profiles, slo_file)

    spec = ExperimentSpec(
        name=raw["name"],
        description=raw.get("description", ""),
        target=TargetConfig(model_id=model.model_id, region=model.region),
        quota_snapshot=QuotaSnapshot(rpm=model.quota_rpm, tpm=model.quota_tpm),
        slo=SloConfig(),  # replaced by the strictest used gate below, after validation
        workloads=workloads,
        duration_s=raw.get("duration_s", 60.0),
        warmup_s=raw.get("warmup_s", 0.0),
        repetitions=raw.get("repetitions", 1),
        stream=raw.get("stream", True),
        sweep=SweepConfig(**sweep),
        provider_headroom=policy.headroom_fraction,
        quota_headroom=policy.quota_headroom_fraction,
        confirmation=ConfirmationConfig(**raw["confirmation"]) if raw.get("confirmation") else None,
        throttle_pause_s=raw.get("throttle_pause_s", 0.0),
        purpose=_purpose(raw, workloads, path, workloads_file),
        seed=raw.get("seed"),
        workload_rotation_index=raw.get("workload_rotation_index", 0),
        transport=TransportConfig(**transport),
        mix=mix_config,
        burst_protocol=raw.get("burst_protocol", False),
        history_protocol=HistoryProtocol(**raw["history_protocol"]) if raw.get("history_protocol") else None,
        baseline_context=raw.get("baseline_context"),
        inter_subject_cooldown_s=_isolation(raw, path),
        recovery_probe=_recovery_probe(raw, path),
        workload_validation_tolerance_pct=raw.get("workload_validation_tolerance_pct", 10.0),
        output_validation_tolerance_pct=raw.get("output_validation_tolerance_pct", 25.0),
        slo_profiles=dict(slos.profiles),
        ttft_budgets=dict(slos.ttft_budgets),
        token_counting=model.token_counting,
        output_burndown=model.output_burndown,
        quota_account=model.account,
        calibration_tolerance_pct=raw.get("calibration_tolerance_pct", 2.0),
        model_name=model.name,
    )
    if retest is not None:
        if spec.purpose != "admission_calibration" or spec.sweep.type != "concurrency" or spec.mix is not None:
            raise ValueError("retest requires an isolated admission_calibration concurrency experiment")
        if set(retest) != {"workload", "concurrency", "duration_s"}:
            raise ValueError("retest requires workload, concurrency and duration_s")
        c, seconds = retest["concurrency"], retest["duration_s"]
        if (not isinstance(c, int) or isinstance(c, bool) or c < 1
                or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError("retest concurrency must be a positive integer and duration_s finite and > 0")
        selected = [w for w in spec.workloads if w.name == retest["workload"]]
        if not selected:
            raise ValueError(f"retest workload {retest['workload']!r} is not selected by this experiment / SLO filter")
        if spec.confirmation is None:
            raise ValueError("retest requires a confirmation configuration")
        spec.workloads = selected
        spec.retest = dict(retest)
        spec.description = (f"Focused retest of {selected[0].name} at C={c}: continuous {seconds:g}s "
                            "discovery windows followed by independent confirmation; finite-duration evidence.")
        spec.sweep = SweepConfig(type="concurrency", values=[c])
        spec.duration_s = seconds
        spec.repetitions = 1
        spec.inter_subject_cooldown_s = 0
        spec.transport = replace(spec.transport, max_connections=max(128, spec.transport.max_connections))
        spec.confirmation = replace(spec.confirmation, candidates=1, max_requests="auto", max_duration_s="auto",
                                    max_repetitions=None, min_steady_state_duration_s=seconds)
    _validate(spec)
    spec.slo = _strictest_gate([spec.slo_profiles[w.slo_profile] for w in spec.workloads])
    spec.provider_ceilings = _ceilings(spec, model)
    _validate_sweep(spec, model, path)
    if spec.baseline_context is not None:
        from ..context_history import validate_binding
        validate_binding(spec)
    return spec


def _recovery_probe(raw: dict, path: str) -> Optional[RecoveryProbe]:
    cfg = (raw.get("isolation") or {}).get("recovery_probe")
    if not cfg:
        return None
    probe = RecoveryProbe(**cfg)
    if (probe.baseline_fraction is not None and not 0 < probe.baseline_fraction < 1
            or not math.isfinite(probe.baseline_duration_s) or probe.baseline_duration_s <= 0
            or probe.baseline_min_requests < 1 or not 0 < probe.baseline_goodput_ratio <= 1):
        raise ValueError(f"{path}: invalid baseline recovery policy")
    if (probe.duration_s <= 0 or probe.concurrency < 1 or probe.max_attempts < 1 or probe.retry_cooldown_s < 0
            or not 0 <= probe.max_throttle_rate <= 1 or not 0 <= probe.min_success_rate <= 1
            or probe.max_ttft_ratio <= 1):
        raise ValueError(f"{path}: isolation.recovery_probe: duration_s > 0, concurrency / max_attempts >= 1, "
                         f"retry_cooldown_s >= 0, rates in [0, 1], max_ttft_ratio > 1")
    return probe


def _isolation(raw: dict, path: str) -> float:
    isolation = raw.get("isolation") or {}
    unknown = sorted(set(isolation) - {"inter_subject_cooldown_s", "recovery_probe"})
    if unknown:
        raise ValueError(f"{path}: isolation takes inter_subject_cooldown_s / recovery_probe, got {unknown}")
    value = float(isolation.get("inter_subject_cooldown_s", 0.0))
    if value < 0:
        raise ValueError(f"{path}: isolation.inter_subject_cooldown_s must be >= 0")
    return value


def _filter_by_slo_profile(workloads, raw, only, defined, slo_file: str) -> List[WorkloadProfile]:
    unknown = sorted(only - set(defined))
    if unknown:
        raise ValueError(f"--slo-profile {unknown} not defined in {slo_file} (has: {sorted(defined)})")
    if raw.get("mix"):
        classes = set((raw["mix"].get("weights") or {}))
        others = sorted(f"{w.name} ({w.slo_profile})" for w in workloads if w.name in classes and w.slo_profile not in only)
        if others:
            raise NoMatchingWorkloads(
                f"mix {raw['mix'].get('name')!r} also includes {', '.join(others)} -- a partial mix is a different mix"
            )
        return workloads
    kept = [w for w in workloads if w.slo_profile in only]
    if not kept:
        raise NoMatchingWorkloads(f"no workloads bound to {sorted(only)}")
    return kept


EXPERIMENT_PURPOSES = ("reference", "admission_calibration", "characterization")
# Purposes whose output feeds gateway configuration -- so it must come
# from independent confirmation, never from discovery alone.
CONFIRMED_PURPOSES = ("reference", "admission_calibration")


def _purpose(raw: dict, workloads: List[WorkloadProfile], path: str, workloads_file: str) -> str:
    purpose = raw.get("purpose")
    if purpose not in EXPERIMENT_PURPOSES:
        raise ValueError(f"{path}: needs `purpose:` one of {list(EXPERIMENT_PURPOSES)} -- reference experiments "
                         f"produce the production admission envelope, admission_calibration ones confirmed "
                         f"points for deriving gateway admission classes, characterization ones only measure")
    if purpose in CONFIRMED_PURPOSES and not raw.get("confirmation"):
        raise ValueError(f"{path}: {purpose} experiments require independent confirmation -- add a "
                         f"`confirmation:` block (discovery only picks candidates; confirmation is the "
                         f"only source of a capacity that feeds gateway configuration)")
    if purpose == "reference":
        other = [w.name for w in workloads if w.role != "reference"]
        if other:
            raise ValueError(f"{path}: a reference experiment may only list reference workloads; {other} are "
                             f"not (role in {workloads_file}) -- use a characterization experiment for them")
    return purpose


def _resolve_workloads(names, path: str, workloads_file: str) -> List[WorkloadProfile]:
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError(
            f"{path}: `workloads:` must be a list of workload names from {workloads_file} "
            f"(e.g. [short_chat]) -- shapes and SLO bindings are defined there, not in experiments"
        )
    catalog = load_workloads(workloads_file)
    unknown = [n for n in names if n not in catalog]
    if unknown:
        raise ValueError(f"{path}: unknown workloads {unknown}; {workloads_file} defines {sorted(catalog)}")
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: workloads listed twice: {names}")
    return [catalog[n] for n in names]


def _strictest_gate(profiles: List[SloConfig]) -> SloConfig:
    """Success/throttle gates only -- latency always comes from each
    workload's own profile."""
    confidences = [p.confidence for p in profiles if p.confidence is not None]
    return SloConfig(
        success_rate_min=max(p.success_rate_min for p in profiles),
        throttle_rate_max=min(p.throttle_rate_max for p in profiles),
        confidence=max(confidences) if confidences else None,
    )


def _ceilings(spec: ExperimentSpec, model: ModelConfig) -> Dict[str, ProviderCeiling]:
    kwargs = dict(rpm=model.quota_rpm, tpm=model.quota_tpm, output_burndown=model.output_burndown)
    by_name = {w.name: w for w in spec.workloads}
    if spec.mix is not None:
        classes = [(by_name[n], w) for n, w in spec.mix.weights.items() if n in by_name]
        return {spec.mix.name: provider_ceiling(classes, **kwargs)} if classes else {}
    return {w.name: provider_ceiling([(w, 1.0)], **kwargs) for w in spec.workloads}


def _validate_sweep(spec: ExperimentSpec, model: ModelConfig, path: str) -> None:
    sweep = spec.sweep
    if sweep.values_by_workload is not None:
        grids = sweep.values_by_workload
        if (sweep.type != 'rate' or spec.mix is not None or sweep.values
                or sweep.quota_fractions is not None or not isinstance(grids, dict)
                or not grids or set(spec.subject_names) - set(grids)):
            raise ValueError(f'{path}: values_by_workload requires isolated rate sweeps, a grid for every workload, and no other rate source')
        for name, values in grids.items():
            if (not isinstance(values, list) or not values
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values)
                    or any(a >= b for a, b in zip(values, values[1:]))):
                raise ValueError(f'{path}: values_by_workload.{name} must be positive finite increasing rates')
    elif sweep.quota_fractions is None:
        if not sweep.values:
            raise ValueError(f"{path}: sweep needs `values` or (rate only) `quota_fractions`")
    else:
        if sweep.values:
            raise ValueError(f"{path}: sweep takes `values` OR `quota_fractions`, not both")
        if sweep.type != "rate":
            raise ValueError(f"{path}: quota_fractions only applies to rate sweeps")
        if not sweep.quota_fractions or any(f <= 0 for f in sweep.quota_fractions):
            raise ValueError(f"{path}: quota_fractions must be non-empty and all > 0")
        missing = [n for n in spec.subject_names if spec.provider_ceilings.get(n) is None
                   or spec.provider_ceilings[n].rps is None]
        if missing:
            raise ValueError(
                f"{path}: quota-relative rate sweep needs quota.rpm or quota.tpm for model {model.name!r} "
                f"in the models file (scripts/fetch_quota.py --all)"
            )
    if sweep.type == "concurrency" and sweep.values and max(sweep.values) > spec.transport.max_connections:
        raise ValueError(
            f"{path}: concurrency {max(sweep.values):g} exceeds transport.max_connections "
            f"({spec.transport.max_connections}) -- the connection pool would cap in-flight calls, "
            f"measuring the client instead of Bedrock"
        )


def _validate(spec: ExperimentSpec) -> None:
    if type(spec.workload_rotation_index) is not int or spec.workload_rotation_index < 0:
        raise ValueError("workload_rotation_index must be a nonnegative integer")
    if type(spec.burst_protocol) is not bool or type(spec.sweep.stop_after_clear_fail) is not bool:
        raise ValueError("burst_protocol and stop_after_clear_fail must be booleans")
    if spec.confirmation and type(spec.confirmation.continuous) is not bool:
        raise ValueError("confirmation.continuous must be a boolean")
    if spec.burst_protocol and (spec.sweep.type != "rate" or spec.purpose != "characterization"
            or spec.confirmation is not None or spec.recovery_probe is None
            or spec.recovery_probe.baseline_fraction is None):
        raise ValueError("burst_protocol requires rate characterization with baseline recovery and no confirmation")

    for name in ("provider_headroom", "quota_headroom"):
        if not 0 <= getattr(spec, name) < 1:
            raise ValueError(f"{name} must be in [0, 1)")
    h = spec.history_protocol
    if h is not None:
        if (spec.purpose != "characterization" or spec.sweep.type != "rate" or spec.mix is not None
                or spec.confirmation is not None or (h.recovery_mode == "verified" and spec.recovery_probe is None)
                or spec.sweep.stop_after_fails is not None or spec.sweep.refinement is not None or spec.warmup_s != 0):
            raise ValueError("history_protocol requires isolated characterization rate measurements, probes in verified mode, "
                             "zero warmup, no confirmation/refinement/early-stop")
        if h.recovery_mode not in {"verified", "fixed_wait"}:
            raise ValueError("history_protocol.recovery_mode must be verified or fixed_wait")
        if h.recovery_delays_s is not None and (
                not isinstance(h.recovery_delays_s, list) or not h.recovery_delays_s or
                any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in h.recovery_delays_s)):
            raise ValueError("history_protocol.recovery_delays_s must be a nonempty list of positive finite delays")
        for name in ("idle_s", "overload_quota_fraction", "overload_duration_s", "recovery_s", "bin_s"):
            value = getattr(h, name)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"history_protocol.{name} must be finite and > 0")
        if h.overload_quota_fraction <= 1 or h.bin_s > spec.duration_s:
            raise ValueError("history_protocol requires overload > 1x nominal ceiling and bin_s <= duration_s")
    c = spec.confirmation
    def bad_cap(value, low) -> bool:
        return (value != AUTO if isinstance(value, str) else
                not isinstance(value, (int, float)) or not math.isfinite(value) or value < low)

    if c is not None and (c.max_looks < 1 or (c.max_repetitions is not None and c.max_repetitions < 1)
                          or bad_cap(c.max_requests, 1) or bad_cap(c.max_duration_s, 1e-9)
                          or c.candidates < 1 or c.cooldown_s < 0 or c.warmup_s < 0
                          or bad_cap(c.min_steady_state_duration_s, 0)
                          or isinstance(c.min_steady_state_duration_s, str)):
        raise ValueError("confirmation: max_looks, max_repetitions, max_requests, candidates must be >= 1, "
                         "max_duration_s > 0 (max_requests / max_duration_s may be `auto`), cooldown_s >= 0 "
                         "and warmup_s / min_steady_state_duration_s >= 0")
    if spec.throttle_pause_s < 0:
        raise ValueError(f"throttle_pause_s must be >= 0, got {spec.throttle_pause_s}")
    if spec.throttle_pause_s > 0 and spec.sweep.type != "concurrency":
        raise ValueError("throttle_pause_s applies to concurrency sweeps only -- a rate sweep's arrivals are "
                         "open-loop and never wait on a response")
    r = spec.sweep.refinement
    if r is not None:
        if spec.sweep.type != "concurrency":
            raise ValueError("sweep.refinement applies to concurrency sweeps only")
        if r.strategy not in REFINEMENT_STRATEGIES:
            raise ValueError(f"sweep.refinement.strategy must be one of {list(REFINEMENT_STRATEGIES)}, got {r.strategy!r}")
        if not r.stop_when_adjacent:
            raise ValueError("sweep.refinement.stop_when_adjacent must be true -- adjacent integers are already resolved")
        if r.max_points < 1:
            raise ValueError(f"sweep.refinement.max_points must be >= 1, got {r.max_points}")
        if r.cooldown_s < 0:
            raise ValueError(f"sweep.refinement.cooldown_s must be >= 0, got {r.cooldown_s}")
    if spec.sweep.stop_after_fails is not None and spec.sweep.stop_after_fails < 1:
        raise ValueError(f"sweep.stop_after_fails must be >= 1, got {spec.sweep.stop_after_fails}")
    if spec.repetitions < 1:
        raise ValueError(f"repetitions must be >= 1, got {spec.repetitions}")
    if spec.warmup_s < 0 or spec.duration_s <= 0:
        raise ValueError("warmup_s must be >= 0 and duration_s > 0")
    if spec.slo.confidence is not None and not 0 < spec.slo.confidence < 1:
        raise ValueError(f"slo.confidence must be in (0, 1), got {spec.slo.confidence}")
    from ..calibration import STRATEGIES
    if spec.token_counting not in STRATEGIES:
        raise ValueError(f"token_counting must be one of {STRATEGIES}, got {spec.token_counting!r}")
    unknown_profiles = sorted({w.slo_profile for w in spec.workloads if w.slo_profile} - set(spec.slo_profiles))
    if unknown_profiles:
        raise ValueError(f"workloads bind SLO profiles not defined in the SLO file: {unknown_profiles}")
    for name, profile in spec.slo_profiles.items():
        if profile.confidence is not None and not 0 < profile.confidence < 1:
            raise ValueError(f"slo_profiles.{name}.confidence must be in (0, 1)")
    if spec.mix is not None:
        names = {w.name for w in spec.workloads}
        unknown = set(spec.mix.weights) - names
        if unknown:
            raise ValueError(f"mix {spec.mix.name!r} references undefined workloads: {sorted(unknown)}")
        if not spec.mix.weights or any(not math.isfinite(w) or w <= 0 for w in spec.mix.weights.values()):
            raise ValueError(f"mix {spec.mix.name!r} needs at least one workload, all weights > 0")
        from ..workload import MIX_ASSIGNMENTS
        if spec.mix.assignment not in MIX_ASSIGNMENTS:
            raise ValueError(f"mix {spec.mix.name!r}: assignment must be one of {list(MIX_ASSIGNMENTS)}")
        if spec.mix.assignment == "stratified":
            from ..workload import block_counts
            total = sum(spec.mix.weights.values())
            block_counts({name: weight / total for name, weight in spec.mix.weights.items()})
