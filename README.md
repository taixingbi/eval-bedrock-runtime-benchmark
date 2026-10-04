# eval-bedrock-runtime-benchmark

Measures the **SLO-qualified operating envelope of a Bedrock inference
profile** -- how much load a model serves within its SLO, statistically
confirmed -- and turns it into an admission-envelope recommendation.

## The paved road

One command at a time (setup: [Install](#install)):

```bash
bedrock-benchmark doctor --model nova-micro
```

```bash
bedrock-benchmark plan capacity-reference-concurrency --model nova-micro
```

```bash
bedrock-benchmark pilot capacity-reference-concurrency --model nova-micro
```

```bash
caffeinate -i bedrock-benchmark run capacity-reference-concurrency --model nova-micro --ticket CAP-123 --purpose "model onboarding"
```

```bash
bedrock-benchmark summary results/run-all-<timestamp>
```

```bash
bedrock-benchmark validate results/
```

```bash
bedrock-benchmark publish results/run-all-<timestamp> --destination s3://<team-bucket>/capacity
```

| Step | Answers | AWS calls |
|---|---|---|
| `doctor` | Is this machine / account / model / config ready? Python, boto3, git clean, config valid, identity + region, quota file = live, model access, ConverseStream, token-counting strategy -> `READY` / `NOT READY -- fix: ...` | 2-3 one-token requests |
| `plan` | What will run, against which quota and ceilings, for how long? | STS only |
| `pilot` | Does each workload reach its shape and SLO at all? | a few requests |
| `run` | The benchmark -> `capacity-profile.yaml` + raw JSONL, then a human summary | the full run |
| `summary` | Can I trust this run? What did we learn? Is it production usable (never, alone)? What next? | none |
| `validate` | Do repeated runs agree across days / times? -> `temporal-capacity-profile.yaml` with a status per envelope: `VALID`, `VALID_CONSERVATIVE`, `INSUFFICIENT_EVIDENCE` | none |
| `publish` | Copies the run to the team's shared, immutable location with a `manifest.yaml` (owner, ticket, sha256 per file) | S3 (or a directory) |
| `validate-profile` | Does an artifact conform to the machine contract ([schemas](src/bedrock_benchmark/schemas/))? | none |

Rules the CLI enforces:

- **No broad expensive defaults.** `plan` / `pilot` / `run` need `--model`
  (repeatable) or an explicit `--all-models`.
- **Every run has an owner.** `run` records a `run:` block in the profile:
  `run_id`, `owner` (default `$USER`), `purpose`, `ticket`, `environment`
  (default `dev`).
- **A single run is never production config.** Only a
  temporal-capacity-profile entry with status `VALID` or
  `VALID_CONSERVATIVE` carries a `production_capacity_input`.

## Who uses what

| Role | Uses | Owns |
|---|---|---|
| Benchmark maintainer | everything; `docs/`, `tests/` | methodology, statistics, experiment YAMLs, the artifact schemas |
| Model onboarding engineer | `doctor` -> `plan` -> `pilot` -> `run` -> `summary` -> `publish` | `catalog/models.yaml`, `constraints/quota.yaml` entries for the new model |
| Platform engineer | published `temporal-capacity-profile.yaml` (`production_capacity_input`) | turning a VALID envelope into capacity planning |
| Gateway engineer | `capacity-profile.yaml` via `eval-bedrock-gateway`'s `capacity_review`, validated with `validate-profile` | the gateway's admission config -- never this repo |
| SRE | `summary`, `validate`, the publish manifest | deciding when a profile is stale and re-measuring |
| Service / product owner | -- | `constraints/slo.yaml` (gold / silver / bronze): SLOs are policy inputs; the benchmark never tunes or relaxes them |

Four layers, each usable without the one below it:

| Layer | Interface | For |
|---|---|---|
| CLI | `bedrock-benchmark ...` | everyone |
| Engine | `bedrock_benchmark.batch.run_batch`, `pilot`, `drift.build_temporal_profile`, `doctor.run_doctor` | automation |
| Artifact API | `capacity-profile.yaml` (schema v23), `temporal-capacity-profile.yaml` (v1), JSON Schemas in `src/bedrock_benchmark/schemas/` | consumers |
| Consumers | `eval-bedrock-gateway` (`capacity_review`), `eval-bedrock-platform` | policy and deployed validation |

## What problem does this solve?

A gateway in front of Bedrock needs limits -- how many requests in
flight, how many per second -- per model and per kind of request.
Guessing them either wastes capacity or lets a model fall over. This
repo measures them directly against Bedrock and answers:

> Given workload W, provider environment E, quota Q and SLO S, what
> operating region satisfies S -- observed, statistically confirmed,
> and safe to configure after headroom?

Two framing rules:

- **What is measured is not the bare model** but the Bedrock
  inference-profile operating envelope: model + Bedrock serving stack +
  inference-profile routing + account/region quota + provider
  conditions at measurement time. A result like "6.67 rps, TTFT 700 ms,
  C=4" describes that whole stack at that time -- never a model's
  intrinsic capacity.
- **The SLO is an input, not a finding.** Gold / silver / bronze
  (`constraints/slo.yaml`) are externally supplied policy; the benchmark
  never derives, tunes or relaxes them from measurements.
- **Capacity = the highest statistically confirmed SLO-compliant
  operating point.** SLO goodput is reported as an observed metric but
  never selects it; headroom is policy, applied only in the
  recommendation.

## Architecture

```
catalog/models.yaml          which models
catalog/workloads.yaml       which requests (shape + SLO profile + role) ─┐
experiments/*.yaml           how to load them (purpose + workloads + sweep) ┤
constraints/slo.yaml         SLO: what quality we require (policy)      ─┤
constraints/quota.yaml       quota: what the provider allows            ─┤
constraints/recommendation-policy.yaml   headroom (policy)              ─┤
                                                                          v
   coarse sweep ──> bracket + refinement ──> adaptive confirmation ──> verdicts (PASS / FAIL / INCONCLUSIVE)
                                                                          |
                                                                          v
                capacity-profile.yaml:  measurement  +  recommendation.admission_envelope
                                                                          |
                                          ───────────── contract ─────────┼──────────────
                                                                          v
                         eval-bedrock-gateway  (scripts/capacity_review.py maps the
                         envelope onto its own global / tenant / quota knobs)
```

Every call goes **directly to Bedrock** (`boto3` Converse /
ConverseStream): no API Gateway, auth, admission control, tenant quota
or queue in the path. The producer knows nothing about its consumers --
there is no gateway config schema in this repo.

| Repo | Question |
|---|---|
| `eval-bedrock-gateway` | Is the gateway's own implementation correct? How does it map an envelope onto its limits? |
| `eval-bedrock-platform` | Does the *deployed platform* (gateway + Bedrock) behave correctly under real workload? |
| `eval-bedrock-runtime-benchmark` | What is the Bedrock inference-profile operating envelope, independent of any gateway? |

## Experiments

| Experiment | Purpose | Sweep | Per model |
|---|---|---|---|
| `capacity-reference-rate` | reference | 0.25–1× ceiling; stop at clear FAIL; recovery and continuous confirmation | ~74 min |
| `capacity-reference-concurrency` | reference | each reference workload alone, concurrency 1..48 until clear FAIL; refinement and continuous confirmation | up to ~170 min |
| `capacity-burst-rate` | characterization | opt-in 90s overload pulses at 1.25 / 1.5 / 2 / 2.5× ceiling, with recovery after each | ~81 min |
| `capacity-mix-rate` | reference | mixed-rate calibration: 0.25x-2.5x ceiling for a workflow mix (`--mix`); default 60/30/10 is a reference example | ~32 min |
| `capacity-shape-concurrency` | admission_calibration | the 4 non-reference shapes plus a long_generation reference control x concurrency 1..48 until 2 consecutive FAILs, each under its own SLO | ~1-2 h |

Only **reference** experiments, on the three **reference** workloads (one
per SLO tier), produce an admission-envelope recommendation.
`capacity-shape-concurrency` is **admission calibration**: for the other four catalog
shapes it produces statistically confirmed `calibration_point`s --
C_safe = f(input/output tokens, SLO, quota, provider conditions):

```
calibration_point -> workload-specific admission evidence -> policy derivation (gateway) -> mixed validation (eval-bedrock-platform)
```

A calibration point is evidence, not a config value: no admission
envelope, no headroom. Different workload shapes may require different concurrency to reach the same provider rate ceiling; therefore concurrency must not be interpreted as a workload cost weight. Rates are kept apart -- `attempted_rps` (inflated by fast 429s under overload), `successful_rps` (served), `throttled_rps`, `slo_goodput_rps` -- plus `ceiling_ratio` (served / nominal ceiling); all are observations of what concurrency and latency produced, not a tested rate like `capacity-reference-rate`'s `sustained_rps`. Calibration points are isolated-workload measurements; per-shape C_safe values don't combine mathematically into a global policy, and any policy derived from them must be validated under representative mixed traffic through the deployed gateway (`eval-bedrock-platform`) before production use -- this repo calls Bedrock directly and never validates gateway policy.

Each reference workload ends up with both an isolated concurrency and an
isolated rate envelope:

| Class | `max_inflight` (`capacity-reference-concurrency`) | `sustained_rps` (`capacity-reference-rate`) |
|---|---|---|
| `short_chat` (gold) | ✓ | ✓ |
| `rag_answer` (silver) | ✓ | ✓ |
| `long_generation` (bronze) | ✓ | ✓ |
| the 60/30/10 mix | -- | ✓ (`capacity-mix-rate`) |

Per-class `max_inflight` (and `sustained_rps`) values are **isolated** limits -- each holds for that class running alone (`scope: isolated_workload_class`). They are not additive across classes and are not a global limit; only `capacity-mix-rate` (`scope: workload_mix`) measures classes together.

What each experiment gives a gateway:

| Experiment | Produces | Gateway use |
|---|---|---|
| `capacity-reference-concurrency` | per-class isolated `max_inflight` for the three reference workloads | reference workload `C_admission` |
| `capacity-reference-rate` | per-class isolated `sustained_rps` for the same three | reference workload `R_admission` |
| `capacity-shape-concurrency` | confirmed `calibration_point` per non-reference workload shape (no envelope, no headroom) | extra workload-shape admission calibration points |
| `capacity-mix-rate` | `sustained_rps` for ONE explicit mix (`scope: workload_mix`) | mix-scoped total-rate `R_admission(mix)` |

None of these validates gateway policy: every call goes straight to
Bedrock. The benchmark produces backend admission evidence; the gateway
derives its config from it; `eval-bedrock-platform` validates the
deployed gateway under production-like mixed traffic.

**Each workflow has its own `R_safe(mix)`.** Define weights from that workflow's
production traffic in `catalog/mixes.yaml`, with `source: production_traffic_profile`
and `observed_from` identifying the data and time range. Measure each workflow separately:

```sh
bedrock-benchmark plan capacity-mix-rate --model nova-micro --mix <workflow-name>
bedrock-benchmark run capacity-mix-rate --model nova-micro --mix <workflow-name>
```

The shipped 60/30/10 mix is a reference example only. Every class must meet its
own SLO before the total rate qualifies as `R_safe(mix)`; recommendation headroom
then produces `R_admission(mix)`. Isolated class capacities cannot be added.

`mix.assignment: stratified` uses shuffled blocks with the configured class
counts (6/3/1 for the reference example); a window may end with a partial block.
Use `stochastic` for independent weighted draws that model traffic randomness.
Stratified mixes require an exact block of at most 10,000 requests; unsupported
weights fail validation instead of silently rounding away rare classes.
Artifacts record `configured_mix` and `observed_mix`, with observed counts and
shares for each sweep point and measured confirmation candidate as well.

Confirmation caps are `auto`: request budget is 1.25 times the last look's
required total; duration is the number of windows needed at the candidate RPS,
including per-window warmup, times 1.25. Each candidate gets its own budget,
so low-RPM models can take hours. Numeric caps still impose explicit limits.

A mixed/global total in-flight limit would need its own experiment
(not built -- add one only if a consumer needs it). `max_inflight` and
`sustained_rps` are two independently confirmed **guardrails**, one per
dimension: the concurrency sweep controls C and lets the rate emerge;
the rate sweep controls R and lets concurrency emerge. Enforcing both
(C <= max_inflight and R <= sustained_rps) is conservative, but it is
not a statistically confirmed 2-D (C, R) capacity surface -- no joint
(C, R) point near the recommendation has been validated (planned:
`joint-capacity`).

**A single run is not a production config.** Every profile is a
`single_run_operating_envelope`; repeat across times and days, and take
`production_capacity_input` from `bedrock-benchmark validate`'s
temporal-capacity-profile -- the conservative value, set only once the
temporal evidence suffices (status `VALID` / `VALID_CONSERVATIVE`).

Every experiment is discovery followed by adaptive confirmation at the
candidate -- the only way a point becomes statistically confirmed.
Times are nova-micro `--dry-run` estimates.

## Install

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

```bash
source .venv/bin/activate
```

```bash
export AWS_PROFILE=<your-profile> AWS_REGION=us-east-1
```

That installs the `bedrock-benchmark` command (without activating the
venv: `.venv/bin/bedrock-benchmark`). You name models and experiments --
never file paths; `bedrock-benchmark list` shows both. It works from any
directory (it finds the checkout, or set `BEDROCK_BENCHMARK_HOME`).
`run ... --dry-run` is `plan`, `run ... --pilot` is `pilot`; `all` runs
every experiment; `--slo-profile gold` keeps only workloads bound to a
profile. Results go to `results/run-all-<timestamp>/<model>/`.
(`scripts/*.py` still work but are not the public interface.)

## Output example

Per workload class, measurement and recommendation are separate blocks.
Abridged, from the 2026-09-26 nova-micro `capacity-reference-rate` run (full schema:
[docs/capacity-profile-schema.md](docs/capacity-profile-schema.md)):

```yaml
workload_classes:
  short_chat:
    slo_profile: gold
    rate:                                       # MEASUREMENT
      observed_nonfailing_offered_rps: 8.3333
      observed_verdict: INCONCLUSIVE            # no violation seen, too few requests to prove it
      statistically_confirmed_offered_rps: 6.6667
      provider_ceiling_rps: 6.6667
      saturation: {observed_edge: 10.0, phase: discovery, status: discovery_resolved}   # observed, not confirmed
    confirmation:
      candidates: [{value: 6.6667, verdict: PASS, stop_reason: confirmed, n: 4173}]
    recommendation:                             # POLICY
      admission_envelope:
        max_inflight: null
        sustained_rps: 5.3334                   # min(6.6667 x 0.8, 6.6667 x 0.9)
        source: statistically_confirmed_measurement
        headroom_fraction: 0.2
        binding: measurement
```

No statistically confirmed point means `admission_envelope: null`: a
recommendation is never derived from an observed-only or INCONCLUSIVE
point, and `reason` says why -- each candidate's verdict and
`stop_reason` (e.g. `confirmation at 6.6667: FAIL (violation_demonstrated,
n=4380)`).

## Not in scope

Tenant quotas, fairness, queue policy, AIMD, adaptive global
concurrency, auth and gateway config -- all `eval-bedrock-gateway`'s
job. This repo outputs only a safe backend operating envelope (measured,
statistically confirmed, plus an admission-envelope recommendation); it
never implements, applies or pre-decides the runtime policy that
enforces it, and reads nothing from a gateway's tables or config.

## Documentation

| Doc | Covers |
|---|---|
| [methodology](docs/methodology.md) | core abstractions, SLO goodput, measurement window, input calibration and workload validation, mixed workloads, provenance and temporal validation |
| [SLO statistics](docs/slo-statistics.md) | SLO profiles, PASS / FAIL / INCONCLUSIVE, exact bounds for latency / success / throttle, adaptive confirmation, false-PASS and false-FAIL control |
| [quota model](docs/quota-model.md) | provider ceiling, TPM reservation vs consumption, quota-relative sweeps, `fetch_quota.py` |
| [capacity-profile schema](docs/capacity-profile-schema.md) | the deliverable and its consumer contract, measurement vs recommendation; the temporal-capacity-profile and the machine-readable JSON Schemas |
| [experiment design](docs/experiment-design.md) | models, workload catalog, experiments, constraints, running, pilot |
| [admission control](docs/admission-control.md) | minimum gateway admission config, two guardrails (not a 2-D region), production capacity input |
| [benchmark outputs](docs/benchmark-outputs.md) | every output, from admission values to evidence and temporal validation |
| [correctness history](docs/correctness-history.md) | every measurement bug found in review, by schema version |

## Testing

```bash
.venv/bin/python -m pytest -q
```

Python 3.11 or 3.12 (`requires-python` in `pyproject.toml`);
`.python-version` pins 3.11. All runner, metrics, statistics and
recommendation logic is tested against a fake Bedrock client
(`tests/fakes.py`) -- no network or AWS credentials needed.

`capacity-shape-concurrency` also runs `long_generation` (4096 input / 1024 output, bronze)
as an experiment-local `reference_control`. Its measurements help compare decode
pressure with large-context behavior; it emits no calibration point or production
admission recommendation in this experiment.

### Follow-up experiments for provider-state behavior

The main input/history study uses three enterprise classes with fixed output=64:

| Class | Input / output | Experiment A rates (RPS) |
| --- | --- | --- |
| `short` | 512 / 64 | 0.5, 1, 2, 3, 4 |
| `medium` | 2048 / 64 | 0.5, 1, 1.5, 2 |
| `long` | 8192 / 64 | 0.25, 0.5, 0.75, 1 |

All three use silver TPOT/reliability and an 8000ms E2E budget. TTFT follows
`constraints/slo.yaml` input bands (800 / 1500 / 3000ms). Thus capacity is
conditional on these SLOs; latency curves help distinguish input cost from
budget effects. Token validation checks actual output as well as input.

**Experiment A:** run `capacity-context-{short,medium,long}-rate` separately.
Discovery uses 900s windows; independent confirmation starts after 300s cooldown
and recovery checks, with at least 900s continuous measured exposure. Use only
`calibration_point.statistically_confirmed_rate` as `R_safe_idle`; discovery,
quota ceilings and production headroom are not baselines. This is an operational
idle/recovery baseline, not proof that provider state reset. If no rate confirms,
extend the sweep downward; if every rate passes, extend upward before treating
the highest tested rate as the capacity boundary.

**Experiment B:** `diagnostic-context-history.yaml` is a template. It is skipped
until A's three valid artifacts are supplied to the generator. The generator
writes three runnable configs, each at 50%, 75%, and 90% of its own baseline.
It rejects missing/unconfirmed baselines and mismatched models, quotas, shapes,
or SLOs, and records source hashes and baseline rates. Generated configs also
check those bindings at load time; regenerate when the environment changes.

Each of the nine cells has an idle arm and three independent overload arms.
Every arm starts with 300s idle. Overload is still 2x the class's nominal quota
ceiling for 120s; drain completes before the fixed 120/300/600s recovery wait.
No recovery probes are sent in B. Each observation is a continuous 900s window
with 30s bins. Two repetitions reverse arm order and pair arrival seeds.
The matrix has 72 observations and takes about **30.9 hours**, plus calibration
and drain (10.3 hours per generated class config; 3.4 hours per Actions cell).
Record other traffic sharing the quota. The artifact records whether offered
overload actually caused throttling. Target pressure is normalized to baseline;
overload pressure retains the existing quota-relative protocol.

The first round compares recovery behavior. Its descriptive bins and three load
levels do not independently confirm `R_safe_after_overload`; establishing that
function requires boundary sweeps and confirmation for each recovery condition.
`diagnostic-context-stress1` runs `short`, `medium`, and `long` at fixed absolute
rates of 0.1, 0.25, 0.5, 1.0, and 1.6667 RPS, with output=64 and the same
history protocol. It runs without baseline artifacts and does not normalize
target pressure. Actions splits it into 15 serial jobs, about 3.4 hours each
and 51.5 hours total before calibration/drain. The 16K/256 shape remains in
the workload catalog but is not selected by this experiment.
Existing results and partially completed runs are left intact.

#### Round two: idle capacity, then normalized history

`diagnostic-context-stress1` is the renamed first-round fixed-rate history
experiment; its protocol and rates are unchanged. Existing artifacts retain
their original experiment names.

Run **`diagnostic-context-stress2`** in Actions to discover and independently
confirm each class's idle baseline. It uses 900s discovery windows and independent
confirmation with a 300s cooldown, recovery checks and at least 900s continuous
measured exposure. It produces calibration evidence without a production
admission envelope. Actions runs three serial jobs, keeping each class's entire
sweep and confirmation together:

| Class | Offered rates (RPS) |
| --- | --- |
| short, 512/64 | 2, 3, 4, 5, 6 |
| medium, 2048/64 | 2, 3, 4, 5 |
| long, 8192/64 | 2, 2.5, 3, 4 |

These grids suit the current nova-micro/nova-lite/qwen quota ranges. With the
current nova-pro and llama quotas, all proposed rates exceed the nominal rate
ceiling and the runner cannot confirm a baseline; lower the grids for those
models before using this workflow for capacity calibration.

The grid searches for a knee; it does not guarantee one is found. If the highest
rate passes, extend that class's grid. If the first point fails, add lower rates.
Add intermediate points to narrow a pass/fail bracket before using a baseline
as a near-boundary capacity estimate. Example capacities such as 4.5/3.2/2.3 are
not baked into the configuration.

After reviewing the three confirmed baselines, download their capacity profiles
and generate the second-stage history configs:

```sh
.venv/bin/python scripts/prepare_context_history.py --model nova-micro \
  --template experiments/diagnostic-context-stress2-history.yaml \
  --profiles path/to/short-profile.yaml path/to/medium-profile.yaml path/to/long-profile.yaml \
  --output-dir experiments
```

Commit and push the generated configs, then select each in Actions:
`diagnostic-context-stress2-history-short`, `diagnostic-context-stress2-history-medium`,
and `diagnostic-context-stress2-history-long`. Each uses **70% / 85% / 95%** of
its own confirmed baseline and recovery waits of **120 / 300 / 600s**. The
unchanged overload is 2x the class's nominal quota ceiling for 120s. Nine history
cells take about 30.9 hours before calibration/drain. The unbound history
template sends no traffic. A highest-tested passing baseline is still a lower
bound if saturation was not reached; normalized fractions are relative to that
baseline, not a proven maximum.

Characterization summaries say `characterization complete; no admission envelope
produced by diagnostic experiment`. Incomplete or invalid history observations
are identified separately. The absence of an admission envelope does not imply
that every offered rate failed its SLO. History observations remain descriptive;
confirming `R_safe(class, provider state)` needs independent confirmation under
each state as well as repeated measurements.

History runs save locally after each arm finishes, without waiting for the full
matrix. Under `results/<model>/<experiment>-<run_id>-checkpoints/`,
`manifest.yaml` reports saved/observed arm counts, the planned total, configuration,
and run status. Each `arm-NNNN.yaml` contains the descriptive summary and bins;
its matching JSONL contains that arm's measurement, overload, and recovery-probe
requests (initial calibration remains in the final full-run output).
Files are replaced atomically; the manifest lists an arm only after both files
are written. Ctrl+C or an error retains previously saved arms. The current
unfinished arm is not checkpointed, and automatic resume is not supported.
A forcibly killed process may leave status `running`; listed files remain usable.
Normal completion also writes the usual combined JSONL and capacity-profile YAML.
No files are uploaded. This does not shorten the configured measurement windows.

Each arm records its requested delay, actual interval since the overload window
ended, and interval since overload drain completed. `recovery_summary` reports
the first throttled request's arrival offset (null when none), successful RPS
in the first 120s (or the whole window if shorter), successful RPS in the final
third, and aggregate throttle rate. These summaries are descriptive and do not
establish SLO compliance. Legacy configurations default to `verified` recovery
and retain their probe behavior and single `recovery_s` delay.

Bins report offered/scheduled/attempted/successful/throttled RPS, request-cohort
success/throttle rates, TTFT, latency, peak inflight, and scheduling lag. Rate
counts use scheduled arrivals, actual starts, or completions as named; the
success/throttle proportions follow requests scheduled in the bin, including
responses that finish later. The final bin may therefore show fewer completions
than eventual successes. Raw JSONL keeps measurement, overload and probe phases.

`capacity-shape-concurrency` supports a focused retest using `--workload`,
`--candidate-concurrency`, and `--steady-state-duration-s` together. For example, retest C=7 using a continuous 30-minute discovery window
and fresh confirmation with at least 30 minutes of measured exposure. Confirmation
may reject the candidate or require additional windows. Its measured RPS is not a
validated rate envelope. Compare repeated runs before deriving an admission policy.

```sh
# Run these sequentially, against the same quiet model quota.
.venv/bin/bedrock-benchmark run capacity-context-short-rate --model nova-micro
.venv/bin/bedrock-benchmark run capacity-context-medium-rate --model nova-micro
.venv/bin/bedrock-benchmark run capacity-context-long-rate --model nova-micro

# Supply the three actual capacity-profile.yaml paths from A.
.venv/bin/python scripts/prepare_context_history.py --model nova-micro \
  --profiles <short-profile.yaml> <medium-profile.yaml> <long-profile.yaml> \
  --output-dir results/context-history-configs
.venv/bin/python scripts/run.py results/context-history-configs/diagnostic-context-history-short.yaml --model nova-micro
# Then run the medium and long generated configs sequentially.
.venv/bin/bedrock-benchmark run capacity-shape-concurrency --model nova-micro \
  --workload medium_context --candidate-concurrency 7 --steady-state-duration-s 1800
```

The full history matrix takes about 30.9 hours before calibration/drain; the
sustain retest typically needs at least an hour if its candidate remains eligible.
Run them separately to avoid contaminating their provider state with each other.
Reports retain `nominal_binding_constraint`, but throttling is described as an
observed symptom. Suspect measurements use `bottleneck: unresolved`; public quota
ratios alone do not establish the actual cause of provider rejection.

Focused retests keep the base experiment name and set `mode: sustain`; regular
capacity sweeps use `mode: sweep`, and history diagnostics use `mode: history`.
Retest parameters are recorded under `measurement.retest`. Temporal comparison
groups by experiment, model, workload/mix, mode, candidate concurrency and
measurement duration (including minimum confirmation exposure). Historical
experiment names remain distinct; result files are never renamed or rewritten.
The retest options work for plan, pilot and run. Without them the original shape
sweep applies.

### Experiment names

| Previous name | Current name |
|---|---|
| `concurrency-sweep` | `capacity-reference-concurrency` |
| `rate-capacity` | `capacity-reference-rate` |
| `mixed-capacity` | `capacity-mix-rate` |
| `workload-shape-calibration` | `capacity-shape-concurrency` |
| `long-context-history` | `diagnostic-context-history` |

Use the current names in commands. `purpose` defines how results can be used;
workload roles such as `reference_control` remain explicit configuration fields.

### Streaming measurements

All five experiments use a shared descriptive metrics structure with latency
p50/p95/p99 and sample counts, request/token throughput, reliability and load state.
Capacity reports expose each repetition under `measurement_windows`, with 30-second
bins and per-class breakdowns for mixed traffic. History reports use
`history_comparison[].aggregate.metrics` and `bins[].metrics`.

Raw JSONL also preserves submission and last-text timing, including partial-stream
failures. Existing TPOT SLO checks keep their definition; the additional
`text_decode_tpot_ms` excludes trailing metadata time and is descriptive only.
See [metric definitions](docs/capacity-profile-schema.md#common-descriptive-metrics-metrics_version-1).

### TTFT by input length

In `constraints/slo.yaml`, `ttft_budgets` assigns the first-response budget by
configured input length: <=512 tokens → 800ms; 513–4096 → 1500ms; >4096 → 3000ms.
These are initial policy budgets. The workload's gold/silver/bronze profile
independently selects TPOT and reliability; E2E stays in the workload catalog.
Reports record the resolved limits in `constraints.slo.effective_by_workload`.

### Sustainable capacity and overload experiments

- `capacity-reference-rate`: Poisson discovery at 0.25 / 0.5 / 0.75 / 1× nominal quota ceiling. Stops at a clear statistical FAIL or severe-throttle guard. An optional 1.25× boundary point can be added explicitly.
- Reference rate and concurrency use liveness **and** a baseline-capacity recovery check before discovery and confirmation. The baseline uses the same workload at 0.25× ceiling for at least 120 seconds and 100 requests, with goodput, throttle and latency checks. It is a recovery control, not proof that a higher candidate is safe. Recovery exhaustion invalidates the subject.
- Fresh reference confirmation uses windows of at least 300 seconds of continuous measured load. A fixed-count statistical PASS and minimum exposure are both required; additional observations can veto PASS. Rate confirmation tries up to three preselected candidates, highest first, with confidence split across candidates and looks.
- `capacity-burst-rate` is opt-in and excluded from `run all`. Each 90-second pulse at 1.25 / 1.5 / 2 / 2.5× ceiling starts after checked baseline recovery. Results record first observed throttling and recovery checks. Missing onset and exhausted recovery are censored observations, not zero durations. Recovery time includes cooldown and probe durations; it is not an exact provider reset timestamp.
- `run` advances a local rotation counter in `results/.workload-rotation-index`. All experiments in that batch share the index. Use `--workload-rotation-index 0` (or 1, 2, …) to replay an order. The actual order and index are recorded in the profile. Rotation does not replace recovery checks.

Single-run results describe the measured conditions. Repeat across times/days for temporal validation. Separately measured concurrency and rate limits still require validation when applied together or to mixed traffic.

## Conditional concurrency operating envelope

The reference concurrency experiment estimates the isolated-workload concurrency
operating envelope under healthy observed provider conditions. Discovery and
refinement only select candidates. Each confirmation candidate requires its own
cooldown, liveness probe and baseline-capacity probe before fresh measurement.
A failed recovery blocks that candidate and later candidates. Probes are recorded
but excluded from capacity measurements; a healthy probe does not prove the
provider has reset or that prior overload has no effect.

Each confirmation candidate now records `provider_state`: its recovery checks,
check time and observed status. `healthy_observed` requires both probe types;
`unverified` means that full baseline checking was not configured.
`anomaly_observed` records a restart after suspicious throttling; `unrecovered`
means recovery failed. A candidate can still have a statistical FAIL while its
capacity interpretation remains unresolved. Throttling below the nominal ceiling
is reported as an explicit `anomaly` with unresolved cause, not proof of a
specific provider mechanism or a universally unsafe candidate.

Per-workload `operating_conditions` binds the result to its observed provider
state, measurement status, quota, time/environment and effective SLO. Existing
`observed_nonfailing`, `statistically_confirmed`, and discovery `saturation`
remain separate. Existing admission envelope fields retain their meaning.
Single-run results require repeated measurements at different times/days and
`bedrock-benchmark validate` before use as production capacity inputs.

### Run experiments in GitHub Actions

Use **Actions → Bedrock benchmark → Run workflow**. Set `experiment` to a name
from `experiments/`, for example `capacity-context-short-rate` (the default),
`capacity-reference-concurrency`, or another experiment. The `.yaml` suffix is
optional. Select the model using `model`. `dry_run` defaults to true and only
checks plans; disable it for live traffic.

The workflow reads the selected YAML. History experiments with explicit rate
values split into serial workload/rate jobs, retaining all trials, recovery
delays and paired seeds. Copy generated class configs into `experiments/` to
select them in Actions: each produces three serial jobs, about 3.4 hours each.
The unbound main template is rejected with a generation instruction. All three
classes total nine jobs and 30.9 hours, plus calibration/drain and setup.
Other experiments run as one job, preserving their full sweep, adaptive
refinement, confirmation and inter-workload isolation. Matrix job ordering is
not guaranteed. Avoid other traffic sharing the quota.

Every job has a 270-minute measurement limit. Check the dry-run duration before
starting a different experiment; longer experiments need a different execution
setup or a protocol-specific split. Incremental arm checkpoints are available
for history mode; other modes may not retain partial measurements on timeout.

Live runs also default to `commit_results: true`. After each job uploads its
artifact, it commits all saved files in `results/` to the **benchmark-results**
branch under `results/actions/<run-id>/<attempt>/<job-id>/`. The branch starts
from the first publishing run's source commit; later jobs append result commits.
The benchmark job requests `contents: write` for its `GITHUB_TOKEN`; branch rules
must permit it to create/update `benchmark-results`. The source branch stays
unchanged. Uncheck `commit_results` for artifact-only storage; dry runs never
commit results. Failed measurements also publish saved files when the final
steps can execute. A hard runner termination may prevent both upload and commit.
Existing runs use their original workflow; start a new run to use this behavior.

Configure repository **Settings → Secrets and variables → Actions → Secrets**:
add `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` for a dedicated IAM user
with the required Bedrock permissions. Store the keys only in Actions secrets.
This workflow uses the keys directly; it does not require `AWS_ROLE_ARN` or OIDC.
The workflow authenticates separately for each job and stops measurement after
270 minutes to leave time for cleanup and artifact upload.

Each finished job uploads its results and local checkpoints as a separate
Actions artifact, retained for 14 days. Earlier jobs' artifacts can be downloaded
while later jobs run. Failed jobs also attempt to upload completed checkpoints;
forced runner termination can prevent the final upload. Checkpoints from a
running job become downloadable when its upload step runs. These subsets are
independent runs with separate run IDs, not repeated measurements of the entire
matrix. A GitHub runner also changes the client/network environment compared
with a local run; keep that context when comparing latency.
