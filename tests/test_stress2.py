from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from bedrock_benchmark.context_history import prepare
from bedrock_benchmark.experiments.schema import load_experiment, NoMatchingWorkloads
from bedrock_benchmark.experiments.executor import ExperimentReport, ProfileReport
from bedrock_benchmark.run_file import estimated_duration_s, recommendation_summary, describe_sweep
from bedrock_benchmark.report import build_capacity_profile
from .test_actions_matrix import build_matrix
from .test_context_history import profiles
from .test_history_protocol import MODEL

GRIDS = {'short': [2, 3, 4, 5, 6, 6.25, 6.5, 6.67],
         'medium': [2, 3, 4, 5, 5.5, 6, 6.5, 6.67],
         'long': [2, 2.5, 3, 4, 4.5, 5, 5.5, 6, 6.5, 6.67]}
BASELINE = 'experiments/diagnostic-context-stress2.yaml'
HISTORY = 'experiments/diagnostic-context-stress2-history.yaml'


def test_stress2_preserves_per_class_sweeps_and_confirmation_in_actions(tmp_path):
    spec = load_experiment(BASELINE, MODEL)
    assert spec.purpose == 'admission_calibration'
    assert spec.history_protocol is None
    assert {n: spec.sweep_values(n) for n in spec.subject_names} == GRIDS
    assert f'short={GRIDS["short"]}' in describe_sweep(spec)
    jobs = build_matrix('diagnostic-context-stress2')['include']
    assert len(jobs) == 3
    durations = []
    for job in jobs:
        cfg = job['config']
        name = cfg['workloads'][0]
        assert cfg['sweep']['values'] == GRIDS[name]
        path = tmp_path / f'{name}.yaml'
        path.write_text(yaml.safe_dump(cfg))
        bound = load_experiment(str(path), MODEL)
        assert bound.confirmation == replace(
            spec.confirmation, values_by_workload={
                name: spec.confirmation.values_by_workload[name]})
        assert bound.recovery_probe == spec.recovery_probe
        durations.append(estimated_duration_s(bound))
        assert durations[-1] < 270 * 60
    assert estimated_duration_s(spec) == sum(durations) + 2 * spec.inter_subject_cooldown_s
    artifact = build_capacity_profile(ExperimentReport(spec=spec))
    assert artifact['sweep']['values_by_workload'] == GRIDS


@pytest.mark.parametrize('grid', [{}, {'short': [2]}, {**GRIDS, 'short': [2, 1]},
                                 {**GRIDS, 'long': [0, 2]}, {**GRIDS, 'long': [float('nan')]}])
def test_bad_per_class_rates_are_rejected(grid, tmp_path):
    cfg = yaml.safe_load(Path(BASELINE).read_text())
    cfg['sweep']['values_by_workload'] = grid
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='values_by_workload'):
        load_experiment(str(path), MODEL)


def test_stress2_history_uses_its_own_artifacts_and_fractions(profiles, tmp_path):
    with pytest.raises(NoMatchingWorkloads):
        load_experiment(HISTORY, MODEL)
    with pytest.raises(ValueError, match='Experiment A'):
        prepare(HISTORY, profiles, MODEL)
    for path in profiles:
        data = yaml.safe_load(path.read_text())
        data['experiment'] = 'diagnostic-context-stress2'
        path.write_text(yaml.safe_dump(data))
    configs = prepare(HISTORY, profiles, MODEL)
    assert len(configs) == 3
    for cfg in configs:
        name = cfg['workloads'][0]
        assert cfg['name'] == f'diagnostic-context-stress2-history-{name}'
        context = cfg['baseline_context']
        assert context['load_fractions'] == [.7, .85, .95]
        assert cfg['sweep']['values'] == [context['r_safe_idle_rps'] * f for f in [.7, .85, .95]]
        assert cfg['history_protocol']['recovery_delays_s'] == [120, 300, 600]
        path = tmp_path / f'{name}-history.yaml'
        path.write_text(yaml.safe_dump(cfg))
        load_experiment(str(path), MODEL)


def test_characterization_summary_distinguishes_completion_from_slo_failure():
    spec = load_experiment('experiments/diagnostic-context-stress1.yaml', MODEL)
    count = len(spec.sweep_values('short')) * spec.repetitions * 4
    profile = ProfileReport('short', history_comparison=[{'status': 'observed'}] * count)
    report = ExperimentReport(spec=spec, profiles=[profile])
    assert recommendation_summary(report) == [
        'short: characterization complete; no admission envelope produced by diagnostic experiment']
    profile.history_comparison = [{'status': 'baseline_unhealthy'}]
    assert 'incomplete or invalid' in recommendation_summary(report)[0]
    # Reference runs retain the existing capacity-failure message.
    report.spec = replace(spec, purpose='reference')
    assert 'NO swept value' in recommendation_summary(report)[0]


def test_grid_above_model_ceiling_has_no_confirmation_time():
    from bedrock_benchmark.run_file import _confirmation_estimate_s
    spec = load_experiment(BASELINE, replace(MODEL, quota_rpm=50))
    assert all(_confirmation_estimate_s(spec, n, 900) == 0 for n in spec.subject_names)
    import math
    assert math.isfinite(estimated_duration_s(spec))
