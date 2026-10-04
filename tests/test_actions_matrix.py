"""The Actions split must retain each experiment's measurement protocol."""
import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('actions_matrix', ROOT / 'scripts/actions_matrix.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_matrix = MODULE.build_matrix


def test_history_covers_all_pairs_and_preserves_protocol():
    original = yaml.safe_load((ROOT / 'experiments/diagnostic-context-stress1.yaml').read_text())
    jobs = build_matrix('diagnostic-context-stress1.yaml', ROOT)['include']
    assert len(jobs) == len(original['workloads']) * len(original['sweep']['values'])
    pairs = set()
    for job in jobs:
        config = job['config']
        rate = config['sweep']['values'][0]
        pairs.add((config['workloads'][0], rate))
        assert config['seed'] == original['seed'] + original['sweep']['values'].index(rate)
        for key in ('history_protocol', 'repetitions', 'duration_s', 'transport'):
            assert config[key] == original[key]
    assert pairs == {(w, r) for w in original['workloads'] for r in original['sweep']['values']}


def test_capacity_keeps_entire_adaptive_experiment():
    original = yaml.safe_load((ROOT / 'experiments/capacity-reference-concurrency.yaml').read_text())
    jobs = build_matrix('capacity-reference-concurrency', ROOT)['include']
    assert len(jobs) == 1
    assert jobs[0]['config'] == original


@pytest.mark.parametrize('name', ['../README', '/tmp/example', 'all', 'missing-experiment'])
def test_rejects_paths_and_unknown_experiments(name):
    with pytest.raises((ValueError, FileNotFoundError)):
        build_matrix(name, ROOT)
