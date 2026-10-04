"""Build Actions jobs from an experiment, preserving adaptive capacity sweeps."""
import copy
import json
import os
from pathlib import Path
import re

import yaml


def build_matrix(experiment, root=Path('.')):
    name = experiment.removesuffix('.yaml')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ValueError('Use an experiment name or filename from experiments/')
    config = yaml.safe_load((root / 'experiments' / f'{name}.yaml').read_text())
    if config.get('baseline_required'):
        raise ValueError('Generate Experiment B with scripts/prepare_context_history.py first')
    sweep = config.get('sweep', {})
    jobs = []
    # Only explicit-rate history arms are independent. Keep other protocols whole.
    if config.get('history_protocol') and sweep.get('type') == 'rate' and sweep.get('values'):
        for workload in config['workloads']:
            for index, rate in enumerate(sweep['values']):
                subset = copy.deepcopy(config)
                subset['workloads'] = [workload]
                subset['sweep']['values'] = [rate]
                if subset.get('baseline_context'):
                    subset['baseline_context']['load_fractions'] = [
                        config['baseline_context']['load_fractions'][index]]
                if subset.get('seed') is not None:
                    subset['seed'] += index
                jobs.append({'label': f'{workload} / {rate} RPS', 'config': subset})
    elif sweep.get('values_by_workload') and sweep.get('type') == 'rate' and not config.get('history_protocol') and not config.get('mix'):
        # Keep each class's discovery, recovery and confirmation together.
        for workload in config['workloads']:
            subset = copy.deepcopy(config)
            subset['workloads'] = [workload]
            grids = subset['sweep'].pop('values_by_workload')
            subset['sweep']['values'] = grids[workload]
            jobs.append({'label': f'{workload} / idle capacity', 'config': subset})
    else:
        jobs.append({'label': name, 'config': config})
    if len(jobs) > 256:
        raise ValueError('Experiment exceeds the Actions matrix limit of 256 jobs')
    for index, job in enumerate(jobs):
        job['id'] = f'{name}-{index + 1}'
    return {'include': jobs}


if __name__ == '__main__':
    matrix = build_matrix(os.environ['EXPERIMENT'])
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        output.write('matrix=' + json.dumps(matrix, separators=(',', ':')) + '\n')
    for job in matrix['include']:
        print(job['label'])
