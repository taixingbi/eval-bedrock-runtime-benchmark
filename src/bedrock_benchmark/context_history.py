"""Materialize the normalized history matrix from confirmed baseline artifacts."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import math
from pathlib import Path

import yaml

from .experiments.schema import load_experiment
from .report import _slo_dict


def binding(spec):
    return {
        'model': {'name': spec.model_name, 'provider': 'bedrock',
                  'model_id': spec.target.model_id, 'region': spec.target.region},
        'quota': {'account': spec.quota_account, 'region': spec.target.region,
                  'rpm': spec.quota_snapshot.rpm, 'tpm': spec.quota_snapshot.tpm,
                  'output_burndown': spec.output_burndown},
        'workloads': {w.name: {'input_tokens': w.input_tokens, 'output_tokens': w.output_tokens,
                              'slo_profile': w.slo_profile, 'latency_p95_ms': w.latency_p95_ms,
                              'role': w.role} for w in spec.workloads},
        'slo': {w.name: {'ttft_budget': spec.ttft_budget_name(w.name), **_slo_dict(spec.slo_for(w.name))}
                for w in spec.workloads},
    }


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate_binding(spec):
    context = spec.baseline_context
    if context.get('binding') != binding(spec):
        raise ValueError('History baseline model, quota, workload or SLO differs; regenerate from Experiment A')
    if spec.mode != 'history' or len(spec.workloads) != 1:
        raise ValueError('Normalized history requires one workload per generated experiment')
    rate = context.get('r_safe_idle_rps')
    fractions = context.get('load_fractions')
    if not positive(rate) or not isinstance(fractions, list) or not fractions or any(
            not positive(f) or f > 1 for f in fractions):
        raise ValueError('History baseline needs a positive confirmed rate and fractions in (0, 1]')
    if spec.sweep_values(spec.workloads[0].name) != [rate * f for f in fractions]:
        raise ValueError('History rates differ from baseline x load fractions')


def prepare(template_path, profile_paths, model, root=Path('.')):
    template = yaml.safe_load(Path(template_path).read_text())
    if template.get('baseline_required') is not True:
        raise ValueError('Expected an Experiment B template requiring baseline results')
    fractions = template['sweep']['baseline_fractions']
    if not fractions or any(not positive(f) or f > 1 for f in fractions):
        raise ValueError('Baseline fractions must be in (0, 1]')
    sources = {}
    for path in profile_paths:
        data = Path(path).read_bytes()
        profile = yaml.safe_load(data)
        for name in template['workloads']:
            if name not in profile.get('workload_classes', {}):
                continue
            if name in sources:
                raise ValueError(f'Duplicate baseline for {name}; select one run explicitly')
            sources[name] = (profile, str(Path(path).resolve()), sha256(data).hexdigest())
    configs = []
    for name in template['workloads']:
        if name not in sources:
            raise ValueError(f'Missing confirmed baseline for {name}')
        profile, source, digest = sources[name]
        baseline_name = template.get('baseline_experiment', f'capacity-context-{name}-rate')
        if not isinstance(baseline_name, str) or Path(baseline_name).name != baseline_name:
            raise ValueError('baseline_experiment must be an experiment name')
        spec = load_experiment(str(root / 'experiments' / f'{baseline_name}.yaml'), model)
        spec = replace(spec, workloads=[w for w in spec.workloads if w.name == name])
        expected = binding(spec)
        constraints = profile.get('constraints', {})
        actual = {'model': profile.get('model'), 'quota': constraints.get('quota'),
                  'workloads': {name: constraints.get('workloads', {}).get(name)},
                  'slo': {name: constraints.get('slo', {}).get('effective_by_workload', {}).get(name)}}
        if actual != expected:
            raise ValueError(f'{name}: baseline model, quota, workload or SLO differs from current configuration')
        if (profile.get('purpose') != 'admission_calibration' or profile.get('mode') != 'sweep'
                or profile.get('experiment') != spec.name):
            raise ValueError(f'{name}: use the Experiment A rate calibration artifact')
        entry = profile['workload_classes'][name]
        point = entry.get('calibration_point') or {}
        rate = point.get('statistically_confirmed_rate')
        validation = entry.get('workload_validation', {})
        if (not positive(rate) or point.get('measurement_validity') == 'invalid'
                or any(validation.get(k, {}).get('valid') is not True for k in ('input', 'output'))
                or validation.get('valid') is not True):
            raise ValueError(f'{name}: no valid, independently confirmed rate baseline; rerun Experiment A')
        cfg = deepcopy(template)
        cfg.pop('baseline_required')
        cfg.pop('baseline_experiment', None)
        cfg['name'] = f'{template["name"]}-{name}'
        cfg['workloads'] = [name]
        cfg['sweep'] = {'type': 'rate', 'values': [rate * f for f in fractions]}
        cfg['baseline_context'] = {
            'binding': expected, 'r_safe_idle_rps': rate, 'load_fractions': fractions,
            'source_profile': source, 'source_sha256': digest,
            'source_measured_at': profile.get('environment', {}).get('measured_at'),
        }
        configs.append(cfg)
    return configs
