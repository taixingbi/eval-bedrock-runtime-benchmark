import asyncio
from dataclasses import replace

import pytest

from bedrock_benchmark.analysis.metrics import MeasurementWindow
from bedrock_benchmark.ceiling import ProviderCeiling
from bedrock_benchmark.client import BedrockConverseTarget
from bedrock_benchmark.experiments import executor
from bedrock_benchmark.experiments.schema import ConfirmationConfig, RecoveryProbe, SweepConfig
from bedrock_benchmark.report import build_capacity_profile
from bedrock_benchmark.results import RequestResult
from .fakes import FakeBedrockRuntimeClient
from .test_executor import _spec


def scripted(monkeypatch, counts):
    counts = iter(counts)
    calls = []

    class Runner:
        def __init__(self, target, subject, *, duration_s, warmup_s, **kwargs):
            self.subject = subject
            self.duration = duration_s
            self.window = MeasurementWindow(1000 + len(calls) * 1000, 1000 + len(calls) * 1000 + duration_s)
            calls.append(self)

        async def run(self):
            clean, bad = next(counts)
            return [RequestResult(request_id=f'{len(calls)}-{i}',
                scheduled_at=self.window.start + (i + .25) / (clean + bad) * self.duration,
                started_at=self.window.start + (i + .25) / (clean + bad) * self.duration,
                completed_at=self.window.start + (i + .25) / (clean + bad) * self.duration + .001,
                latency_ms=1, ttft_ms=.5, success=i < clean, throttled=i >= clean,
                input_tokens=100, output_tokens=16, tags={'workload': self.subject.name})
                for i in range(clean + bad)]

    monkeypatch.setattr(executor, 'RateRunner', Runner)
    monkeypatch.setattr(executor, 'ConcurrencyRunner', Runner)
    return calls


def run(spec):
    with_target = BedrockConverseTarget(model_id='m', client=FakeBedrockRuntimeClient())
    return asyncio.run(executor.run_experiment(spec, target=with_target))


def test_clear_fail_stops_before_higher_load(monkeypatch):
    calls = scripted(monkeypatch, [(100, 0), (0, 100)])
    report = run(_spec(sweep=SweepConfig(type='rate', values=[1, 2, 3], stop_after_clear_fail=True), repetitions=1))
    assert len(calls) == 2
    assert [p.rps for p in report.profiles[0].points] == [1, 2]
    assert report.profiles[0].measurement_validity['events'][-1]['outcome'] == 'stopped_clear_fail'


def test_liveness_cannot_override_failed_baseline(monkeypatch):
    calls = scripted(monkeypatch, [(100, 0), (10, 90)])
    spec = _spec(repetitions=1, recovery_probe=RecoveryProbe(
        baseline_fraction=.25, baseline_duration_s=10, baseline_min_requests=10,
        max_attempts=1, retry_cooldown_s=0))
    spec.provider_ceilings = {'short': ProviderCeiling(tokens_per_request=116, rpm_rps=40, tpm_rps=None)}
    report = run(spec)
    assert len(calls) == 2
    assert report.profiles[0].points == []
    validity = report.profiles[0].measurement_validity
    assert validity['status'] == 'invalid'
    assert validity['recovery_probes'][0]['healthy']
    assert not validity['recovery_probes'][1]['healthy']
    assert build_capacity_profile(report)['workload_classes']['short']['recommendation']['admission_envelope'] is None


def test_continuous_confirmation_uses_single_long_window(monkeypatch):
    calls = scripted(monkeypatch, [(100, 0), (2000, 0)])
    report = run(_spec(repetitions=1, duration_s=90, warmup_s=0,
        sweep=SweepConfig(type='rate', values=[5]),
        confirmation=ConfirmationConfig(continuous=True, min_steady_state_duration_s=300,
                                        max_duration_s='auto', max_requests='auto'),
        slo=replace(_spec().slo, throttle_rate_max=.01, success_rate_min=.99)))
    assert [c.duration for c in calls] == [90, 300]
    candidate = report.profiles[0].confirmations[0]
    assert candidate.verdict == 'PASS'
    assert candidate.repetitions == 1
    assert candidate.steady_state['measured_duration_s'] == 300


def test_rotation_is_reproducible_and_reported(monkeypatch):
    scripted(monkeypatch, [(100, 0)] * 3)
    spec = _spec(repetitions=1, workload_rotation_index=1)
    spec.workloads = [replace(spec.workloads[0], name=n) for n in ['short', 'rag', 'long']]
    report = run(spec)
    assert [p.workload_name for p in report.profiles] == ['rag', 'long', 'short']
    assert build_capacity_profile(report)['measurement']['workload_order'] == ['rag', 'long', 'short']


def test_invalidated_window_is_not_reported_as_empty(monkeypatch):
    scripted(monkeypatch, [(100, 0)])
    report = run(_spec(repetitions=1))
    for r in report.all_results:
        r.tags['phase'] = 'invalidated'
    report.measurement_windows[0]['phase'] = 'invalidated'
    assert build_capacity_profile(report)['workload_classes']['short']['measurement_windows'] == []


def test_burst_records_onset_and_recovery_without_capacity(monkeypatch):
    scripted(monkeypatch, [(100, 0), (100, 0), (50, 50), (100, 0), (100, 0)])
    spec = _spec(purpose='characterization', burst_protocol=True, duration_s=10, warmup_s=0,
        repetitions=1, sweep=SweepConfig(type='rate', values=[50]),
        recovery_probe=RecoveryProbe(baseline_fraction=.25, baseline_duration_s=10,
                                    baseline_min_requests=10, max_attempts=1, retry_cooldown_s=0))
    spec.provider_ceilings = {'short': ProviderCeiling(tokens_per_request=116, rpm_rps=40, tpm_rps=None)}
    report = run(spec)
    burst = report.profiles[0].measurement_validity['burst_results'][0]
    assert burst['throttle_onset_s'] == pytest.approx(5.025)
    assert burst['recovered']
    assert report.profiles[0].confirmations == []


def test_retry_retags_window_metadata_with_rows(monkeypatch):
    scripted(monkeypatch, [(100, 0), (0, 100), (100, 0), (100, 0)])
    spec = _spec(repetitions=1, recovery_probe=RecoveryProbe(max_attempts=1, retry_cooldown_s=0))
    spec.provider_ceilings = {'short': ProviderCeiling(tokens_per_request=116, rpm_rps=1000, tpm_rps=None)}
    report = run(spec)
    assert [w['phase'] for w in report.measurement_windows] == ['invalidated', 'discovery']
    windows = build_capacity_profile(report)['workload_classes']['short']['measurement_windows']
    assert len(windows) == 1
    assert windows[0]['metrics']['reliability']['n'] == 100


def test_clear_fail_does_not_repeat_overload_before_confirmation(monkeypatch):
    calls = scripted(monkeypatch, [(100, 0), (0, 100)])
    spec = _spec(repetitions=1, sweep=SweepConfig(type='rate', values=[40, 80], stop_after_clear_fail=True),
                 recovery_probe=RecoveryProbe(max_attempts=1, retry_cooldown_s=0))
    spec.provider_ceilings = {'short': ProviderCeiling(tokens_per_request=116, rpm_rps=1000, tpm_rps=None)}
    report = run(spec)
    assert len(calls) == 2
    assert len(report.profiles[0].points) == 1


def test_rotation_counter_and_explicit_replay(tmp_path):
    from bedrock_benchmark.experiments.order import rotation_index
    assert [rotation_index(tmp_path) for _ in range(3)] == [0, 1, 2]
    assert rotation_index(tmp_path, 1) == 1
    assert rotation_index(tmp_path) == 3


def test_burst_profile_is_characterization_only(monkeypatch):
    scripted(monkeypatch, [(100, 0), (100, 0), (100, 0), (100, 0), (100, 0)])
    spec = _spec(purpose='characterization', burst_protocol=True, duration_s=10, warmup_s=0,
        repetitions=1, sweep=SweepConfig(type='rate', values=[50]),
        recovery_probe=RecoveryProbe(baseline_fraction=.25, baseline_duration_s=10,
                                    baseline_min_requests=10, max_attempts=1, retry_cooldown_s=0))
    spec.provider_ceilings = {'short': ProviderCeiling(tokens_per_request=116, rpm_rps=40, tpm_rps=None)}
    entry = build_capacity_profile(run(spec))['workload_classes']['short']
    assert entry['burst'][0]['throttle_onset_censored']
    assert 'rate' not in entry
    assert entry['recommendation']['admission_envelope'] is None


def test_recovery_rejects_lost_goodput_even_without_throttles(monkeypatch):
    calls = scripted(monkeypatch, [(100, 0), (100, 0), (100, 0), (100, 0), (80, 0)])
    spec = _spec(repetitions=1, duration_s=10, warmup_s=0,
        confirmation=ConfirmationConfig(),
        recovery_probe=RecoveryProbe(baseline_fraction=.25, baseline_duration_s=10,
                                    baseline_min_requests=10, max_attempts=1, retry_cooldown_s=0))
    spec.provider_ceilings = {'short': ProviderCeiling(tokens_per_request=116, rpm_rps=40, tpm_rps=None)}
    report = run(spec)
    assert len(calls) == 5
    assert report.profiles[0].measurement_validity['status'] == 'invalid'
    assert report.profiles[0].confirmations[0].stop_reason == 'provider_state_invalid'
    assert report.profiles[0].recommendation.confirmed_point is None


def test_invalid_shape_blocks_confirmed_recommendation(monkeypatch):
    scripted(monkeypatch, [(4000, 0)])
    spec = _spec(repetitions=1, sweep=SweepConfig(type="concurrency", values=[2]))
    spec.workloads = [replace(spec.workloads[0], input_tokens=512)]
    report = run(spec)
    entry = build_capacity_profile(report)["workload_classes"]["short"]
    assert entry["workload_validation"]["valid"] is False
    assert entry["concurrency"]["statistically_confirmed"] == 2
    assert entry["recommendation"]["admission_envelope"] is None
    assert entry["recommendation"]["reason"] == "workload_validation_failed"


def test_sdk_concurrency_violation_invalidates_subject(monkeypatch):
    calls = scripted(monkeypatch, [(1000, 0)])
    original = executor.ConcurrencyRunner.run

    async def violate(self):
        rows = await original(self)
        target.peak_sdk_inflight = 3
        return rows

    monkeypatch.setattr(executor.ConcurrencyRunner, "run", violate)
    target = BedrockConverseTarget(model_id="m", client=FakeBedrockRuntimeClient())
    report = asyncio.run(executor.run_experiment(
        _spec(repetitions=1, sweep=SweepConfig(type="concurrency", values=[2, 4])), target=target))
    assert len(calls) == 1
    entry = build_capacity_profile(report)["workload_classes"]["short"]
    assert entry["measurement_validity"]["status"] == "invalid"
    assert entry["measurement_validity"]["events"][0]["peak_sdk_inflight"] == 3
    assert entry["recommendation"]["admission_envelope"] is None


def test_temporal_excludes_historical_invalid_shape(tmp_path):
    import yaml
    from bedrock_benchmark.drift import summarize
    from .test_drift import _profile
    profile = _profile("2026-09-29T00:00:00+00:00", 10, 8)
    profile["workload_classes"]["short_chat"]["workload_validation"] = {"valid": False}
    (tmp_path / "capacity-reference-rate-x-capacity-profile.yaml").write_text(yaml.safe_dump(profile))
    validation = summarize([tmp_path])[0]["temporal_validation"]
    assert validation["runs"] == 0
    assert validation["invalid_runs"] == 1


@pytest.mark.parametrize("stream", [False, True])
def test_sdk_counter_tracks_overlapping_calls_and_resets(stream):
    import threading
    from bedrock_benchmark.client import InvokeRequest
    barrier = threading.Barrier(2, timeout=5)

    class Client(FakeBedrockRuntimeClient):
        def converse(self, **kwargs):
            barrier.wait()
            return super().converse(**kwargs)

        def converse_stream(self, **kwargs):
            barrier.wait()
            return super().converse_stream(**kwargs)

    target = BedrockConverseTarget(model_id="m", client=Client())

    async def invoke_pair():
        return await asyncio.gather(*[
            target.invoke(InvokeRequest(prompt="hello", max_tokens=16, stream=stream))
            for _ in range(2)])

    try:
        assert all(r.success for r in asyncio.run(invoke_pair()))
        assert target.peak_sdk_inflight == 2
        target.reset_peak()
        assert target.peak_sdk_inflight == 0
    finally:
        target.close()


def test_invalid_measurement_blocks_otherwise_confirmed_capacity(monkeypatch):
    scripted(monkeypatch, [(4000, 0)])
    report = run(_spec(repetitions=1, sweep=SweepConfig(type="concurrency", values=[2])))
    report.profiles[0].measurement_validity = {"status": "invalid", "events": []}
    entry = build_capacity_profile(report)["workload_classes"]["short"]
    assert entry["workload_validation"]["valid"] is True
    assert entry["concurrency"]["statistically_confirmed"] == 2
    assert entry["recommendation"]["admission_envelope"] is None


@pytest.mark.parametrize('recover_next', [True, False])
def test_each_candidate_requires_its_own_healthy_baseline(monkeypatch, recover_next):
    # Initial probes, two discovery points, candidate 2 probes + failed
    # confirmation, then candidate 1 probes (and confirmation only if healthy).
    counts = [(100, 0), (100, 0), (200, 0), (200, 0),
              (100, 0), (100, 0), (100, 900),
              (100, 0), (100, 0) if recover_next else (10, 90)]
    if recover_next:
        counts.append((1000, 0))
    calls = scripted(monkeypatch, counts)
    spec = _spec(repetitions=1, duration_s=10, warmup_s=0,
        sweep=SweepConfig(type='concurrency', values=[1, 2]),
        recovery_probe=RecoveryProbe(baseline_fraction=.25, baseline_duration_s=10,
            baseline_min_requests=10, max_attempts=1, retry_cooldown_s=0),
        confirmation=ConfirmationConfig(candidates=2, cooldown_s=0, continuous=True,
            min_steady_state_duration_s=300, max_requests='auto', max_duration_s='auto'),
        slo=replace(_spec().slo, throttle_rate_max=.01, success_rate_min=.99))
    spec.provider_ceilings = {'short': ProviderCeiling(tokens_per_request=116, rpm_rps=40, tpm_rps=None)}
    # This test isolates candidate transitions, independently of anomaly retries.
    monkeypatch.setattr(executor, 'suspect_point', lambda *args: False)
    report = run(spec)
    first, second = report.profiles[0].confirmations
    assert first.verdict == 'FAIL'
    assert first.provider_state['status'] == 'healthy_observed'
    assert second.verdict == ('PASS' if recover_next else 'INCONCLUSIVE')
    assert second.provider_state['status'] == ('healthy_observed' if recover_next else 'unrecovered')
    assert len(calls) == (10 if recover_next else 9)
    assert all(p['reason'] == 'before candidate 1' for p in second.provider_state['recovery_checks'])
    assert [p['kind'] for p in second.provider_state['recovery_checks']] == ['liveness', 'baseline_capacity']
    artifact = build_capacity_profile(report)['workload_classes']['short']
    assert artifact['operating_conditions']['temporal_validation_required']
    anomaly = artifact['confirmation']['candidates'][0]
    assert anomaly['verdict'] == 'FAIL'
    assert anomaly['anomaly']['cause'] == 'unresolved'
    assert 'safety unresolved' in anomaly['capacity_interpretation']
    assert artifact['confirmation']['candidates'][1]['provider_state'] == second.provider_state
    if recover_next:
        assert second.n == 1000  # discovery and recovery data are excluded
        assert artifact['operating_conditions']['provider_state']['status'] == 'healthy_observed'


def test_ascending_confirmation_stops_on_non_pass(monkeypatch):
    calls = scripted(monkeypatch, [(2000, 0)] * 3 + [(2000, 0), (0, 2000)])
    report = run(_spec(repetitions=1, duration_s=90, warmup_s=0,
        sweep=SweepConfig(type='rate', values=[5, 6, 7]),
        confirmation=ConfirmationConfig(order='ascending', stop_after_fail=True,
            values_by_workload={'short': [5, 6, 7]}, continuous=True,
            max_requests='auto', max_duration_s='auto'),
        slo=replace(_spec().slo, throttle_rate_max=.01, success_rate_min=.99)))
    results = report.profiles[0].confirmations
    assert [(r.value, r.verdict) for r in results] == [(5, 'PASS'), (6, 'FAIL'), (7, 'INCONCLUSIVE')]
    assert results[-1].stop_reason == 'not_tested'
    assert report.profiles[0].confirmation_plan.to_dict()['order'] == 'ascending'
    assert report.profiles[0].recommendation.confirmed_point.rps == 5
    assert len(calls) == 5


def test_recovery_requires_consecutive_healthy_probes(monkeypatch):
    calls = scripted(monkeypatch, [(100, 0), (0, 100), (100, 0), (100, 0), (100, 0)])
    report = run(_spec(repetitions=1, sweep=SweepConfig(type='rate', values=[1]),
        recovery_probe=RecoveryProbe(max_attempts=4, retry_cooldown_s=0,
                                    required_consecutive_healthy=2)))
    assert len(calls) == 5
    assert len(report.profiles[0].points) == 1
    assert len(report.profiles[0].measurement_validity['recovery_probes']) == 4
