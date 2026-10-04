import tempfile
import unittest
from pathlib import Path

from bedrock_benchmark.experiments.schema import NoMatchingWorkloads, load_experiment
from bedrock_benchmark.models import ModelConfig, load_models

MICRO = ModelConfig(name="nova-micro", model_id="us.amazon.nova-micro-v1:0", quota_rpm=400, quota_tpm=8_000_000)
PRO = ModelConfig(name="nova-pro", model_id="us.amazon.nova-pro-v1:0", quota_rpm=50, quota_tpm=2_000_000)
NO_QUOTA = ModelConfig(name="mystery", model_id="x.y-v1:0")

MINIMAL = (
    "name: minimal\n"
    "purpose: reference\n"
    "workloads: [short_chat]\n"
    "sweep: {type: concurrency, values: [1]}\n"
    "confirmation: {max_looks: 2}\n"
)


def _load_text(text: str, model: ModelConfig = MICRO):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        return load_experiment(path, model)
    finally:
        Path(path).unlink()


class ShippedExperimentTests(unittest.TestCase):
    def test_every_shipped_experiment_binds_to_every_shipped_model(self):
        for path in sorted(p for p in Path("experiments").glob("*.yaml") if p.stem not in {"diagnostic-context-history", "diagnostic-context-stress2-history"}):
            for model in load_models(include_disabled=True):
                with self.subTest(experiment=path.name, model=model.name):
                    spec = load_experiment(str(path), model)
                    self.assertEqual(spec.target.model_id, model.model_id)
                    self.assertEqual(spec.model_name, model.name)
                    self.assertEqual(spec.transport.total_max_attempts, 1)

    def test_no_shipped_experiment_stops_on_a_repetition_count(self):
        """Confirmation stops on requests / time / a look -- repetitions
        are only how data is collected."""
        for path in sorted(p for p in Path("experiments").glob("*.yaml") if p.stem not in {"diagnostic-context-history", "diagnostic-context-stress2-history"}):
            spec = load_experiment(str(path), MICRO)
            with self.subTest(path=path.name):
                if spec.history_protocol is not None or spec.burst_protocol:
                    self.assertEqual(spec.purpose, "characterization")
                    self.assertIsNone(spec.confirmation)
                else:
                    self.assertIsNotNone(spec.confirmation)
                    self.assertIsNone(spec.confirmation.max_repetitions)

    def test_experiment_files_never_name_a_model(self):
        model_words = {m.name for m in load_models(include_disabled=True)} | {"nova", "llama", "qwen"}
        for path in sorted(p for p in Path("experiments").glob("*.yaml") if p.stem not in {"diagnostic-context-history", "diagnostic-context-stress2-history"}):
            spec = load_experiment(str(path), MICRO)
            with self.subTest(path=path.name):
                self.assertFalse(any(w in spec.name for w in model_words), spec.name)
                self.assertFalse(any(w in path.stem for w in model_words), path.stem)

    def test_workload_shape_calibration_includes_an_experiment_local_reference_control(self):
        """The control reuses the reference shape without changing other experiments."""
        spec = load_experiment("experiments/capacity-shape-concurrency.yaml", MICRO)
        self.assertEqual([w.name for w in spec.workloads],
                         ["tiny_request", "medium_context", "long_context_short_answer", "very_large_context", "long_generation"])
        self.assertTrue(all(w.role == "characterization" for w in spec.workloads[:-1]))
        control = spec.workloads[-1]
        self.assertEqual((control.input_tokens, control.output_tokens, control.slo_profile,
                          control.latency_p95_ms, control.role), (4096, 1024, "bronze", 60000, "reference_control"))
        reference = load_experiment("experiments/capacity-reference-rate.yaml", MICRO)
        self.assertEqual(next(w.role for w in reference.workloads if w.name == "long_generation"), "reference")

    def test_reference_control_override_cannot_change_production_experiments(self):
        with self.assertRaisesRegex(ValueError, "non-reference experiment"):
            _load_text(MINIMAL + "workload_roles: {short_chat: reference_control}\n")
        with self.assertRaisesRegex(ValueError, "reference -> reference_control"):
            _load_text(MINIMAL.replace("purpose: reference", "purpose: characterization")
                       + "workload_roles: {short_chat: characterization}\n")
        with self.assertRaisesRegex(ValueError, "selected workload"):
            _load_text(MINIMAL + "workload_roles: {missing: reference_control}\n")

    def test_concurrency_sweep_covers_the_three_reference_workloads(self):
        spec = load_experiment("experiments/capacity-reference-concurrency.yaml", MICRO)
        self.assertEqual([(w.name, w.slo_profile) for w in spec.workloads],
                         [("short_chat", "gold"), ("rag_answer", "silver"), ("long_generation", "bronze")])
        self.assertEqual(spec.sweep.stop_after_fails, 1)
        # Sample-count driven: no repetition cap, only requests and time.
        c = spec.confirmation
        self.assertEqual((c.max_repetitions, c.max_requests, c.max_duration_s, c.candidates, c.cooldown_s),
                         (None, 8000, 3000, 2, 120))

    def test_mixed_capacity_defines_a_valid_mix(self):
        spec = load_experiment("experiments/capacity-mix-rate.yaml", MICRO)
        self.assertEqual(spec.mix.weights, {"short_chat": 0.6, "rag_answer": 0.3, "long_generation": 0.1})


class ModelBindingTests(unittest.TestCase):
    def test_target_and_quota_come_from_the_model(self):
        spec = load_experiment("experiments/capacity-reference-concurrency.yaml", PRO)
        self.assertEqual((spec.target.model_id, spec.target.region), ("us.amazon.nova-pro-v1:0", "us-east-1"))
        self.assertEqual((spec.quota_snapshot.rpm, spec.quota_snapshot.tpm), (50, 2_000_000))

    def test_quota_fractions_resolve_against_each_models_own_quota(self):
        micro = load_experiment("experiments/capacity-reference-rate.yaml", MICRO)
        pro = load_experiment("experiments/capacity-reference-rate.yaml", PRO)
        self.assertEqual(micro.sweep.quota_fractions, pro.sweep.quota_fractions)
        i = micro.sweep.quota_fractions.index(1.0)
        self.assertAlmostEqual(micro.sweep_values("short_chat")[i], 400 / 60, places=3)  # 1.0x ceiling (RPM-bound)
        self.assertAlmostEqual(pro.sweep_values("short_chat")[i], 50 / 60, places=3)

    def test_quota_relative_sweep_without_a_quota_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "quota.rpm"):
            load_experiment("experiments/capacity-reference-rate.yaml", NO_QUOTA)

    def test_concurrency_sweep_needs_no_quota(self):
        load_experiment("experiments/capacity-reference-concurrency.yaml", NO_QUOTA)

    def test_experiment_files_with_a_model_are_rejected(self):
        for key in ("target: {model_id: m}\n", "quota_snapshot: {rpm: 1}\n"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "model-agnostic"):
                _load_text(MINIMAL + key)


class SloProfileTests(unittest.TestCase):
    def test_workloads_resolve_the_slo_profile_the_catalog_binds(self):
        spec = load_experiment("experiments/capacity-reference-concurrency.yaml", MICRO)
        self.assertEqual(spec.slo_for("short_chat").tpot_p95_ms, 40)        # gold
        self.assertEqual(spec.slo_for("rag_answer").tpot_p95_ms, 70)        # silver
        self.assertEqual(spec.slo_for("long_generation").tpot_p95_ms, 120)  # bronze


class WorkloadE2ECapTests(unittest.TestCase):
    def test_workload_latency_cap_is_applied_over_its_profile(self):
        spec = load_experiment("experiments/capacity-shape-concurrency.yaml", MICRO)
        for w in spec.workloads:
            with self.subTest(workload=w.name):
                slo = spec.slo_for(w.name)
                self.assertEqual(slo.latency_p95_ms, w.latency_p95_ms)         # workload-level
                self.assertEqual(slo.tpot_p95_ms, spec.slo_profiles[w.slo_profile].tpot_p95_ms)  # profile-level

    def test_every_catalog_workload_has_its_own_e2e_cap(self):
        from bedrock_benchmark.workload import load_workloads
        caps = {n: w.latency_p95_ms for n, w in load_workloads().items()}
        self.assertTrue(all(caps.values()))
        self.assertLess(caps["tiny_request"], caps["short_chat"])
        self.assertLess(caps["short_chat"], caps["rag_answer"])
        self.assertLess(caps["rag_answer"], caps["long_generation"])


class SloProfileFilterTests(unittest.TestCase):
    def test_isolated_sweep_keeps_only_matching_workloads(self):
        spec = load_experiment("experiments/capacity-reference-concurrency.yaml", MICRO, only_slo_profiles={"gold"})
        self.assertEqual([w.name for w in spec.workloads], ["short_chat"])
        self.assertEqual(set(spec.provider_ceilings), {"short_chat"})

    def test_several_profiles(self):
        spec = load_experiment("experiments/capacity-reference-concurrency.yaml", MICRO, only_slo_profiles={"gold", "bronze"})
        self.assertEqual([w.name for w in spec.workloads], ["short_chat", "long_generation"])

    def test_silver_selects_both_silver_shapes(self):
        spec = load_experiment("experiments/capacity-shape-concurrency.yaml", MICRO, only_slo_profiles={"silver"})
        self.assertEqual([w.name for w in spec.workloads], ["medium_context", "long_context_short_answer"])

    def test_nothing_matching_is_a_skip_not_an_error(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(MINIMAL)  # short_chat (gold) only
        try:
            with self.assertRaisesRegex(NoMatchingWorkloads, "bronze"):
                load_experiment(f.name, MICRO, only_slo_profiles={"bronze"})
        finally:
            Path(f.name).unlink()

    def test_partial_mix_is_skipped_whole_mix_runs(self):
        with self.assertRaisesRegex(NoMatchingWorkloads, "partial mix"):
            load_experiment("experiments/capacity-mix-rate.yaml", MICRO, only_slo_profiles={"gold"})
        spec = load_experiment("experiments/capacity-mix-rate.yaml", MICRO,
                               only_slo_profiles={"gold", "silver", "bronze"})
        self.assertEqual(len(spec.workloads), 3)

    def test_unknown_profile_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "gld"):
            load_experiment("experiments/capacity-reference-rate.yaml", MICRO, only_slo_profiles={"gld"})


class ValidationTests(unittest.TestCase):
    def test_defaults(self):
        spec = _load_text(MINIMAL)
        self.assertEqual((spec.warmup_s, spec.repetitions, spec.slo.confidence), (0.0, 1, 0.95))  # constraints/slo.yaml sets it
        self.assertEqual(spec.transport.total_max_attempts, 1)
        self.assertEqual(spec.workload_validation_tolerance_pct, 10.0)

    def test_invalid_specs_are_rejected(self):
        bad = [
            MINIMAL + "repetitions: 0\n",
            MINIMAL + "slo: {confidence: 1.5}\n",
            MINIMAL + "mix: {name: x, weights: {nope: 1}}\n",
            MINIMAL + "mix: {name: x, weights: {short_chat: 0}}\n",
            MINIMAL.replace("values: [1]", "values: [1], quota_fractions: [1.0]"),
            MINIMAL.replace("values: [1]", "quota_fractions: [1.0]"),  # concurrency can't be quota-relative
            MINIMAL.replace("{type: concurrency, values: [1]}", "{type: rate}"),
            MINIMAL.replace("values: [1]", "values: [1, 128]"),  # > transport.max_connections (64)
            MINIMAL.replace("[short_chat]", "[nope]"),                          # not in the catalog
            MINIMAL.replace("[short_chat]", "[short_chat, short_chat]"),        # listed twice
            MINIMAL.replace("[short_chat]", "[{name: w, input_tokens: 1, output_tokens: 1}]"),  # inline shape
            MINIMAL + "mix: {name: x, weights: {rag_answer: 1}}\n",            # mixes an unlisted workload
            MINIMAL + "transport: {max_connections: 64, executor_workers: 8}\n",
            MINIMAL + "quota_headroom: 1.0\n",
            MINIMAL + "confirmation: {max_looks: 0}\n",
            MINIMAL + "confirmation: {max_duration_s: 0}\n",
            MINIMAL + "confirmation: {cooldown_s: -1}\n",
            MINIMAL + "provider_headroom: 0.2\n",                              # policy lives in constraints/
            MINIMAL.replace("purpose: reference\n", ""),                       # purpose is explicit
            MINIMAL.replace("confirmation: {max_looks: 2}\n", ""),             # reference needs confirmation
            MINIMAL.replace("purpose: reference", "purpose: admission_calibration")
                   .replace("confirmation: {max_looks: 2}\n", ""),             # so does admission_calibration
            MINIMAL.replace("values: [1]}", "values: [1], refinement: {strategy: golden_section}}"),
            MINIMAL.replace("values: [1]}", "values: [1], refinement: {max_points: 0}}"),
            MINIMAL.replace("values: [1]}", "values: [1], refinement: {stop_when_adjacent: false}}"),
            MINIMAL.replace("{type: concurrency, values: [1]}", "{type: rate, values: [1], refinement: {}}"),
            MINIMAL.replace("[short_chat]", "[short_chat, tiny_request]"),      # reference lists a characterization workload
            MINIMAL + "throttle_pause_s: -1\n",
            MINIMAL.replace("{type: concurrency, values: [1]}", "{type: rate, values: [1]}") + "throttle_pause_s: 1\n",  # open-loop
        ]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                _load_text(text)


if __name__ == "__main__":
    unittest.main()
