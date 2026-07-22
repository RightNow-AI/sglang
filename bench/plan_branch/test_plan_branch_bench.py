import unittest

from bench.plan_branch.plan_branch_bench import (
    compute_trial_metrics,
    evaluate_publish_guards,
    numeric_stats,
)


class MetricsMathTest(unittest.TestCase):
    def test_metrics_math_and_missing_tokens(self):
        metrics = compute_trial_metrics(2.0, 100, 3.0)
        self.assertEqual(metrics["total_completion_tokens"], 100)
        self.assertEqual(metrics["throughput_tokens_per_second"], 50.0)
        self.assertAlmostEqual(metrics["gpu_hours_per_1k_trees"], 5 / 9)
        self.assertAlmostEqual(
            metrics["gpu_cost_per_1k_trees_usd"], 5 / 3
        )

        missing = compute_trial_metrics(2.0, None, None)
        self.assertIsNone(missing["total_completion_tokens"])
        self.assertIsNone(missing["throughput_tokens_per_second"])
        self.assertIsNone(missing["gpu_cost_per_1k_trees_usd"])

    def test_mean_median_ignore_null_but_never_zero_fill(self):
        self.assertEqual(
            numeric_stats([1.0, None, 5.0]),
            {"count": 2, "mean": 3.0, "median": 3.0},
        )
        self.assertEqual(
            numeric_stats([None]),
            {"count": 0, "mean": None, "median": None},
        )


class PublishGuardTableTest(unittest.TestCase):
    def test_table_driven_guards(self):
        cases = [
            (
                "publishable",
                {
                    "speedup_x": 2.0,
                    "baseline_prefix_caching": {"vllm_n": "on"},
                    "cold_count": 1,
                    "warm_count": 2,
                    "accuracy_delta_pp": 0.5,
                },
                [],
            ),
            (
                "not dominant",
                {
                    "speedup_x": 1.99,
                    "baseline_prefix_caching": {"vllm_n": "on"},
                    "cold_count": 1,
                    "warm_count": 1,
                    "accuracy_delta_pp": 0.0,
                },
                ["SPEEDUP_LT_2X"],
            ),
            (
                "prefix caching off",
                {
                    "speedup_x": 3.0,
                    "baseline_prefix_caching": {"vllm_n": "off"},
                    "cold_count": 1,
                    "warm_count": 1,
                    "accuracy_delta_pp": 0.0,
                },
                ["BASELINE_PREFIX_CACHING_NOT_ON"],
            ),
            (
                "warm missing",
                {
                    "speedup_x": 3.0,
                    "baseline_prefix_caching": {"vllm_n": "on"},
                    "cold_count": 1,
                    "warm_count": 0,
                    "accuracy_delta_pp": 0.0,
                },
                ["COLD_AND_WARM_NOT_BOTH_REPORTED"],
            ),
            (
                "accuracy delta too large",
                {
                    "speedup_x": 3.0,
                    "baseline_prefix_caching": {"vllm_n": "on"},
                    "cold_count": 1,
                    "warm_count": 1,
                    "accuracy_delta_pp": -0.5001,
                },
                ["ACCURACY_DELTA_GT_0_5PP"],
            ),
        ]
        for name, kwargs, expected in cases:
            with self.subTest(name):
                result = evaluate_publish_guards(**kwargs)
                self.assertEqual(result["triggered_codes"], expected)
                self.assertEqual(result["publishable"], not expected)


if __name__ == "__main__":
    unittest.main()
