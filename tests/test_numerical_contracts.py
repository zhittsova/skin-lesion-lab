"""Independent bin, denominator and rational decision oracles."""

import contextlib
import io
import itertools
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np
from src import bayes, calibration, evaluation, reporting


class NumericalContractTests(unittest.TestCase):
    def test_reliability_and_ece_reject_invalid_inputs(self):
        invalid = [
            ([], [], 2),
            ([0], [0.1, 0.2], 2),
            ([[0]], [0.1], 2),
            ([0], [[0.1]], 2),
            ([2], [0.1], 2),
            ([None], [0.1], 2),
        ]
        invalid += [([0], [p], 2) for p in (np.nan, np.inf, -0.1, 1.1)]
        invalid += [([0], [0.1], n) for n in (0, -1, 1.5, True)]
        for function in (
            evaluation.compute_calibration_curve,
            evaluation.compute_calibration_error,
            calibration.probability_report,
        ):
            for y, p, bins in invalid:
                with self.subTest(function=function.__name__, y=y, p=p, bins=bins):
                    with self.assertRaises(ValueError):
                        function(y, p, bins)

    def test_bin_edges_and_sparse_bins_have_hand_counted_ece(self):
        # Bins [0,.25), [.25,.5), [.5,.75), [.75,1].
        y, p = [0, 1, 0, 1, 1], [0, 0.25, 0.5, 0.75, 1]
        means, frequencies, counts = evaluation.compute_calibration_curve(y, p, 4)
        np.testing.assert_array_equal(counts, [1, 1, 1, 2])
        np.testing.assert_allclose(means, [0, 0.25, 0.5, 0.875])
        np.testing.assert_array_equal(frequencies, [0, 1, 0, 1])
        self.assertAlmostEqual(evaluation.compute_calibration_error(y, p, 4), 0.3)
        self.assertAlmostEqual(evaluation.compute_calibration_error(y, p, 1), 0.1)
        bins = calibration.probability_report([0, 1], [0, 1], 4)["reliability_bins"]
        self.assertEqual([row["count"] for row in bins], [1, 0, 0, 1])
        self.assertIsNone(bins[1]["mean_probability"])
        self.assertIsNone(bins[1]["positive_fraction"])

    def test_confusion_cells_obey_null_denominators(self):
        # Enumerate empty/nonempty cells independently, including one-class draws.
        for tn, fp, fn, tp in itertools.product((0, 1), repeat=4):
            if not tn + fp + fn + tp:
                continue
            y = [0] * (tn + fp) + [1] * (fn + tp)
            predicted = [0] * tn + [1] * fp + [0] * fn + [1] * tp
            metric = evaluation.compute_classification_metrics(y, predicted, predicted)

            def ratio(a, b):
                return a / b if b else None

            expected = dict(
                precision=ratio(tp, tp + fp),
                recall=ratio(tp, tp + fn),
                specificity=ratio(tn, tn + fp),
                f1=ratio(2 * tp, 2 * tp + fp + fn),
            )
            for name, value in expected.items():
                with self.subTest(cells=(tn, fp, fn, tp), metric=name):
                    self.assertEqual(metric[name], value)
            if len(set(y)) == 1:
                self.assertIsNone(metric["roc_auc"])
            json.dumps(metric, allow_nan=False)
        for args in (([], [], []), ([0, 1], [0], [0, 1]), ([0, 2], [0, 1], [0, 1])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                evaluation.compute_classification_metrics(*args)

    def test_versioned_metrics_and_strict_json_formatting(self):
        new = evaluation.compute_classification_metrics([0, 1], [0, 0], [0.2, 0.3])
        old = evaluation.compute_classification_metrics(
            [0, 1], [0, 0], [0.2, 0.3], metrics_version=1
        )
        self.assertIsNone(new["precision"])
        self.assertEqual(old["precision"], 0)
        with self.assertRaises(ValueError):
            evaluation.compute_classification_metrics(
                [0], [0], [0.2], metrics_version=99
            )
        with contextlib.redirect_stdout(io.StringIO()) as output:
            evaluation.print_evaluation_summary(new)
        self.assertIn("undefined", output.getvalue())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            reporting.save_json(new, path)
            self.assertIsNone(json.loads(path.read_text())["precision"])
            for bad in (float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    reporting.save_json({"bad": bad}, path)

    def test_extreme_costs_reject_unrepresentable_cutoffs(self):
        tiny = np.nextafter(0.0, 1.0)
        largest = np.finfo(float).max
        for fn, fp in (
            (1e308, 1e-308),
            (1e-308, 1e308),
            (largest, tiny),
            (tiny, largest),
        ):
            with (
                self.subTest(costs=(fn, fp)),
                self.assertRaisesRegex(ValueError, "represent"),
            ):
                bayes.threshold_with_costs(np.array([0.0, 1.0]), fn, fp)
        for fn, fp in (
            (10.0, 1.0),
            (1.0, 10.0),
            (1.0, 1.0),
            (largest, largest),
            (tiny, tiny),
            (1e300, 1e299),
        ):
            p = [0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0]
            oracle = [
                int(Fraction(fp) * (1 - Fraction(v)) <= Fraction(fn) * Fraction(v))
                for v in p
            ]
            np.testing.assert_array_equal(bayes.threshold_with_costs(p, fn, fp), oracle)
        for fn, fp in ((0, 1), (1, 0), (-1, 1), (np.nan, 1), (1, np.inf)):
            with self.subTest(costs=(fn, fp)), self.assertRaises(ValueError):
                calibration.cost_threshold(fn, fp)

    def test_threshold_enumeration_against_4644_rational_cases(self):
        count = 0
        for n in (2, 3, 4):
            for y in itertools.product((0, 1), repeat=n):
                for p in itertools.product((0.0, 0.5, 1.0), repeat=n):
                    candidates = sorted(set((0.0, *p, np.nextafter(max(p), np.inf))))
                    for fn_cost, fp_cost in ((1, 1), (10, 1), (1, 10)):

                        def key(t):
                            pred = [v >= t for v in p]
                            fp = sum(a == 0 and b for a, b in zip(y, pred))
                            fn = sum(a == 1 and not b for a, b in zip(y, pred))
                            tp, tn = sum(y) - fn, n - sum(y) - fp
                            return (
                                Fraction(fn_cost * fn + fp_cost * fp, n),
                                -Fraction(tp, max(1, sum(y))),
                                -Fraction(tn, max(1, n - sum(y))),
                                t,
                            )

                        expected = min(candidates, key=key)
                        actual, _ = calibration.select_cost_threshold(
                            y, p, fn_cost, fp_cost
                        )
                        self.assertEqual(actual, expected)
                        count += 1
        self.assertEqual(count, 4644)


class MetricExportTests(unittest.TestCase):
    def test_null_exports_and_plot_labels_do_not_turn_undefined_into_zero(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        from src import plots

        metric = evaluation.compute_classification_metrics([0, 1], [0, 0], [0.1, 0.2])
        metric = evaluation.add_average_cost(metric, 2, 10, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reporting.save_metrics_tables(
                {"development": {"map": metric}}, {"development": 0.45}, root
            )
            frame = pd.read_csv(root / "metrics_summary.csv")
            self.assertTrue(pd.isna(frame.precision.iloc[0]))
            self.assertEqual(frame.f1.iloc[0], 0.0)
            figure = plots.plot_threshold_comparison({"map": metric})
            self.assertIn(
                "undefined", [text.get_text() for text in figure.axes[0].texts]
            )
            self.assertTrue(np.isnan(figure.axes[0].patches[2].get_height()))
            figure.savefig(root / "null-metric.png")
            plt.close(figure)

    def test_linspace_edge_membership_and_reliability_agree(self):
        edge = np.linspace(0, 1, 11)[3]
        p = [np.nextafter(edge, 0), edge, np.nextafter(edge, 1)]
        report = calibration.probability_report([0, 1, 1], p)
        _, _, counts = evaluation.compute_calibration_curve([0, 1, 1], p)
        self.assertEqual(counts[2:4].tolist(), [1, 2])
        self.assertEqual(
            counts.tolist(), [row["count"] for row in report["reliability_bins"]]
        )

    def test_policy_report_dispatches_metric_bins_independently_of_policy_version(self):
        fixture = json.loads(
            (Path(__file__).parent / "fixtures/legacy-policy-v1.json").read_text()
        )
        policy = fixture["policy"]
        current = calibration.policy_report(policy, [0, 1], [0.3, 0.35])
        legacy = calibration.policy_report(
            policy, [0, 1], [0.3, 0.35], metrics_version=1
        )
        self.assertEqual(
            [row["count"] for row in current["raw_score"]["reliability_bins"]][2:4],
            [1, 1],
        )
        self.assertEqual(
            [row["count"] for row in legacy["raw_score"]["reliability_bins"]][2:4],
            [0, 2],
        )
        for version in (True, 0, 3):
            with self.subTest(version=version), self.assertRaises(ValueError):
                calibration.policy_report(
                    policy, [0, 1], [0.3, 0.35], metrics_version=version
                )
