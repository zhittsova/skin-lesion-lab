"""Portable MC input and threshold figure contracts."""

import tempfile
import unittest
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plot_mc_dropout_uncertainty as mc_plot
from src import plots

matplotlib.use("Agg")


def entropy(probability):
    if probability in (0, 1):
        return 0.0
    return -probability * np.log(probability) - (1 - probability) * np.log1p(
        -probability
    )


class McCsvTests(unittest.TestCase):
    def test_csv_npy_statistics_match_independent_passes_by_samples_oracle(self):
        arrays = (
            [[0.1, 0.2, 0.7], [0.5, 0.8, 0.9]],
            [[0.1, 0.6], [0.3, 0.8], [0.9, 0.2]],
            [[0.1, 0.4], [0.7, 0.9]],
            [[0.1], [0.9]],
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, rows in enumerate(arrays):
                with self.subTest(shape=np.shape(rows)):
                    array = np.asarray(rows)
                    csv = root / f"{index}.csv"
                    npy = root / f"{index}.npy"
                    header = ",".join(f"sample_{j}" for j in range(array.shape[1]))
                    csv.write_text(
                        header
                        + "\n"
                        + "\n".join(",".join(map(str, row)) for row in rows)
                        + "\n"
                    )
                    np.save(npy, array)
                    csv_values = mc_plot.load_mc_probabilities(csv)
                    npy_values = mc_plot.load_mc_probabilities(npy)
                    np.testing.assert_array_equal(csv_values, array)
                    np.testing.assert_array_equal(npy_values, array)
                    for values in (csv_values, npy_values):
                        summary = mc_plot.build_uncertainty_frame(
                            values, None, "fixture"
                        )
                        for sample in range(array.shape[1]):
                            column = array[:, sample]
                            mean = sum(column) / len(column)
                            variance = sum((p - mean) ** 2 for p in column) / (
                                len(column) - 1
                            )
                            self.assertAlmostEqual(
                                summary.mean_prob_melanoma[sample], mean
                            )
                            self.assertAlmostEqual(
                                summary.predictive_uncertainty[sample], variance**0.5
                            )
                            self.assertAlmostEqual(
                                summary.predictive_entropy[sample], entropy(mean)
                            )
                            self.assertAlmostEqual(
                                summary.mutual_information[sample],
                                entropy(mean) - sum(map(entropy, column)) / len(column),
                            )

    def test_rejects_ambiguous_or_malformed_csv_and_invalid_npy(self):
        invalid = (
            "",
            "sample_0\n",
            "sample_0\n0.1\n",
            "sample_0,sample_1\n0.1,0.2\n0.3\n",
            "sample_0,sample_1\n0.1,0.2\n0.3,\n",
            "sample_0,sample_1\n0.1,0.2,0.4\n0.3,0.5\n",
            "sample_0,sample_0\n0.1,0.2\n0.3,0.4\n",
            "0,1\n0.1,0.2\n0.3,0.4\n",
            "pass,sample_0\n1,0.2\n2,0.4\n",
            "sample_0,sample_2\n0.1,0.2\n0.3,0.4\n",
            "sample_0\n0.1\nabc\n",
            "sample_0\n0.1\nnan\n",
            "sample_0\n0.1\ninf\n",
            "sample_0\n0.1\n-0.1\n",
            "sample_0\n0.1\n1.1\n",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "bad.csv"
            for content in invalid:
                with self.subTest(content=content):
                    path.write_text(content)
                    with self.assertRaises(ValueError):
                        mc_plot.load_mc_probabilities(path)
            npy = root / "bad.npy"
            for values in ([[0.1]], [[0.1], [np.nan]], [[0.1], [1.1]]):
                with self.subTest(npy=values):
                    np.save(npy, np.asarray(values))
                    with self.assertRaises(ValueError):
                        mc_plot.load_mc_probabilities(npy)


class ThresholdChartTests(unittest.TestCase):
    def test_cost_range_annotations_and_default_role(self):
        for cost in (0.0, 0.4, 2.75, 8.0):
            with self.subTest(cost=cost):
                metrics = {
                    "recall": 0.5,
                    "specificity": 0.6,
                    "precision": None,
                    "f1": 0.0,
                    "average_cost": cost,
                }
                figure = plots.plot_threshold_comparison({"MAP": metrics})
                try:
                    axis = figure.axes[0]
                    self.assertIn("development", axis.get_title().lower())
                    self.assertGreater(axis.get_ylim()[1], cost)
                    self.assertEqual(axis.patches[-1].get_height(), cost)
                    self.assertIn("undefined", [t.get_text() for t in axis.texts])
                    self.assertIn(f"{cost:.2f}", [t.get_text() for t in axis.texts])
                    with tempfile.TemporaryDirectory() as temporary:
                        figure.savefig(Path(temporary) / "threshold.png")
                finally:
                    plt.close(figure)


if __name__ == "__main__":
    unittest.main()
