"""Independent runtime closure facts and strict release drift controls."""

import copy
import importlib.metadata
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from src import external_inference, external_release
from tests.external_fixtures import ReleaseFixture, write


class RuntimeReleaseTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.fixture = ReleaseFixture(self.root)
        self.frozen = json.loads((self.root / "environment.json").read_text())

    def verify(self):
        return external_release.verify_release(
            self.root / "release.json", self.fixture.save(), self.root
        )

    def test_active_closure_has_independent_expected_numerical_dependencies(self):
        software = self.frozen["software"]
        self.assertEqual(self.frozen["schema_version"], 2)
        self.assertEqual(list(software), sorted(software))
        self.assertTrue(
            {
                "numpy",
                "pandas",
                "scikit-learn",
                "torch",
                "torchvision",
                "pillow",
                "opencv-python-headless",
                "matplotlib",
                "seaborn",
                "tqdm",
                "scipy",
                "joblib",
                "threadpoolctl",
                "sympy",
                "mpmath",
                "packaging",
            }
            <= set(software)
        )
        self.assertTrue(all(name == name.lower() for name in software))
        self.assertEqual(self.verify()["schema_version"], 2)

    def test_installed_scipy_numpy_and_joblib_drift_reject_before_prediction(self):
        original = importlib.metadata.version
        for package in ("scipy", "numpy", "joblib"):
            with self.subTest(package=package):

                def changed(name):
                    drift = {"scipy": "1.17.2", "numpy": "2.4.7", "joblib": "1.5.4"}
                    return drift[name] if name == package else original(name)

                with patch.object(
                    external_release.importlib.metadata,
                    "version",
                    side_effect=changed,
                ):
                    with self.assertRaisesRegex(ValueError, "environment|dependency"):
                        self.verify()
                    with patch.object(external_inference, "predict_raw") as predictor:
                        with self.assertRaises(ValueError):
                            external_inference.evaluate_run(**self.fixture.kwargs())
                        predictor.assert_not_called()

    def test_omitted_entry_old_format_and_unknown_format_reject(self):
        for mutation in (
            "omit scipy",
            "recorded scipy version",
            "old format",
            "unknown format",
        ):
            with self.subTest(mutation=mutation):
                env = copy.deepcopy(self.frozen)
                if mutation == "omit scipy":
                    del env["software"]["scipy"]
                elif mutation == "recorded scipy version":
                    env["software"]["scipy"] = "0.0"
                elif mutation == "old format":
                    del env["schema_version"]
                else:
                    env["schema_version"] = 99
                write(self.root / "environment.json", env)
                self.fixture.pin("environment.json")
                message = {
                    "omit scipy": "omits runtime dependencies: scipy",
                    "recorded scipy version": "version mismatch: scipy",
                    "old format": "unsupported execution environment schema",
                    "unknown format": "unsupported execution environment schema",
                }[mutation]
                with self.assertRaisesRegex(ValueError, message):
                    self.verify()

    def test_missing_required_distribution_metadata_rejects(self):
        original = importlib.metadata.distribution

        def missing(name):
            if name == "scipy":
                raise importlib.metadata.PackageNotFoundError(name)
            return original(name)

        with patch.object(
            external_release.importlib.metadata, "distribution", side_effect=missing
        ):
            with self.assertRaisesRegex(ValueError, "scipy"):
                self.verify()

    def test_reporting_rechecks_installed_scipy_before_reconstruction(self):
        original = importlib.metadata.version
        kwargs = self.fixture.kwargs()
        common = {
            name: value
            for name, value in kwargs.items()
            if name not in {"images_dir", "run_id", "output"}
        }
        output = self.root / "unaccepted-report.json"

        def changed(name):
            return "1.17.2" if name == "scipy" else original(name)

        with patch.object(
            external_release.importlib.metadata, "version", side_effect=changed
        ):
            with self.assertRaisesRegex(ValueError, "version mismatch: scipy"):
                external_inference.write_report(
                    **common,
                    runs_dir=self.root / "unmade-runs",
                    output_path=output,
                )
        self.assertFalse(output.exists())

    def test_unrelated_tool_version_does_not_change_boundary(self):
        original = importlib.metadata.version

        def changed(name):
            return "999" if name == "ruff" else original(name)

        with patch.object(
            external_release.importlib.metadata, "version", side_effect=changed
        ):
            self.assertEqual(self.verify()["schema_version"], 2)


class RuntimeGraphTests(unittest.TestCase):
    def test_nested_active_requirements_markers_extras_and_unrelated_tool(self):
        graph = {
            "root": (
                "1.0",
                [
                    "Bridge>=2",
                    "LinuxOnly; sys_platform == 'linux'",
                    "OtherOS; sys_platform == 'win32'",
                    "Optional; extra == 'all'",
                ],
            ),
            "bridge": ("2.1", ["SciPy>=1", "Joblib>=1; python_version >= '3.14'"]),
            "scipy": ("1.17", []),
            "joblib": ("1.5", []),
            "linuxonly": ("1.0", []),
            "otheros": ("1.0", []),
            "optional": ("1.0", []),
            "unrelated-tool": ("9.0", []),
        }

        def distribution(name):
            version, requires = graph[name.lower()]
            return SimpleNamespace(
                version=version, requires=requires, metadata={"Name": name}
            )

        def version(name):
            return graph[name.lower()][0]

        with (
            patch.object(
                external_release.importlib.metadata,
                "distribution",
                side_effect=distribution,
            ),
            patch.object(
                external_release.importlib.metadata, "version", side_effect=version
            ),
        ):
            software = external_release.runtime_software(["Root==1.0"])
        expected = {"root", "bridge", "scipy", "joblib"}
        expected.add(
            "linuxonly"
            if external_release.platform.system() == "Linux"
            else "otheros"
            if external_release.platform.system() == "Windows"
            else ""
        )
        expected.discard("")
        self.assertEqual(set(software), expected)
        self.assertEqual(list(software), sorted(software))

        with (
            patch.object(
                external_release.importlib.metadata,
                "distribution",
                side_effect=distribution,
            ),
            patch.object(
                external_release.importlib.metadata, "version", side_effect=version
            ),
        ):
            with_extra = external_release.runtime_software(["Root[all]==1.0"])
            forward = external_release.runtime_software(
                ["Root==1.0", "SciPy>=1", "OtherOS; python_version < '3.0'"]
            )
            reversed_roots = external_release.runtime_software(
                ["SciPy>=1", "Root==1.0"]
            )
        self.assertEqual(set(with_extra), expected | {"optional"})
        self.assertEqual(
            json.dumps(forward, separators=(",", ":")),
            json.dumps(reversed_roots, separators=(",", ":")),
        )


if __name__ == "__main__":
    unittest.main()
