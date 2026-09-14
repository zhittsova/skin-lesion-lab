"""Regression checks for the publishable environment contract."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_repository.py"
SPEC = importlib.util.spec_from_file_location("check_repository", SCRIPT)
check_repository = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_repository)


class RepositoryContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = SCRIPT.parents[1]
        for name in ("pyproject.toml", ".python-version", "Dockerfile", "uv.lock"):
            shutil.copy2(source / name, self.root / name)
        (self.root / ".github" / "workflows").mkdir(parents=True)
        shutil.copy2(
            source / ".github" / "workflows" / "ci.yml",
            self.root / ".github" / "workflows" / "ci.yml",
        )
        self.root_patch = patch.object(check_repository, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_linux_torch_source_must_be_cpu(self):
        config = self.root / "pyproject.toml"
        config.write_text(
            config.read_text().replace(
                'index = "pytorch-cpu"', 'index = "pytorch-gpu"', 2
            )
        )
        self.assertTrue(any("CPU" in error for error in check_repository.check_pins()))

    def test_lock_must_not_include_cuda_toolkit(self):
        lock = self.root / "uv.lock"
        lock.write_text(
            lock.read_text() + '\n[[package]]\nname = "cuda-toolkit"\nversion = "1.0"\n'
        )
        self.assertTrue(any("CUDA" in error for error in check_repository.check_pins()))

    def test_python_pin_must_match_container(self):
        (self.root / ".python-version").write_text("3.14.6\n")
        self.assertTrue(
            any("Python pin" in error for error in check_repository.check_pins())
        )

    def test_private_path_is_rejected(self):
        self.assertTrue(check_repository.check_paths(["data/raw/private.jpg"]))


if __name__ == "__main__":
    unittest.main()
