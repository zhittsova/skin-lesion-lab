"""Check publishable paths, notebook hygiene, and reproducibility pins."""

import argparse
import json
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIRS = {
    ".agents",
    ".claude",
    ".codex",
    "specs",
    "data",
    "results",
    "reports",
    "models",
    "runs",
    "outputs",
    ".venv",
    "__pycache__",
}
PRIVATE_NAMES = {
    "AGENTS.md",
    "CLAUDE.md",
    "DEFENSE_NOTES.md",
    "MC_DROPOUT_EXTENSION.md",
}
PRIVATE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".pdf",
    ".pptx",
    ".docx",
    ".zip",
    ".pkl",
    ".joblib",
    ".pt",
    ".pth",
    ".npy",
    ".npz",
    ".csv",
    ".tsv",
    ".parquet",
    ".h5",
    ".hdf5",
    ".onnx",
    ".safetensors",
    ".pem",
    ".key",
}


def check_paths(paths: list[str]) -> list[str]:
    errors = []
    for name in paths:
        path = Path(name)
        if (
            set(path.parts) & PRIVATE_DIRS
            or path.name in PRIVATE_NAMES
            or path.name.startswith(("REPORT", ".env"))
            or path.suffix.lower() in PRIVATE_SUFFIXES
            or path.name.endswith(".tar.gz")
        ):
            errors.append(f"Private/generated path: {name}")
        if path.suffix == ".ipynb":
            notebook = json.loads((ROOT / path).read_text())
            for cell in notebook.get("cells", []):
                if cell.get("outputs") or cell.get("execution_count") is not None:
                    errors.append(f"Executed notebook: {name}")
                if cell.get("attachments"):
                    errors.append(f"Embedded notebook attachment: {name}")
    return errors


def check_pins() -> list[str]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    dependencies = list(config["project"]["dependencies"])
    for group in config.get("dependency-groups", {}).values():
        dependencies.extend(item for item in group if isinstance(item, str))
    errors = [
        f"Dependency is not exactly pinned: {dep}"
        for dep in dependencies
        if not re.fullmatch(r"[\w.-]+(?:\[[\w,.-]+\])?==[\w.+-]+", dep)
    ]
    uv_version = config["tool"]["uv"]["required-version"]
    if not re.fullmatch(r"==\d+\.\d+\.\d+", uv_version):
        errors.append("uv must use an exact required-version")
    python_version = (ROOT / ".python-version").read_text().strip()
    dockerfile = (ROOT / "Dockerfile").read_text()
    if f"python:{python_version}-" not in dockerfile:
        errors.append("Python pin differs between Dockerfile and .python-version")
    if f"/uv:{uv_version[2:]}@sha256:" not in dockerfile:
        errors.append("uv pin differs between Dockerfile and pyproject.toml")
    for line in dockerfile.splitlines():
        if line.startswith("FROM ") and not re.search(r"@sha256:[a-f0-9]{64}", line):
            errors.append(f"Container base lacks digest: {line}")
    for workflow in (ROOT / ".github/workflows").glob("*.yml"):
        for ref in re.findall(r"uses:\s*(\S+)", workflow.read_text()):
            if not re.fullmatch(r"[\w./-]+@[a-f0-9]{40}", ref):
                errors.append(f"Action lacks full commit pin: {ref}")

    indexes = {
        item["name"]: item
        for item in config.get("tool", {}).get("uv", {}).get("index", [])
    }
    cpu_index = indexes.get("pytorch-cpu", {})
    if cpu_index.get(
        "url"
    ) != "https://download.pytorch.org/whl/cpu" or not cpu_index.get("explicit"):
        errors.append("CPU PyTorch index must be explicit")
    sources = config.get("tool", {}).get("uv", {}).get("sources", {})
    for package in ("torch", "torchvision"):
        selected = sources.get(package, [])
        if selected != [{"index": "pytorch-cpu", "marker": "sys_platform == 'linux'"}]:
            errors.append(f"CPU source missing for Linux {package}")
        linux_locked = [
            item
            for item in lock.get("package", [])
            if item.get("name") == package
            and item.get("source", {}).get("registry")
            == "https://download.pytorch.org/whl/cpu"
        ]
        if len(linux_locked) != 1 or not linux_locked[0]["version"].endswith("+cpu"):
            errors.append(f"CPU wheel missing from lock for Linux {package}")
    if any(
        item.get("name", "").startswith(("nvidia-", "cuda-"))
        or item.get("name") == "triton"
        for item in lock.get("package", [])
    ):
        errors.append("CUDA package present in default lock")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracked", action="store_true", help="Check Git index paths")
    args = parser.parse_args()
    command = ["git", "ls-files", "-z"]
    if not args.tracked:
        command.extend(["--cached", "--others", "--exclude-standard"])
    paths = subprocess.check_output(command, cwd=ROOT).decode().split("\0")
    errors = check_paths([name for name in paths if name]) + check_pins()
    if errors:
        raise SystemExit("\n".join(errors))
    print("Repository hygiene and pin checks passed")


if __name__ == "__main__":
    main()
