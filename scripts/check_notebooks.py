"""Execute notebook sources in clean kernels and save private results."""

import argparse
import os
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("01-groups.ipynb", "02-development-report.ipynb")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "accepted"), required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.mode == "accepted" and args.report is None:
        parser.error("--report is required for accepted mode")
    os.environ["SKIN_LESION_MODE"] = args.mode
    if args.report is not None:
        os.environ["SKIN_LESION_REPORT"] = str(args.report.resolve())
    os.environ["MPLBACKEND"] = "Agg"
    destination = ROOT / "outputs/notebooks"
    destination.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"{args.mode}-", dir=destination))
    for source in SOURCES:
        notebook = nbformat.read(ROOT / "notebooks" / source, as_version=4)
        NotebookClient(
            notebook,
            timeout=120,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        if source == "02-development-report.ipynb":
            outputs = [
                item for cell in notebook.cells for item in cell.get("outputs", [])
            ]
            plots = sum("image/png" in item.get("data", {}) for item in outputs)
            required_plots = 2 if args.mode == "accepted" else 1
            if plots != required_plots:
                raise RuntimeError("development notebook did not render expected plots")
            printed = "".join(item.get("text", "") for item in outputs)
            if "run IDs:" not in printed or (
                args.mode == "synthetic" and "synthetic-" not in printed
            ):
                raise RuntimeError("development notebook did not print source run IDs")
        nbformat.write(notebook, output / source)
        print(f"executed {source}: {output / source}")


if __name__ == "__main__":
    main()
