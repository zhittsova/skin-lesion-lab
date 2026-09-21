"""Evaluate frozen releases on an audited external cohort; never fit models."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import external, external_inference, run_contract  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["run", "report"])
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.operation == "run":
        if any(
            value is None
            for value in (args.audit, args.audit_sha256, args.images_dir, args.run_id)
        ):
            parser.error("run requires audit, audit-sha256, images-dir, and run-id")
        result = external_inference.evaluate_run(
            root=ROOT,
            release_path=args.release,
            release_sha256=args.release_sha256,
            manifest_path=args.manifest,
            manifest_sha256=args.manifest_sha256,
            audit_path=args.audit,
            audit_sha256=args.audit_sha256,
            images_dir=args.images_dir,
            run_id=args.run_id,
            output=args.runs_dir / args.run_id,
            device=args.device,
        )
        print(
            json.dumps(
                {
                    "run_id": result["run_id"],
                    "status": result["status"],
                    "runtime_seconds": result["runtime_seconds"],
                }
            )
        )
        return
    release = external.verify_files(args.release, args.release_sha256, ROOT)
    if run_contract.sha256(args.manifest) != args.manifest_sha256:
        raise ValueError("external manifest hash mismatch")
    manifest = run_contract._json(args.manifest)
    external_inference.validate_audit(
        args.audit, args.audit_sha256, args.manifest_sha256
    )
    models = {}
    registry = {}
    for name, run in release["runs"].items():
        path = args.runs_dir / name
        frame = external_inference.validate_predictions(
            path,
            release,
            args.release_sha256,
            manifest,
            args.manifest_sha256,
            ROOT,
            expected_run_id=name,
            audit_sha256=args.audit_sha256,
        )
        models.setdefault(run["family"], {})[run["seed"]] = frame
        registry[name] = {
            "run_sha256": run_contract.sha256(path / "run.json"),
            "prediction_sha256": run_contract.sha256(path / "predictions.csv"),
        }
    report = external.paired_report(models)
    report.update(
        release_sha256=args.release_sha256,
        manifest_sha256=args.manifest_sha256,
        cohort_sha256=manifest["cohort_sha256"],
        audit_sha256=args.audit_sha256,
        registry=registry,
        exclusions=manifest["counts"],
    )
    with (args.runs_dir / "external-report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "report": str(args.runs_dir / "external-report.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
