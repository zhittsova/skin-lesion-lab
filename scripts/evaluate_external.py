"""Evaluate frozen releases on an audited external cohort; never fit models."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import external_inference  # noqa: E402


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
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--legacy-source-root", type=Path)
    parser.add_argument("--legacy-protocol", type=Path)
    args = parser.parse_args()
    if args.operation == "run" and (args.legacy_source_root or args.legacy_protocol):
        parser.error("legacy inspection cannot run inference")
    if bool(args.legacy_source_root) != bool(args.legacy_protocol):
        parser.error(
            "legacy inspection requires source and protocol snapshots together"
        )
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
    report = external_inference.write_report(
        root=ROOT,
        release_path=args.release,
        release_sha256=args.release_sha256,
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        audit_path=args.audit,
        audit_sha256=args.audit_sha256,
        runs_dir=args.runs_dir,
        output_path=args.output_report,
        legacy_source_root=args.legacy_source_root,
        legacy_protocol_path=args.legacy_protocol,
    )
    print(json.dumps({"counts": report["counts"], "contract": report["contract"]}))


if __name__ == "__main__":
    main()
