"""Freeze development groups once before comparing models."""

import argparse
from pathlib import Path

from src import data, splitting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=["ham10000", "isic2018_task3"], required=True
    )
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    args = parser.parse_args()
    frame, _, _ = data.prepare_dataset(
        str(args.metadata_path),
        str(args.images_dir),
        source=args.source,
        attrition_path=args.split_manifest.parent / "cohort_attrition.json",
    )
    if args.split_manifest.exists():
        manifest = splitting.load_manifest(frame, args.split_manifest)
        print("Validated existing manifest; allocation file unchanged.")
    else:
        manifest = splitting.create_manifest(frame)
        splitting.save_manifest(manifest, args.split_manifest)
        print("Saved development manifest.")
    print(f"cohort_hash: {manifest['cohort_hash']}")
    print(f"split_hash: {manifest['split_hash']}")
    parts, _ = splitting.load_development_split(frame, args.split_manifest)
    splitting.print_split_summary(
        frame.target.to_numpy(), parts, frame.lesion_id.to_numpy()
    )


if __name__ == "__main__":
    main()
