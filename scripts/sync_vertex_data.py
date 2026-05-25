from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run_gsutil(args: list[str]) -> None:
    command = ["gsutil", *args]
    print("Running:", " ".join(command))
    result = subprocess.run(command)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def sync_vertex_data(bucket: str, raw_dir: Path, processed_dir: Path) -> None:
    labels_path = processed_dir / "binary_labels.csv"
    report_path = processed_dir / "cleaning_report.json"

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw image directory not found: {raw_dir}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Processed labels file not found: {labels_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Cleaning report file not found: {report_path}")

    bucket_uri = f"gs://{bucket}"
    raw_target = f"{bucket_uri}/data/raw/nih_chest_xray"
    processed_target = f"{bucket_uri}/data/processed"

    run_gsutil(["-m", "rsync", "-r", str(raw_dir), raw_target])
    run_gsutil(["cp", str(labels_path), f"{processed_target}/binary_labels.csv"])
    run_gsutil(["cp", str(report_path), f"{processed_target}/cleaning_report.json"])

    print("Vertex data sync complete.")
    print(f"Images: {raw_target}")
    print(f"Labels: {processed_target}/binary_labels.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync local DVC-produced data to the GCS bucket used by Vertex AI training."
    )
    parser.add_argument(
        "--bucket",
        default="nih-xray-data",
        help="GCS bucket name used by Vertex training, without gs://.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/nih_chest_xray"),
        help="Local raw image directory to sync.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Local processed directory containing binary_labels.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sync_vertex_data(args.bucket, args.raw_dir, args.processed_dir)
