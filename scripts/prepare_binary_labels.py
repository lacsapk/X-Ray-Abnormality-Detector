from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


NO_FINDING = "No Finding"
NORMAL_LABEL = "no_abnormality"
ABNORMAL_LABEL = "abnormality"

# Changes the Labels to binary and also adds the full paths to each image, for easier access

# Builds the absolut paths for each image
def build_image_index(raw_dir: Path) -> dict[str, Path]:
    return {path.name: path for path in raw_dir.glob("images_*/images/*.png")}


def clean_labels(raw_dir: Path, output_dir: Path) -> None:
    metadata_path = raw_dir / "Data_Entry_2017.csv"
    output_path = output_dir / "binary_labels.csv"
    report_path = output_dir / "cleaning_report.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    image_index = build_image_index(raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "raw_rows": 0,
        "written_rows": 0,
        "skipped_missing_image_id": 0,
        "skipped_missing_label": 0,
        "skipped_missing_image_file": 0,
        "normal_rows": 0,
        "abnormal_rows": 0,
    }

    # Loading the CSV
    with metadata_path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required_columns = {"Image Index", "Finding Labels"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required column(s): {missing}")

        with output_path.open("w", newline="", encoding="utf-8") as target:
            fieldnames = ["image_id", "image_path", "label", "target"]
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                stats["raw_rows"] += 1

                image_id = (row.get("Image Index") or "").strip()
                finding_label = (row.get("Finding Labels") or "").strip()

                if not image_id:
                    stats["skipped_missing_image_id"] += 1
                    continue
                if not finding_label:
                    stats["skipped_missing_label"] += 1
                    continue

                image_path = image_index.get(image_id)
                if image_path is None:
                    stats["skipped_missing_image_file"] += 1
                    continue

                is_normal = finding_label == NO_FINDING
                label = NORMAL_LABEL if is_normal else ABNORMAL_LABEL
                target_value = 0 if is_normal else 1

                writer.writerow(
                    {
                        "image_id": image_id,
                        "image_path": image_path.as_posix(),
                        "label": label,
                        "target": target_value,
                    }
                )

                stats["written_rows"] += 1
                if is_normal:
                    stats["normal_rows"] += 1
                else:
                    stats["abnormal_rows"] += 1

    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(stats, report_file, indent=2)
        report_file.write("\n")

    print(f"Wrote cleaned labels to: {output_path}")
    print(f"Wrote cleaning report to: {report_path}")
    print(json.dumps(stats, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create binary abnormality/no_abnormality labels for NIH Chest X-rays."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/nih_chest_xray"),
        help="Directory containing Data_Entry_2017.csv and images_* folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where cleaned files will be written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    clean_labels(args.raw_dir, args.output_dir)
