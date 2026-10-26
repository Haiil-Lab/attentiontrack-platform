"""Validated, participant-disjoint manifests shared by every trainer."""
import csv
from pathlib import Path


def read_fold_manifest(manifest):
    manifest = Path(manifest)
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "subject_id", "label", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns {required} in {manifest}")
        rows = list(reader)
    paths = set()
    subjects = {}
    splits = {name: [] for name in ("train", "val", "test")}
    labels = {name: set() for name in splits}
    for row in rows:
        split, subject = row["split"], row["subject_id"].strip()
        if split not in splits or not subject or row["label"] not in {"0", "1"}:
            raise ValueError(f"Invalid split, subject or binary label in {manifest}: {row}")
        path = Path(row["path"])
        if not path.is_absolute():
            path = manifest.parent / path
        path = path.resolve()
        if path in paths:
            raise ValueError(f"Duplicate clip in {manifest}: {path}")
        if subject in subjects and subjects[subject] != split:
            raise ValueError(f"Subject leakage in {manifest}: {subject}")
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.add(path)
        subjects[subject] = split
        splits[split].append(path)
        labels[split].add(row["label"])
    for split in splits:
        if labels[split] != {"0", "1"}:
            raise ValueError(f"{manifest}: {split} must contain both classes")
    return splits
