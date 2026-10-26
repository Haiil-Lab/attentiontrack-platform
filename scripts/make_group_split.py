"""Create shared DAiSEE participant-level CV manifests, without copying features.

Run from the project root: python -m scripts.make_group_split --help
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from attentiontrack.group_splits import read_fold_manifest


def load_records(features_dir, source_manifest):
    """Use explicit IDs or original subject/clip/file hierarchy, never clip prefixes."""
    files = {}
    for split in ("train", "val", "test"):
        for path in sorted((features_dir / split).rglob("*.npz")):
            if path.name in files:
                raise ValueError(f"Duplicate feature filename: {path.name}")
            files[path.name] = path.resolve()
    if not files:
        raise ValueError(f"No NPZ features found under {features_dir}")
    records = []
    seen = set()
    with source_manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            original = PurePosixPath(row.get("path", "").replace("\\", "/"))
            filename = row.get("filename") or original.name
            if filename in seen:
                raise ValueError(f"Duplicate manifest clip: {filename}")
            seen.add(filename)
            if filename not in files:
                raise ValueError(f"Manifest clip has no feature file: {filename}")
            subject = (row.get("subject_id") or "").strip()
            if not subject:
                # DAiSEE source hierarchy: split / participant / clip / clip.npz.
                if (len(original.parts) < 4 or original.suffix != ".npz" or original.parent.name != original.stem
                        or original.parents[2].name.lower() not in {"train", "val", "test"}
                        or not original.parents[1].name.isdigit()):
                    raise ValueError(f"Cannot establish subject ID for {filename}; supply subject_id in source CSV")
                subject = original.parents[1].name
            path = files[filename]
            with np.load(path, allow_pickle=False) as data:
                value = np.asarray(data["y"])
                if value.size != 1 or value.item() not in (0, 1):
                    raise ValueError(f"Invalid binary label in {path}")
                label = int(value.item())
            if row.get("label") not in {str(label)}:
                raise ValueError(f"Source/NPZ label mismatch for {filename}")
            records.append({"path": str(path), "subject_id": subject, "label": label})
    if seen != set(files):
        raise ValueError(f"{len(set(files) - seen)} feature files missing from source manifest")
    return sorted(records, key=lambda row: row["path"])


def assign_folds(records, n_splits=5, seed=42):
    if n_splits < 3:
        raise ValueError("At least three folds are needed for separate train/val/test groups")
    labels = np.array([row["label"] for row in records])
    groups = np.array([row["subject_id"] for row in records])
    if len(set(groups)) < n_splits:
        raise ValueError("Fewer participants than folds")
    for label in (0, 1):
        if len(set(groups[labels == label])) < n_splits:
            raise ValueError(f"Class {label} occurs in fewer than {n_splits} participants")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    membership = np.full(len(records), -1, dtype=int)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(records)), labels, groups)):
        membership[test] = fold
    folds = []
    for fold in range(n_splits):
        rows = []
        for index, record in enumerate(records):
            split = "test" if membership[index] == fold else "val" if membership[index] == (fold + 1) % n_splits else "train"
            rows.append({**record, "split": split})
        # Fail rather than silently produce undefined binary classification metrics.
        for split in ("train", "val", "test"):
            if {r["label"] for r in rows if r["split"] == split} != {0, 1}:
                raise ValueError(f"Fold {fold + 1} {split} lacks a class; choose fewer folds")
        folds.append(rows)
    return folds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features_dir", type=Path, required=True)
    parser.add_argument("--source_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Output directory already exists; use a new directory to preserve fold assignments")
    records = load_records(args.features_dir, args.source_manifest)
    folds = assign_folds(records, args.n_splits, args.seed)
    report = {
        "protocol": "participant_disjoint_rotating_validation",
        "splitter": "StratifiedGroupKFold",
        "sklearn_version": sklearn.__version__,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "validation": "Next outer group cyclically; all other groups form training",
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": hashlib.sha256(args.source_manifest.read_bytes()).hexdigest(),
        "clips": len(records),
        "participants": len({r["subject_id"] for r in records}),
        "folds": [],
    }
    for index, rows in enumerate(folds, 1):
        directory = args.output_dir / f"fold_{index}"
        directory.mkdir(parents=True)
        manifest = directory / "group_manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "subject_id", "label", "split"])
            writer.writeheader()
            writer.writerows(rows)
        read_fold_manifest(manifest)
        stats = {"fold": index, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
        for split in ("train", "val", "test"):
            subset = [r for r in rows if r["split"] == split]
            stats[split] = {
                "clips": len(subset), "subjects": len({r["subject_id"] for r in subset}),
                "y0": sum(r["label"] == 0 for r in subset), "y1": sum(r["label"] == 1 for r in subset),
            }
        report["folds"].append(stats)
        print(f"fold_{index}: {stats}")
    (args.output_dir / "protocol.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {len(folds)} validated folds to {args.output_dir}. No training was run.")


if __name__ == "__main__":
    main()
