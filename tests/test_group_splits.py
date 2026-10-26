import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from attentiontrack.dataset import load_npz_paths, NPZSequenceDataset
from attentiontrack.group_splits import read_fold_manifest
from scripts.make_group_split import assign_folds, load_records, main


class GroupSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.features = self.root / "features"
        (self.features / "train").mkdir(parents=True)
        self.source = self.root / "source.csv"
        self.source_rows = []
        for subject in range(15):
            for label in (0, 1):
                clip = f"{subject:06d}{label:04d}"
                np.savez(self.features / "train" / f"{clip}.npz",
                         X=np.ones((3, 2), dtype=np.float32), y=label, length=3)
                self.source_rows.append({"filename": f"{clip}.npz", "label": label,
                                         "path": f"raw/train/{subject:06d}/{clip}/{clip}.npz"})
        self.write_source()

    def write_source(self):
        with self.source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename", "label", "path"])
            writer.writeheader()
            writer.writerows(self.source_rows)

    def test_reproducible_disjoint_complete_folds(self):
        records = load_records(self.features, self.source)
        folds = assign_folds(records)
        self.assertEqual(folds, assign_folds(records))
        tested = Counter()
        for rows in folds:
            self.assertEqual(len(rows), 30)
            groups = [{r["subject_id"] for r in rows if r["split"] == split}
                      for split in ("train", "val", "test")]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            tested.update(r["path"] for r in rows if r["split"] == "test")
        self.assertEqual(len(tested), 30)
        self.assertEqual(set(tested.values()), {1})

    def test_manifest_loader_and_dataset_integration(self):
        output = self.root / "folds"
        main(["--features_dir", str(self.features), "--source_manifest", str(self.source),
              "--output_dir", str(output)])
        self.assertTrue((output / "protocol.json").is_file())
        fold = output / "fold_1"
        splits = read_fold_manifest(fold / "group_manifest.csv")
        for name, paths in splits.items():
            self.assertEqual(load_npz_paths(fold, name), paths)
            dataset = NPZSequenceDataset(paths)
            self.assertEqual(tuple(dataset[0][0].shape), (3, 2))
        self.assertEqual(len(load_npz_paths(self.features, "train")), 30)

    def test_label_mismatch_and_missing_mapping_fail(self):
        self.source_rows[0]["label"] = 1
        self.write_source()
        with self.assertRaisesRegex(ValueError, "label mismatch"):
            load_records(self.features, self.source)
        self.source_rows.pop(0)
        self.write_source()
        with self.assertRaisesRegex(ValueError, "missing from source"):
            load_records(self.features, self.source)

    def test_ambiguous_subject_path_fails(self):
        self.source_rows[0]["path"] = "flat/clip.npz"
        self.write_source()
        with self.assertRaisesRegex(ValueError, "Cannot establish subject"):
            load_records(self.features, self.source)

    def test_loader_rejects_subject_leakage(self):
        records = load_records(self.features, self.source)
        rows = assign_folds(records)[0]
        subject = rows[0]["subject_id"]
        affected = [row for row in rows if row["subject_id"] == subject]
        affected[0]["split"] = "test"
        affected[1]["split"] = "train"
        manifest = self.root / "group_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "subject_id", "label", "split"])
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "Subject leakage"):
            read_fold_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
