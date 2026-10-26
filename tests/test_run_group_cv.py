import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_group_cv import METRICS, metric_rows, write_report


class GroupCVReportTests(unittest.TestCase):
    def test_baselines_preserve_fold_and_model_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "baseline_metrics.json"
            models = ["logistic_regression", "random_forest"]
            path.write_text(json.dumps({"results": {
                model: {"test": dict.fromkeys(METRICS, 0.5)} for model in models}}))
            rows = metric_rows(path, models, 3)
            self.assertEqual([r["model"] for r in rows], models)
            self.assertEqual([r["fold"] for r in rows], [3, 3])
            write_report(path.parent, rows)
            report = (path.parent / "summary.md").read_text(encoding="utf-8")
            self.assertIn("1/5", report)
            self.assertIn("SD unavailable", report)
            with self.assertRaisesRegex(ValueError, "Duplicate fold"):
                write_report(path.parent, rows + rows)

    def test_nonfinite_metric_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.json"
            path.write_text(json.dumps({"test": dict.fromkeys(METRICS, float("nan"))}))
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                metric_rows(path, ["gru"], 1)


if __name__ == "__main__":
    unittest.main()
