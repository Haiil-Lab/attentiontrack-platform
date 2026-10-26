import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import aggregate_5runs as aggregate


class AggregateResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_metrics(self, relative, payload):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def metrics(self, value=0.5):
        return {"test": dict.fromkeys(aggregate.METRICS, value)}

    def test_baselines_and_experiments_are_preserved(self):
        self.write_metrics("batch_a/xgboost_run1/metrics.json", self.metrics(0.6))
        self.write_metrics("batch_b/xgboost_run1/metrics.json", self.metrics(0.9))
        self.write_metrics("batch_a/baseline_tabular_run1/baseline_metrics.json", {
            "results": {"logistic_regression": self.metrics(), "random_forest": self.metrics()}
        })
        rows, warnings = aggregate.collect_rows(self.root)
        self.assertFalse(warnings)
        summaries = aggregate.summarize_rows(rows)
        self.assertEqual(len(summaries), 4)
        xgb = [s for s in summaries if s["Model"] == "XGBoost"]
        self.assertEqual({s["Experiment"] for s in xgb}, {"batch_a", "batch_b"})
        self.assertTrue(all(s["Records"] == 1 for s in xgb))
        self.assertEqual({r["model"] for r in rows}, {"XGBoost", "Logistic Regression", "Random Forest"})

    def test_invalid_metrics_have_separate_counts(self):
        payload = self.metrics()
        payload["test"]["roc_auc"] = float("inf")
        del payload["test"]["f1"]
        self.write_metrics("gru_run1/metrics.json", payload)
        self.write_metrics("gru_run2/metrics.json", self.metrics(0.7))
        rows, warnings = aggregate.collect_rows(self.root)
        summary = aggregate.summarize_rows(rows)[0]
        self.assertEqual(len(warnings), 2)
        self.assertEqual(summary["Records"], 2)
        self.assertEqual(summary["ROC-AUC n"], 1)
        self.assertEqual(summary["Accuracy n"], 2)
        self.assertIn("SD unavailable", summary["F1"])
        self.assertEqual(summary["Accuracy"], "0.600 ± 0.141")
        self.assertEqual(aggregate.fmt_mean_std([]), "N/A (n=0)")
        for value in (None, True, "bad", float("nan"), float("-inf")):
            self.assertIsNone(aggregate.safe_float(value))

    def test_duplicate_model_run_fails(self):
        self.write_metrics("xgboost_run1/metrics.json", self.metrics())
        self.write_metrics("xgboost_run1/baseline_metrics.json", {"results": {"xgboost": self.metrics()}})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate.collect_rows(self.root)

    def test_malformed_results_fail(self):
        self.write_metrics("gru_run1/metrics.json", {"val": {"acc": 1.0}})
        with self.assertRaisesRegex(ValueError, "Missing test"):
            aggregate.collect_rows(self.root)

    def test_cli_creates_outputs_and_protects_existing_files(self):
        self.write_metrics("gru_run1/metrics.json", self.metrics())
        prefix = self.root / "output" / "report"
        args = ["--root", str(self.root), "--out_prefix", str(prefix)]
        with contextlib.redirect_stdout(io.StringIO()):
            aggregate.main(args)
        report = Path(f"{prefix}_summary.md")
        before = report.read_bytes()
        self.assertIn("not verified independent folds", before.decode("utf-8"))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            aggregate.main(args)
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(report.read_bytes(), before)
        with contextlib.redirect_stdout(io.StringIO()):
            aggregate.main(args + ["--overwrite"])


if __name__ == "__main__":
    unittest.main()
