"""Run existing trainers on frozen DAiSEE folds; resume only recorded jobs.

Run from the repository root with python -m scripts.run_group_cv.
One run per fold, not repeated independent CV. Existing BiLSTM checkpoints
are checked against their saved arguments; their training seed is unknown.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import runpy
import subprocess
import sys
import time

from aggregate_5runs import METRICS, LABELS, fmt_mean_std
from attentiontrack.group_splits import read_fold_manifest

ROOT = Path(__file__).resolve().parents[1]
JOBS = {
    "baseline_tabular": ("scripts.baseline_tabular", ["logistic_regression", "random_forest"]),
    "xgboost": ("scripts.train_xgboost", ["xgboost"]),
    "tcn": ("scripts.train_tcn", ["tcn"]),
    "gru": ("scripts.train_gru", ["gru"]),
    "lstm_attention": ("scripts.train_lstm", ["lstm_attention"]),
    "cnn_gru": ("scripts.train_cnn_gru", ["cnn_gru"]),
    "transformer_encoder": ("scripts.train_transformer_encoder", ["transformer_encoder"]),
}
NAMES = {
    "logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
    "xgboost": "XGBoost", "tcn": "TCN", "gru": "GRU",
    "lstm_attention": "LSTM + Temporal Attention", "cnn_gru": "CNN + GRU",
    "transformer_encoder": "Transformer Encoder",
    "bilstm_attention": "BiLSTM + Temporal Attention",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_folds(root):
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if protocol["n_splits"] != 5:
        raise ValueError("This comparison requires exactly five folds")
    counts = Counter()
    universe = None
    hashes = {}
    for fold in range(1, 6):
        manifest = root / f"fold_{fold}" / "group_manifest.csv"
        hashes[fold] = digest(manifest)
        if hashes[fold] != protocol["folds"][fold - 1]["manifest_sha256"]:
            raise ValueError(f"Fold {fold} changed since protocol generation")
        splits = read_fold_manifest(manifest)
        paths = {p for subset in splits.values() for p in subset}
        if universe is not None and paths != universe:
            raise ValueError("Folds do not cover the same clips")
        universe = paths
        counts.update(splits["test"])
    if set(counts) != universe or set(counts.values()) != {1}:
        raise ValueError("Each clip must occur in exactly one test fold")
    return hashes


def metric_rows(path, models, fold):
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload["results"] if "results" in payload else {models[0]: payload}
    rows = []
    for model in models:
        test = results[model]["test"]
        if any(not math.isfinite(float(test[key])) for key in METRICS):
            raise ValueError(f"Non-finite test metric: {path} ({model})")
        rows.append({"model": model, "fold": fold, "metrics_path": str(path),
                     **{key: test[key] for key in METRICS}})
    return rows


def write_report(directory, rows):
    with (directory / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "fold", "metrics_path"] + METRICS)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# DAiSEE participant-disjoint five-fold comparison", "",
        "One training run per fold. Mean and sample SD describe variation across folds, "
        "not independent repetitions or confidence intervals. Training sets overlap across folds.",
        "Existing BiLSTM results use checkpoint-verified fold paths and default settings; "
        "their training seed and historical source hashes were not recorded. "
        "New runs use seed 42. Current manifest hashes are checked against protocol.json.",
        "Tabular models use time-aggregated features; sequence models use sequences. "
        "This compares the existing pipelines with their default hyperparameters, not an equal-budget hyperparameter search.",
        "Do not rank models until every row has five folds. LSTM here includes temporal attention.", "",
        "| Model | Completed folds | " + " | ".join(LABELS[m] for m in METRICS) + " |",
        "| --- | --- | " + " | ".join("---" for _ in METRICS) + " |",
    ]
    for model, name in NAMES.items():
        subset = [r for r in rows if r["model"] == model]
        folds = [r["fold"] for r in subset]
        if len(folds) != len(set(folds)):
            raise ValueError(f"Duplicate fold for {model}")
        lines.append("| " + " | ".join([name, f"{len(subset)}/5"] + [
            fmt_mean_std([r[m] for r in subset]) for m in METRICS]) + " |")
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def worker(module, arguments):
    import numpy as np
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Avoid CPU thread oversubscription on small feature sequences.
    torch.set_num_threads(4)
    sys.argv = [module, *arguments]
    runpy.run_module(module, run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=Path, default=Path("data/processed/daisee_group_cv_v1"))
    parser.add_argument("--results", type=Path, default=Path("runs_full/daisee_group_cv_v1"))
    parser.add_argument("--report", type=Path, default=Path("reports/daisee_group_cv_comparison"))
    parser.add_argument("--check_only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    args.folds, args.results, args.report = (p.resolve() for p in (args.folds, args.results, args.report))
    hashes = validate_folds(args.folds)
    import torch
    rows = []
    for fold in range(1, 6):
        directory = args.results / f"bilstm_attention_run{fold}"
        # Local user-created checkpoints only; these contain saved CLI arguments.
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        saved = checkpoint["args"]
        expected = {"bidirectional": True, "epochs": 30, "patience": 7,
                    "decision_metric": "bal_acc", "thr_min": 0.2, "thr_max": 0.9, "thr_steps": 15}
        if Path(saved["features_dir"]).resolve() != args.folds / f"fold_{fold}":
            raise ValueError(f"BiLSTM run {fold} used a different fold")
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError(f"BiLSTM run {fold} has incompatible settings")
        test = json.loads((directory / "metrics.json").read_text())["test"]
        with (args.folds / f"fold_{fold}" / "group_manifest.csv").open(newline="", encoding="utf-8") as handle:
            labels = Counter(r["label"] for r in csv.DictReader(handle) if r["split"] == "test")
        if test["tp"] + test["fn"] != labels["1"] or test["tn"] + test["fp"] != labels["0"]:
            raise ValueError(f"BiLSTM run {fold} test counts do not match the manifest")
        rows.extend(metric_rows(directory / "metrics.json", ["bilstm_attention"], fold))
    # Import every trainer before starting expensive work to catch missing dependencies.
    import importlib
    for module, _ in JOBS.values():
        importlib.import_module(module)
    print("Validated five frozen folds, five existing BiLSTM results and trainer imports.", flush=True)
    if args.check_only:
        return
    args.report.mkdir(parents=True, exist_ok=True)
    lock = args.report / "runner.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        write_report(args.report, rows)
        source_files = sorted((ROOT / "attentiontrack").glob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
        source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_files}
        for name, (module, models) in JOBS.items():
            for fold in range(1, 6):
                directory = args.results / f"{name}_run{fold}"
                manifest = args.folds / f"fold_{fold}" / "group_manifest.csv"
                if digest(manifest) != hashes[fold]:
                    raise ValueError("Manifest changed while training")
                command = [sys.executable, "-u", "-m", "scripts.run_group_cv", "--worker", module,
                           "--features_dir", str(manifest.parent), "--save_dir", str(directory),
                           "--decision_metric", "bal_acc", "--thr_min", "0.20", "--thr_max", "0.90", "--thr_steps", "15"]
                metadata = {"model": name, "fold": fold, "manifest_sha256": hashes[fold],
                            "seed": 42, "torch_threads": 4, "command": command, "source_sha256": source_hashes}
                provenance = directory / "group_cv_run.json"
                metrics = directory / ("baseline_metrics.json" if name == "baseline_tabular" else "metrics.json")
                if directory.exists():
                    if not provenance.is_file() or json.loads(provenance.read_text()) != metadata:
                        raise ValueError(f"Refusing to overwrite unverified or changed run: {directory}")
                else:
                    directory.mkdir(parents=True)
                    write_json(provenance, metadata)
                if not (directory / "completed.json").is_file():
                    write_json(args.report / "status.json", {"status": "running", "model": name, "fold": fold,
                               "pid": os.getpid(), "log": str(directory / "train.log")})
                    print(f"START {name} fold {fold}/5", flush=True)
                    env = dict(os.environ, PYTHONHASHSEED="42", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
                    with (directory / "train.log").open("a", encoding="utf-8") as log:
                        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                    metric_rows(metrics, models, fold)
                    write_json(directory / "completed.json", {"metrics_sha256": digest(metrics), "finished": time.time()})
                completion = json.loads((directory / "completed.json").read_text())
                if completion["metrics_sha256"] != digest(metrics):
                    raise ValueError(f"Completed metrics changed: {metrics}")
                rows.extend(metric_rows(metrics, models, fold))
                write_report(args.report, rows)
                print(f"DONE {name} fold {fold}/5", flush=True)
        write_json(args.report / "status.json", {"status": "complete", "records": len(rows)})
    except BaseException as exc:
        write_json(args.report / "status.json", {"status": "failed", "error": str(exc)})
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3:])
    else:
        main()
