import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

METRICS = ["acc", "prec", "rec", "f1", "spec", "bal_acc", "roc_auc", "pr_auc"]

LABELS = {
    "acc": "Accuracy",
    "prec": "Precision",
    "rec": "Recall",
    "f1": "F1",
    "spec": "Specificity",
    "bal_acc": "Balanced Acc.",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gru": "GRU",
    "cnn_gru": "CNN+GRU",
    "tcn": "TCN",
    "transformer_encoder": "Transformer Encoder",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "bilstm": "BiLSTM + Temporal Attention",
    "bilstm_attention": "BiLSTM + Temporal Attention",
    "bilstm_temporal_attention": "BiLSTM + Temporal Attention",
}

ORDER = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "LSTM",
    "GRU",
    "CNN+GRU",
    "TCN",
    "Transformer Encoder",
    "BiLSTM + Temporal Attention",
]

def infer_model_name(run_dir_name):
    base = re.sub(r"_run\d+$", "", run_dir_name)
    return MODEL_NAMES.get(base, base)

def safe_float(x):
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except (TypeError, ValueError, OverflowError):
        return None

def fmt_mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return "N/A (n=0)"
    if len(values) == 1:
        return f"{values[0]:.3f} (n=1; SD unavailable)"
    m = mean(values)
    s = stdev(values)
    return f"{m:.3f} ± {s:.3f}"

def sort_key(model):
    if model in ORDER:
        return (ORDER.index(model), model)
    return (999, model)

def collect_rows(root):
    """Keep experiment directories separate; folder names do not verify folds."""
    rows = []
    warnings = []
    paths = sorted(set(root.rglob("metrics.json")) | set(root.rglob("baseline_metrics.json")))
    seen = set()
    for metrics_path in paths:
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read {metrics_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object in {metrics_path}")

        if metrics_path.name == "baseline_metrics.json":
            results = data.get("results")
            if not isinstance(results, dict) or not results:
                raise ValueError(f"Missing baseline results in {metrics_path}")
        else:
            results = {re.sub(r"_run\d+$", "", metrics_path.parent.name): data}

        # Relative path preserves separate batches when --root is runs_full.
        experiment = metrics_path.parent.parent.relative_to(root).as_posix() if metrics_path.parent != root else "."
        run_name = metrics_path.parent.name
        for model_key, result in results.items():
            if not isinstance(result, dict) or not isinstance(result.get("test"), dict):
                raise ValueError(f"Missing test metrics for {model_key} in {metrics_path}")
            model = MODEL_NAMES.get(model_key, model_key)
            identity = (experiment, model, run_name)
            if identity in seen:
                raise ValueError(f"Duplicate model/run record: {identity} in {metrics_path}")
            seen.add(identity)
            if model_key not in MODEL_NAMES:
                warnings.append(f"Unknown model name '{model_key}' in {metrics_path}; kept unchanged.")
            row = {
                "experiment": experiment,
                "model": model,
                "run": run_name,
                "metrics_path": str(metrics_path),
            }
            for metric in METRICS:
                row[metric] = safe_float(result["test"].get(metric))
                if row[metric] is None:
                    warnings.append(f"Missing/non-finite {metric}: {metrics_path} ({model}).")
            rows.append(row)
    return rows, warnings


def summarize_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["experiment"], row["model"])].append(row)
    summaries = []
    for experiment, model in sorted(grouped, key=lambda key: (key[0], sort_key(key[1]))):
        items = grouped[(experiment, model)]
        summary = {"Experiment": experiment, "Model": model, "Records": len(items)}
        for metric in METRICS:
            values = [r[metric] for r in items if r[metric] is not None]
            summary[LABELS[metric]] = fmt_mean_std(values)
            summary[f"{LABELS[metric]} n"] = len(values)
        summaries.append(summary)
    return summaries


def markdown_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out_prefix", default="daisee_5runs")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing aggregate outputs.")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if not root.is_dir():
        parser.error(f"Results directory does not exist: {root}")

    try:
        rows, warnings = collect_rows(root)
    except ValueError as exc:
        parser.error(str(exc))

    if not rows:
        parser.error(f"No metrics.json or baseline_metrics.json files found under: {root}")

    raw_csv = Path(f"{args.out_prefix}_raw.csv")
    summary_csv = Path(f"{args.out_prefix}_summary.csv")
    summary_md = Path(f"{args.out_prefix}_summary.md")
    for output in (raw_csv, summary_csv, summary_md):
        if output.exists() and not args.overwrite:
            parser.error(f"Output already exists: {output}. Use a new prefix or --overwrite.")
    for output in (raw_csv, summary_csv, summary_md):
        output.parent.mkdir(parents=True, exist_ok=True)
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "model", "run", "metrics_path"] + METRICS)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = summarize_rows(rows)
    counts = defaultdict(set)
    for row in summary_rows:
        counts[row["Experiment"]].add(row["Records"])
    for experiment, sizes in counts.items():
        if len(sizes) > 1:
            warnings.append(f"Unequal record counts in {experiment}: {sorted(sizes)}.")
    fieldnames = ["Experiment", "Model", "Records"] + [col for m in METRICS for col in (LABELS[m], f"{LABELS[m]} n")]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "Descriptive mean and sample SD across available result records. Each metric has its own valid count (n).",
        "Records are not verified independent folds or distinct seeded runs. Participant separation, shared splits, and matching configurations are not established by this aggregation.",
        "Experiment directories are summarized separately; even within a directory, protocol compatibility must be checked before comparison.",
        "A displayed SD of 0.000 can reflect identical results or rounding; it is not evidence of generalization. A single record has no estimable SD.",
        "Model labels are inferred from directory names or baseline keys; architecture settings must be verified separately.",
        "",
    ]
    lines.append("| " + " | ".join(fieldnames) + " |")
    lines.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")
    for row in summary_rows:
        lines.append("| " + " | ".join(markdown_cell(row.get(col, "")) for col in fieldnames) + " |")
    if warnings:
        lines.extend(["", "Aggregation warnings:", ""])
        lines.extend(f"- {markdown_cell(warning)}" for warning in warnings)
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    print(f"Saved raw results: {raw_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved summary Markdown: {summary_md}")

if __name__ == "__main__":
    main()
