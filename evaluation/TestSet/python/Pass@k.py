import json
import os
from collections import defaultdict
import numpy as np


MANIFEST_PATH = "../pass_at_k_samples.jsonl"
REPORT_PATH = "../pass_at_k_report.txt"
K_VALUES = (1, 5, 10)


def pass_at_k(n, c, k):
    if c == 0:
        return 0.0
    if n < k:
        return 1.0
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def load_manifest(path):
    grouped = defaultdict(list)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            grouped[record["problem_id"]].append(record)

    if not grouped:
        raise ValueError("Manifest is empty. Run full_llm_test.py first.")
    return grouped


def compute_metric(grouped, field, k):
    scores = []
    for records in grouped.values():
        n = len(records)
        c = sum(1 for r in records if r.get(field, False))
        scores.append(pass_at_k(n, c, k))
    return float(np.mean(scores)) if scores else 0.0


def summarize(grouped):
    total_problems = len(grouped)
    summary = {
        "total_problems": total_problems,
        "avg_samples_per_problem": sum(len(v) for v in grouped.values()) / total_problems,
        "syntax_pass_at_k": {},
        "compile_pass_at_k": {},
        "functional_pass_at_k": {},
    }

    for k in K_VALUES:
        summary["syntax_pass_at_k"][k] = compute_metric(grouped, "syntax_ok", k)
        summary["compile_pass_at_k"][k] = compute_metric(grouped, "compile_ok", k)
        summary["functional_pass_at_k"][k] = compute_metric(grouped, "functional_ok", k)
    return summary


def write_report(summary, path):
    lines = [
        f"Total problems: {summary['total_problems']}",
        f"Average samples per problem: {summary['avg_samples_per_problem']:.2f}",
    ]

    for k, value in summary["syntax_pass_at_k"].items():
        lines.append(f"Syntax Pass@{k}: {value:.4f}")
    for k, value in summary["compile_pass_at_k"].items():
        lines.append(f"Compilation Pass@{k}: {value:.4f}")
    for k, value in summary["functional_pass_at_k"].items():
        lines.append(f"Functional Pass@{k}: {value:.4f}")

    report = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report, end="")
    print(f"Report saved to {path}")


if __name__ == "__main__":
    grouped_records = load_manifest(MANIFEST_PATH)
    summary = summarize(grouped_records)
    write_report(summary, REPORT_PATH)
