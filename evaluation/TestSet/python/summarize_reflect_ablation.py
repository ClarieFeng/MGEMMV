import argparse
import csv
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def load_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def resolve_output_path(raw_arg: str, default_filename: str) -> Path:
    raw_arg = (raw_arg or "").strip()
    if not raw_arg:
        return (SCRIPT_DIR / default_filename).resolve()

    candidate = (SCRIPT_DIR / raw_arg).resolve()
    if candidate.exists() and candidate.is_dir():
        return candidate / default_filename
    if candidate.name == "":
        return candidate / default_filename
    if candidate.suffix == "":
        return candidate / default_filename
    return candidate


def build_mode_summary(rows):
    summary = defaultdict(lambda: {"total": 0, "syntax": 0, "functional": 0, "patched": 0})
    for row in rows:
        mode = row["mode"]
        syntax_ok = parse_bool(row["syntax_ok"])
        functional_ok = parse_bool(row["functional_ok"])
        summary[mode]["total"] += 1
        summary[mode]["syntax"] += int(syntax_ok)
        summary[mode]["functional"] += int(functional_ok)
        summary[mode]["patched"] += int(row.get("final_source", "") == "patched")
    return summary


def build_case_summary(rows):
    grouped = defaultdict(dict)
    for row in rows:
        key = (int(row["module_id"]), int(row["test_id"]))
        grouped[key][row["mode"]] = row

    case_rows = []
    mode_order = ["origin", "reflect_no_patch", "reflect_patch"]
    for (module_id, test_id) in sorted(grouped):
        entry = {"module_id": module_id, "test_id": test_id}
        for mode in mode_order:
            row = grouped[(module_id, test_id)].get(mode, {})
            entry[f"{mode}_syntax"] = row.get("syntax_ok", "")
            entry[f"{mode}_functional"] = row.get("functional_ok", "")
            entry[f"{mode}_source"] = row.get("final_source", "")
        case_rows.append(entry)
    return case_rows


def write_mode_summary(summary, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "total",
                "syntax",
                "functional",
                "syntax_rate",
                "functional_rate",
                "patched_successes",
            ],
        )
        writer.writeheader()
        for mode in ["origin", "reflect_no_patch", "reflect_patch"]:
            if mode not in summary:
                continue
            stats = summary[mode]
            total = stats["total"] or 1
            writer.writerow(
                {
                    "mode": mode,
                    "total": stats["total"],
                    "syntax": stats["syntax"],
                    "functional": stats["functional"],
                    "syntax_rate": f"{stats['syntax'] / total:.3f}",
                    "functional_rate": f"{stats['functional'] / total:.3f}",
                    "patched_successes": stats["patched"],
                }
            )


def write_case_summary(case_rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "module_id",
                "test_id",
                "origin_syntax",
                "origin_functional",
                "origin_source",
                "reflect_no_patch_syntax",
                "reflect_no_patch_functional",
                "reflect_no_patch_source",
                "reflect_patch_syntax",
                "reflect_patch_functional",
                "reflect_patch_source",
            ],
        )
        writer.writeheader()
        writer.writerows(case_rows)


def print_summary(summary, case_rows):
    print("Reflect Ablation Report")
    print("=======================")
    print("Mode Summary")
    print("------------")
    for mode in ["origin", "reflect_no_patch", "reflect_patch"]:
        if mode not in summary:
            continue
        stats = summary[mode]
        total = stats["total"] or 1
        print(
            f"{mode:<18} total={stats['total']} "
            f"syntax={stats['syntax']} ({stats['syntax'] / total:.1%}) "
            f"functional={stats['functional']} ({stats['functional'] / total:.1%}) "
            f"patched_successes={stats['patched']}"
        )

    print("\nCase Comparison")
    print("---------------")
    for row in case_rows:
        print(
            f"m{row['module_id']}/t{row['test_id']}: "
            f"O={row['origin_functional']} "
            f"R={row['reflect_no_patch_functional']} "
            f"RP={row['reflect_patch_functional']} "
            f"(src={row['reflect_patch_source'] or '-'})"
        )


def main():
    parser = argparse.ArgumentParser(description="Summarize reflect ablation CSV into report-ready tables.")
    parser.add_argument(
        "--input-csv",
        default="../reflect_ablation.csv",
        help="Path to reflect_ablation.csv relative to this script.",
    )
    parser.add_argument(
        "--mode-summary-csv",
        default="../reflect_ablation_mode_summary.csv",
        help="Output CSV for per-mode summary.",
    )
    parser.add_argument(
        "--case-summary-csv",
        default="../reflect_ablation_case_summary.csv",
        help="Output CSV for per-case comparison.",
    )
    args = parser.parse_args()

    input_csv = (SCRIPT_DIR / args.input_csv).resolve()
    mode_summary_csv = resolve_output_path(args.mode_summary_csv, "reflect_ablation_mode_summary.csv")
    case_summary_csv = resolve_output_path(args.case_summary_csv, "reflect_ablation_case_summary.csv")

    rows = load_rows(input_csv)
    mode_summary = build_mode_summary(rows)
    case_summary = build_case_summary(rows)

    write_mode_summary(mode_summary, mode_summary_csv)
    write_case_summary(case_summary, case_summary_csv)
    print_summary(mode_summary, case_summary)
    print(f"\nMode summary CSV saved to {mode_summary_csv}")
    print(f"Case summary CSV saved to {case_summary_csv}")


if __name__ == "__main__":
    main()
