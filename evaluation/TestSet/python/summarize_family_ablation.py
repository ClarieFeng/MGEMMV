import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

csv.field_size_limit(min(sys.maxsize, 10**7))


MODULE_FAMILY = {
    1: "adder_fa",
    2: "ppa_adder",
    3: "ppa_adder",
    4: "ppa_adder",
    5: "carry_skip_adder",
    6: "carry_select_adder",
    7: "cla_adder",
    8: "signed_adder_tree",
    9: "unsigned_adder_tree",
    10: "signed_adder_tree",
    11: "unsigned_adder_tree",
    12: "signed_booth_multiplier",
    13: "unsigned_booth_multiplier",
    14: "signed_multiplier",
    15: "unsigned_multiplier",
    16: "signed_booth_multiplier",
    17: "parallel_multiplier",
    18: "signed_multiplier",
    19: "serial_multiplier",
    20: "serial_multiplier",
    21: "systolic_dataflow",
    22: "winograd_transform",
    23: "winograd_transform",
    24: "signed_mac",
    25: "processing_element",
    26: "processing_element_row",
    27: "processing_element_array",
}


def parse_bool(value):
    return str(value).strip().lower() == "true"


def resolve_output_path(raw_arg: str, default_filename: str) -> Path:
    raw_arg = (raw_arg or "").strip()
    if not raw_arg:
        return (SCRIPT_DIR / default_filename).resolve()
    candidate = (SCRIPT_DIR / raw_arg).resolve()
    if candidate.exists() and candidate.is_dir():
        return candidate / default_filename
    if candidate.name == "" or candidate.suffix == "":
        return candidate / default_filename
    return candidate


def load_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_family_summary(rows):
    summary = defaultdict(lambda: defaultdict(lambda: {"total": 0, "syntax": 0, "functional": 0}))
    for row in rows:
        module_id = int(row["module_id"])
        family = MODULE_FAMILY.get(module_id, "unknown")
        mode = row["mode"]
        summary[family][mode]["total"] += 1
        summary[family][mode]["syntax"] += int(parse_bool(row["syntax_ok"]))
        summary[family][mode]["functional"] += int(parse_bool(row["functional_ok"]))
    return summary


def write_family_summary(summary, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family",
        "mode",
        "total",
        "syntax",
        "functional",
        "syntax_rate",
        "functional_rate",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for family in sorted(summary):
            for mode in ["origin", "reflect_no_patch", "reflect_patch"]:
                if mode not in summary[family]:
                    continue
                stats = summary[family][mode]
                total = stats["total"] or 1
                writer.writerow(
                    {
                        "family": family,
                        "mode": mode,
                        "total": stats["total"],
                        "syntax": stats["syntax"],
                        "functional": stats["functional"],
                        "syntax_rate": f"{stats['syntax'] / total:.3f}",
                        "functional_rate": f"{stats['functional'] / total:.3f}",
                    }
                )


def print_family_summary(summary):
    print("Family Breakdown")
    print("================")
    for family in sorted(summary):
        print(family)
        for mode in ["origin", "reflect_no_patch", "reflect_patch"]:
            if mode not in summary[family]:
                continue
            stats = summary[family][mode]
            total = stats["total"] or 1
            print(
                f"  {mode:<18} total={stats['total']} "
                f"syntax={stats['syntax']} ({stats['syntax'] / total:.1%}) "
                f"functional={stats['functional']} ({stats['functional'] / total:.1%})"
            )


def main():
    parser = argparse.ArgumentParser(description="Summarize ablation results by module family.")
    parser.add_argument(
        "--input-csv",
        default="../reflect_ablation_full_seed1234.csv",
        help="Path to reflect ablation CSV relative to this script.",
    )
    parser.add_argument(
        "--output-csv",
        default="../reflect_ablation_full_seed1234_family_summary.csv",
        help="Output CSV for family summary.",
    )
    args = parser.parse_args()

    input_csv = (SCRIPT_DIR / args.input_csv).resolve()
    output_csv = resolve_output_path(args.output_csv, "reflect_ablation_family_summary.csv")

    rows = load_rows(input_csv)
    summary = build_family_summary(rows)
    write_family_summary(summary, output_csv)
    print_family_summary(summary)
    print(f"\nFamily summary CSV saved to {output_csv}")


if __name__ == "__main__":
    main()
