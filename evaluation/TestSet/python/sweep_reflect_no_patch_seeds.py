import argparse
import csv
import json
import time
from pathlib import Path

from origin_llm_test import SCRIPT_DIR
from single_agent_reflect import run_single_case


DEFAULT_CASES = [
    (6, 1),
    (7, 1),
    (24, 1),
]


def parse_cases(case_text: str):
    if not case_text.strip():
        return DEFAULT_CASES
    cases = []
    for item in case_text.split(","):
        module_id, test_id = item.strip().split(":")
        cases.append((int(module_id), int(test_id)))
    return cases


def parse_seeds(seed_text: str):
    return [int(item.strip()) for item in seed_text.split(",") if item.strip()]


def classify_history(history: list[dict]):
    if not history:
        return {
            "final_functional_ok": False,
            "final_source": "",
            "num_iterations": 0,
            "had_failure_before_success": False,
            "diagnosis_success": False,
            "first_preferred_action": "",
            "first_diagnosis_family": "",
            "first_confidence": "",
        }

    final = history[-1]
    final_ok = bool(final.get("functional_ok", False))
    final_source = final.get("functional_source", final.get("syntax_source", "raw"))
    prior_failures = [
        item
        for item in history[:-1]
        if not item.get("syntax_ok", False) or not item.get("functional_ok", True)
    ]
    diagnosis_entries = [
        item for item in history if item.get("preferred_action") or item.get("diagnosis_family")
    ]
    first_diag = diagnosis_entries[0] if diagnosis_entries else {}
    diagnosis_success = bool(
        final_ok
        and final_source == "raw"
        and prior_failures
        and first_diag.get("preferred_action")
    )

    return {
        "final_functional_ok": final_ok,
        "final_source": final_source,
        "num_iterations": len(history),
        "had_failure_before_success": bool(prior_failures) and final_ok,
        "diagnosis_success": diagnosis_success,
        "first_preferred_action": first_diag.get("preferred_action", ""),
        "first_diagnosis_family": first_diag.get("diagnosis_family", ""),
        "first_confidence": first_diag.get("diagnosis_confidence", ""),
    }


def write_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "module_id",
                "test_id",
                "seed",
                "final_functional_ok",
                "final_source",
                "num_iterations",
                "had_failure_before_success",
                "diagnosis_success",
                "first_preferred_action",
                "first_diagnosis_family",
                "first_confidence",
                "result_dir_name",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Sweep multiple seeds for reflect_no_patch on a small set of cases.")
    parser.add_argument("--base-directory", default="..", help="Dataset root relative to this script.")
    parser.add_argument(
        "--cases",
        default="",
        help="Comma-separated module:test list, e.g. '6:1,7:1,24:1'. Default uses a small targeted set.",
    )
    parser.add_argument(
        "--seeds",
        default="1001,1234,2024,4096",
        help="Comma-separated seed list. Default: 1001,1234,2024,4096",
    )
    parser.add_argument("--max-iters", type=int, default=3, help="Maximum iterations per case.")
    parser.add_argument(
        "--output-csv",
        default="../reflect_no_patch_seed_sweep.csv",
        help="Output CSV path. Default: ../reflect_no_patch_seed_sweep.csv",
    )
    args = parser.parse_args()

    base_path = (SCRIPT_DIR / args.base_directory).resolve()
    output_csv = (SCRIPT_DIR / args.output_csv).resolve()
    cases = parse_cases(args.cases)
    seeds = parse_seeds(args.seeds)

    rows = []
    start = time.time()
    for module_id, test_id in cases:
        for seed in seeds:
            result_dir_name = f"reflect_result_reflect_no_patch_sweep_seed{seed}_m{module_id}_t{test_id}"
            reflection_log = (SCRIPT_DIR / f"{result_dir_name}.jsonl").resolve()
            history = run_single_case(
                base_path,
                module_id,
                test_id,
                args.max_iters,
                rag_path=None,
                reflection_log=reflection_log,
                seed=seed,
                enable_patch=False,
                result_dir_name=result_dir_name,
            )
            summary = classify_history(history)
            rows.append(
                {
                    "module_id": module_id,
                    "test_id": test_id,
                    "seed": seed,
                    **summary,
                    "result_dir_name": result_dir_name,
                }
            )
            label = "diagnosis_success" if summary["diagnosis_success"] else ("success" if summary["final_functional_ok"] else "fail")
            print(
                f"m{module_id}/t{test_id} seed={seed} "
                f"status={label} source={summary['final_source']} "
                f"action={summary['first_preferred_action'] or '-'} "
                f"family={summary['first_diagnosis_family'] or '-'}"
            )

    write_csv(rows, output_csv)
    end = time.time()

    diagnosis_successes = sum(1 for row in rows if row["diagnosis_success"])
    total_successes = sum(1 for row in rows if row["final_functional_ok"])
    print("\nReflect No-Patch Seed Sweep")
    print("===========================")
    print(f"Cases: {len(cases)}  Seeds: {len(seeds)}  Runs: {len(rows)}")
    print(f"Successful runs: {total_successes}")
    print(f"Diagnosis-success runs: {diagnosis_successes}")
    print(f"CSV saved to {output_csv}")
    print(f"Elapsed: {end - start:.2f}s")


if __name__ == "__main__":
    main()
