import argparse
import csv
import random
import tempfile
import time
from pathlib import Path

from origin_llm_test import SCRIPT_DIR, clean_generated_code, generate_code
from single_agent_reflect import load_case, run_single_case, syntax_check, functional_check

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None


DEFAULT_CASES = [
    (1, 1),
    (1, 2),
    (5, 1),
    (6, 1),
    (8, 1),
    (10, 1),
]

EXPANDED_CASES = [
    (1, 1),
    (1, 2),
    (5, 1),
    (5, 2),
    (6, 1),
    (6, 2),
    (7, 1),
    (7, 2),
    (8, 1),
    (8, 2),
    (9, 1),
    (10, 1),
    (11, 1),
    (13, 1),
    (22, 1),
    (24, 1),
]


def set_generation_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    if np is not None:
        np.random.seed(seed % (2**32))
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def parse_cases(case_text: str, preset: str):
    if not case_text.strip():
        return EXPANDED_CASES if preset == "expanded" else DEFAULT_CASES
    cases = []
    for item in case_text.split(","):
        module_id, test_id = item.strip().split(":")
        cases.append((int(module_id), int(test_id)))
    return cases


def mode_result_dir(mode: str, seed: int, module_id: int, test_id: int):
    return f"reflect_result_{mode}_seed{seed}_m{module_id}_t{test_id}"


def run_origin_case(base_path: Path, module_id: int, test_id: int, seed: int | None):
    case_dir, code_dir, prompt, image_path = load_case(base_path, module_id, test_id)
    set_generation_seed(seed)
    raw_output = generate_code(prompt, str(image_path))
    candidate = clean_generated_code(raw_output)
    syntax_ok, syntax_error = syntax_check(candidate, code_dir)
    functional_ok = False
    functional_detail = ""

    with tempfile.TemporaryDirectory(prefix="ablation_origin_") as temp_dir:
        candidate_path = Path(temp_dir) / "candidate.v"
        candidate_path.write_text(candidate, encoding="utf-8")
        if syntax_ok:
            functional_ok, functional_detail = functional_check(candidate_path, code_dir, case_dir / f"ablation_origin_result_seed{seed}")

    return {
        "mode": "origin",
        "module_id": module_id,
        "test_id": test_id,
        "syntax_ok": syntax_ok,
        "functional_ok": functional_ok,
        "final_source": "raw",
        "reflection_categories": "",
        "preferred_action": "",
        "diagnosis_family": "",
        "detail": functional_detail if functional_detail else syntax_error,
    }


def run_agent_case(base_path: Path, module_id: int, test_id: int, *, enable_patch: bool, seed: int | None):
    mode = "reflect_patch" if enable_patch else "reflect_no_patch"
    result_dir_name = mode_result_dir(mode, seed or 0, module_id, test_id)
    reflection_log = SCRIPT_DIR / f"{result_dir_name}.jsonl"
    history = run_single_case(
        base_path,
        module_id,
        test_id,
        max_iters=3,
        rag_path=None,
        reflection_log=reflection_log.resolve(),
        seed=seed,
        enable_patch=enable_patch,
        result_dir_name=result_dir_name,
    )
    last = history[-1] if history else {}
    detail = last.get("functional_detail") or last.get("syntax_error", "")
    return {
        "mode": mode,
        "module_id": module_id,
        "test_id": test_id,
        "syntax_ok": bool(last.get("syntax_ok", False)),
        "functional_ok": bool(last.get("functional_ok", False)),
        "final_source": last.get("functional_source", last.get("syntax_source", "raw")),
        "reflection_categories": "",
        "preferred_action": last.get("preferred_action", ""),
        "diagnosis_family": last.get("diagnosis_family", ""),
        "detail": detail,
    }


def write_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "module_id",
                "test_id",
                "syntax_ok",
                "functional_ok",
                "final_source",
                "reflection_categories",
                "preferred_action",
                "diagnosis_family",
                "detail",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = {}
    for row in rows:
        mode = row["mode"]
        summary.setdefault(mode, {"total": 0, "syntax": 0, "functional": 0})
        summary[mode]["total"] += 1
        summary[mode]["syntax"] += int(bool(row["syntax_ok"]))
        summary[mode]["functional"] += int(bool(row["functional_ok"]))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run a small ablation over origin / reflect-only / reflect+patch.")
    parser.add_argument("--base-directory", default="..", help="Dataset root relative to this script.")
    parser.add_argument(
        "--cases",
        default="",
        help="Comma-separated module:test list, e.g. '1:1,5:1,8:1'. Default uses a small preset.",
    )
    parser.add_argument(
        "--preset",
        choices=["small", "expanded"],
        default="small",
        help="Choose a built-in case preset when --cases is empty.",
    )
    parser.add_argument(
        "--output-csv",
        default="../reflect_ablation.csv",
        help="Output CSV path. Default: ../reflect_ablation.csv",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Base random seed for reproducible sampling.")
    args = parser.parse_args()

    base_path = (SCRIPT_DIR / args.base_directory).resolve()
    output_csv = (SCRIPT_DIR / args.output_csv).resolve()
    cases = parse_cases(args.cases, args.preset)

    rows = []
    start = time.time()
    for case_idx, (module_id, test_id) in enumerate(cases):
        case_seed = args.seed + case_idx * 100
        rows.append(run_origin_case(base_path, module_id, test_id, case_seed))
        rows.append(run_agent_case(base_path, module_id, test_id, enable_patch=False, seed=case_seed))
        rows.append(run_agent_case(base_path, module_id, test_id, enable_patch=True, seed=case_seed))
    end = time.time()

    write_csv(rows, output_csv)
    summary = summarize(rows)

    print("Reflect Ablation Summary")
    print("========================")
    for mode, stats in summary.items():
        print(
            f"{mode:<18} total={stats['total']} "
            f"syntax={stats['syntax']} functional={stats['functional']}"
        )
    print(f"\nCSV saved to {output_csv}")
    print(f"Elapsed: {end - start:.2f}s")


if __name__ == "__main__":
    main()
