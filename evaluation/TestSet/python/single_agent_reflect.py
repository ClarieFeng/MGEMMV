import argparse
import json
import random
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from origin_llm_test import (
    SCRIPT_DIR,
    clean_generated_code,
    device,
    load_image,
    model,
    processor,
)

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None


ERROR_DB = [
    {
        "category": "multi_driver_or_reg_assign",
        "patterns": [
            r"cannot be driven by primitives or continuous assignment",
            r"multiple drivers",
            r"Unable to bind wire/reg/memory",
        ],
        "reflection": [
            "Do not drive a reg with assign. Use always blocks for reg outputs, or change the port to wire.",
            "Ensure each signal has exactly one driver.",
        ],
    },
    {
        "category": "bad_instantiation",
        "patterns": [
            r"Unknown module type",
            r"Invalid module instantiation",
            r"Wrong number of ports",
            r"is not a port of",
        ],
        "reflection": [
            "Do not invent helper modules.",
            "Match helper module ports exactly and only instantiate modules that exist in current task files.",
        ],
    },
    {
        "category": "nonsynth_or_bad_always",
        "patterns": [
            r"syntax error",
            r"Malformed statement",
            r"invalid module item",
        ],
        "reflection": [
            "Keep synthesizable Verilog-2001 syntax only.",
            "Check always block structure, declarations, and generate-loop syntax.",
        ],
    },
    {
        "category": "width_or_slice",
        "patterns": [
            r"part select .* out of order",
            r"part select .* out of range",
            r"indefinite width",
            r"Concatenation operand .* has indefinite width",
        ],
        "reflection": [
            "Align vector widths exactly with the declared interface.",
            "Avoid mixing [width:1] and [width-1:0] conventions.",
        ],
    },
    {
        "category": "functional_x_propagation",
        "patterns": [
            r"\bx+\b",
            r"\bxxxxxxxx",
            r"Mismatch! Expected .* got .*x",
        ],
        "reflection": [
            "The candidate propagates unknown X values. Check for self-driven combinational chains or unassigned intermediate signals.",
            "For adder-style modules, prefer a direct arithmetic implementation over invented carry equations.",
        ],
    },
    {
        "category": "functional_wrong_adder_structure",
        "patterns": [
            r"Mismatch!",
            r"Pass rate:\s*0\.00%",
        ],
        "reflection": [
            "The structure is syntactically valid but functionally wrong. Rebuild the logic from the interface rather than editing a broken carry chain.",
            "If the module is an adder family, use a proven ripple-carry or direct addition template.",
        ],
    },
]


PATCH_LIBRARY = [
    {
        "name": "reg_assign_to_wire",
        "match": r"reg\s+([^\n;]*\b)(\w+)\s*;\s*(?:.|\n)*?assign\s+\2\s*=",
        "apply": None,
        "description": "If an output reg is driven only by assign, convert it to output wire.",
    },
    {
        "name": "strip_duplicate_endmodule_tail",
        "match": r"endmodule\s+endmodule",
        "apply": None,
        "description": "Collapse accidental duplicated endmodule tokens.",
    },
    {
        "name": "fa_direct_add_template",
        "match": r"adder_interface_two_inputs_with_cin",
        "apply": None,
        "description": "Replace a broken FA_Nbit implementation with a stable direct-add template.",
    },
    {
        "name": "addertree_signed_direct_sum",
        "match": r"signed_multi_input_sum_interface",
        "apply": None,
        "description": "Replace a broken signed adder tree with a direct signed sum template.",
    },
]


@dataclass
class RetrievalNote:
    category: str
    reflection: list[str]
    source: str


def extract_module_name(candidate: str):
    match = re.search(r"module\s+([A-Za-z_][A-Za-z0-9_$]*)\b", candidate)
    return match.group(1) if match else None


def infer_parameter_width(candidate: str, default: str = "17"):
    match = re.search(r"parameter\s+width\s*=\s*(\d+)", candidate)
    return match.group(1) if match else default


def detect_adder_interface(candidate: str):
    return all(
        pattern in candidate
        for pattern in [
            "input wire [width:1] A,",
            "input wire [width:1] B,",
            "input wire cin,",
            "output wire [width:1] S,",
            "output wire cout",
        ]
    )


def detect_cla_family_interface(candidate: str):
    return all(
        pattern in candidate
        for pattern in [
            "input wire [width:1] A,",
            "input wire [width:1] B,",
            "input wire cin,",
            "output wire [width:1] S,",
            "output wire cout,",
        ]
    ) and "p_1_" in candidate and "g_1_" in candidate


def detect_signed_multi_input_sum_interface(candidate: str):
    input_names = re.findall(r"input wire \[width:1\] (A\d+),", candidate)
    output_match = re.search(r"output wire \[width\s*\+\s*(\d+):1\] S", candidate)
    if not output_match:
        return None
    if len(input_names) < 4:
        return None
    if input_names != [f"A{i}" for i in range(1, len(input_names) + 1)]:
        return None
    return {
        "input_names": input_names,
        "extra_bits": output_match.group(1),
    }


def detect_unsigned_multi_input_sum_interface(candidate: str):
    sum_interface = detect_signed_multi_input_sum_interface(candidate)
    if not sum_interface:
        return None
    module_name = extract_module_name(candidate) or ""
    candidate_lower = candidate.lower()
    if "signed adder tree" in candidate_lower or "signed" in module_name.lower():
        return None
    return sum_interface


def detect_unsigned_multiplier_interface(candidate: str):
    return all(
        pattern in candidate
        for pattern in [
            "input wire [width:1] A,",
            "input wire [width:1] B,",
            "output wire [width*2:1] P",
        ]
    )


def detect_mac_signed_interface(candidate: str):
    normalized = re.sub(r"\s+", " ", candidate)
    return all(
        token in normalized
        for token in [
            "input wire signed [width:1] A",
            "input wire signed [width:1] B",
            "input wire signed [width*2:1] acc_in",
            "output wire signed [width*2+1:1] result",
        ]
    )


def split_prompt_at_image_tag(prompt):
    parts = re.split(r"<\s*image\s*>|<\s*img\s*>", prompt, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return prompt.strip(), None


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


def load_case(base_path: Path, module_id: int, test_id: int, file_index: int = 0):
    case_dir = base_path / f"module{module_id}" / f"test{test_id}"
    description_dir = case_dir / "description"
    code_dir = case_dir / "code"
    prompt = (description_dir / f"description{file_index}.txt").read_text(encoding="utf-8").strip()
    image_name = (description_dir / f"image_name{file_index}.txt").read_text(encoding="utf-8").strip()
    image_path = description_dir / image_name
    return case_dir, code_dir, prompt, image_path


def generate_code_with_limits(prompt: str, image_path: str, *, max_new_tokens: int = 1536):
    image = load_image(image_path)
    before_text, after_text = split_prompt_at_image_tag(prompt)

    messages = [{"role": "user", "content": []}]
    if before_text:
        messages[0]["content"].append({"type": "text", "text": before_text})
    messages[0]["content"].append({"type": "image"})
    if after_text:
        messages[0]["content"].append({"type": "text", "text": after_text})

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt").to(device)

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
    )
    output_text = processor.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text


def generate_text_only_with_limits(prompt: str, *, max_new_tokens: int = 384):
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=input_text, add_special_tokens=False, return_tensors="pt").to(device)

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
    )
    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = generate_ids[0][prompt_len:]
    output_text = processor.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text


def find_candidate_support_files(code_dir: Path):
    return [p for p in sorted(code_dir.glob("*.v")) if not p.name.startswith("llm_code") and p.name != "tb.v"]


def syntax_check(verilog_code: str, code_dir: Path):
    with tempfile.TemporaryDirectory(prefix="reflect_syntax_") as temp_dir:
        temp_dir = Path(temp_dir)
        candidate = temp_dir / "candidate.v"
        candidate.write_text(verilog_code, encoding="utf-8")

        support_files = []
        for file_path in find_candidate_support_files(code_dir):
            copied = temp_dir / file_path.name
            shutil.copy2(file_path, copied)
            support_files.append(copied)

        out_vvp = temp_dir / "syntax.vvp"
        cmd = ["iverilog", "-o", str(out_vvp), "-y", str(temp_dir), "-I", str(temp_dir), str(candidate)]
        cmd.extend(str(p) for p in support_files)
        result = subprocess.run(cmd, check=False, text=True, capture_output=True)
        return result.returncode == 0, (result.stderr or result.stdout).strip()


def functional_check(candidate_path: Path, code_dir: Path, result_dir: Path):
    result_dir.mkdir(parents=True, exist_ok=True)
    output_vvp = result_dir / "out.vvp"
    output_txt = result_dir / "out.txt"

    with tempfile.TemporaryDirectory(prefix="reflect_eval_") as temp_dir:
        temp_dir = Path(temp_dir)
        copied = []
        for file_path in sorted(code_dir.glob("*.v")):
            if file_path.name.startswith("llm_code") and file_path != candidate_path:
                continue
            dst = temp_dir / file_path.name
            shutil.copy2(file_path, dst)
            copied.append(dst)
        candidate_copy = temp_dir / candidate_path.name
        if not candidate_copy.exists():
            shutil.copy2(candidate_path, candidate_copy)
            copied.append(candidate_copy)

        cmd = ["iverilog", "-o", str(output_vvp), "-y", str(temp_dir), "-s", "tb", "-I", str(temp_dir)]
        cmd.extend(str(p) for p in copied)
        compile_result = subprocess.run(cmd, check=False, text=True, capture_output=True)
        if compile_result.returncode != 0:
            detail = f"Compile error:\n{compile_result.stderr}"
            output_txt.write_text(detail, encoding="utf-8")
            return False, detail

        sim_result = subprocess.run(["vvp", str(output_vvp)], check=False, text=True, capture_output=True)
        sim_output = (sim_result.stdout or "") + (sim_result.stderr or "")
        output_txt.write_text(sim_output, encoding="utf-8")
        match = re.search(r"(Pass(?:ed)? rate|Accuracy):\s*([0-9.]+)%", sim_output, re.IGNORECASE)
        return bool(match and float(match.group(2)) == 100.0), sim_output


def retrieve_error_notes(error_text: str, rag_path: Path | None):
    notes = []
    for entry in ERROR_DB:
        if any(re.search(pattern, error_text, re.IGNORECASE) for pattern in entry["patterns"]):
            notes.append(RetrievalNote(entry["category"], entry["reflection"], "builtin_error_db"))

    if rag_path and rag_path.exists():
        with rag_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                signature = record.get("error_signature", "")
                if signature and signature.lower() in error_text.lower():
                    notes.append(
                        RetrievalNote(
                            record.get("category", "retrieved_case"),
                            record.get("reflection", []),
                            str(rag_path),
                        )
                    )
    return notes


def detect_interface_family(candidate: str):
    if detect_mac_signed_interface(candidate):
        return "signed_mac"
    if detect_cla_family_interface(candidate):
        return "cla_family"
    if detect_adder_interface(candidate):
        return "two_input_adder"
    if detect_signed_multi_input_sum_interface(candidate):
        return "signed_multi_input_sum"
    if detect_unsigned_multi_input_sum_interface(candidate):
        return "unsigned_multi_input_sum"
    if detect_unsigned_multiplier_interface(candidate):
        return "unsigned_multiplier"
    return "unknown"


def summarize_failure(error_text: str, notes, candidate: str):
    error_lower = error_text.lower()
    family = detect_interface_family(candidate)
    stage = "functional"
    if any(
        token in error_lower
        for token in [
            "syntax error",
            "invalid module instantiation",
            "unknown module type",
            "wrong number of ports",
            "is not a port of",
            "error(s) during elaboration",
            "errors in port declarations",
            "i give up",
        ]
    ):
        stage = "syntax"

    categories = [note.category for note in notes] or ["unclassified"]
    goals = []
    avoid = []

    if family == "two_input_adder":
        goals.extend([
            "Implement the datapath as a single arithmetic addition that preserves the declared 1-based port widths.",
            "Ensure cout and S together equal A + B + cin for all inputs.",
        ])
        avoid.extend([
            "Do not invent intermediate helper adders or custom carry equations.",
            "Do not mix [width:1] and [width-1:0] indexing in internal logic.",
        ])
    elif family == "cla_family":
        goals.extend([
            "Preserve the CLA interface, but prioritize correct {cout,S} arithmetic behavior.",
            "Drive p/g summary outputs consistently with the repaired datapath.",
        ])
        avoid.extend([
            "Do not instantiate missing CLA sub-blocks unless they already exist in current task files.",
            "Do not change port names p_1_* or g_1_*.",
        ])
    elif family == "signed_multi_input_sum":
        goals.extend([
            "Treat every Ai input as signed and preserve sign during extension before summation.",
            "Make S exactly equal the signed sum of all inputs with the declared output width.",
        ])
        avoid.extend([
            "Do not reinterpret signed inputs as unsigned zero-extended values.",
            "Do not build an incomplete CSA/PPA tree if you cannot guarantee correctness.",
        ])
    elif family == "unsigned_multi_input_sum":
        goals.extend([
            "Treat every Ai input as unsigned and zero-extend before summation.",
            "Make S exactly equal the sum of all inputs with the declared output width.",
        ])
        avoid.extend([
            "Do not sign-extend unsigned inputs.",
            "Do not invent helper modules or partial tree nodes not present in task files.",
        ])
    elif family == "unsigned_multiplier":
        goals.extend([
            "Make P exactly equal A * B with the declared output width.",
            "Keep the implementation synthesizable and self-contained.",
        ])
        avoid.extend([
            "Do not instantiate signed Booth helpers with mismatched ports.",
            "Do not truncate the product width.",
        ])
    elif family == "signed_mac":
        goals.extend([
            "Make result exactly equal signed(A) * signed(B) + signed(acc_in).",
            "Preserve the full signed output width width*2+1.",
        ])
        avoid.extend([
            "Do not drop the top sign bit when extending product or acc_in.",
            "Do not mix unsigned multiplier outputs with signed accumulation without explicit signed casting.",
        ])
    else:
        goals.append("Repair the module while keeping the interface unchanged and making the testbench pass.")
        avoid.append("Do not invent helper modules that are not present in the current task files.")

    if stage == "syntax":
        goals.insert(0, "First make the code compile cleanly with iverilog.")
    else:
        goals.insert(0, "The code already compiles or nearly compiles; focus on functional correctness.")

    if "xxxxxxxx" in error_lower or "mismatch!" in error_lower:
        avoid.append("Do not leave signals partially assigned or with unknown X propagation.")

    return {
        "stage": stage,
        "family": family,
        "categories": categories,
        "goals": goals,
        "avoid": avoid,
    }


def truncate_for_reflection(previous_code: str, error_text: str, notes):
    code_tail = "\n".join(previous_code.splitlines()[-80:])
    err_tail = "\n".join(error_text.splitlines()[-40:])
    guidance = []
    for note in notes[:8]:
        guidance.append(f"[{note.category}] " + " ".join(note.reflection))
    return code_tail, err_tail, "\n".join(guidance)


def build_diagnosis_prompt(original_prompt: str, previous_code: str, error_text: str, notes):
    code_tail, err_tail, guidance_text = truncate_for_reflection(previous_code, error_text, notes)
    failure_summary = summarize_failure(error_text, notes, previous_code)
    if not guidance_text:
        guidance_text = "Fix the error with minimal changes and keep the interface unchanged."

    goals_text = "\n".join([f"- {item}" for item in failure_summary["goals"]])
    avoid_text = "\n".join([f"- {item}" for item in failure_summary["avoid"]])
    categories_text = ", ".join(failure_summary["categories"])

    return (
        original_prompt
        + "\n\nDiagnose the failed Verilog candidate before attempting any rewrite."
        + "\nDo not output Verilog code."
        + f"\nFailure stage: {failure_summary['stage']}"
        + f"\nDetected interface family: {failure_summary['family']}"
        + f"\nDetected error categories: {categories_text}\n"
        + "\nPrimary repair objectives:\n"
        + goals_text
        + "\n\nHard constraints:\n"
        + avoid_text
        + "\n\nIf helper modules are not guaranteed to exist, say that a self-contained implementation is preferred."
        + "\nReturn exactly these sections:\n"
        + "[Failure Stage]\n[Root Cause]\n[Interface Family]\n[Preferred Action]\n[Fallback Action]\n[Confidence]\n[Repair Plan]\n[Do Not]\n"
        + "\nAllowed Preferred Action values:\n"
        + "- regenerate\n"
        + "- regenerate_self_contained\n"
        + "- patch\n"
        + "- patch_template\n"
        + "\nWrite exactly one allowed value in [Preferred Action] and [Fallback Action]."
        + "\nPrevious candidate excerpt:\n"
        + code_tail
        + "\n\nCompiler / checker errors:\n"
        + err_tail
        + "\n\nReflection guidance:\n"
        + guidance_text
    )


def build_reflection_prompt(original_prompt: str, previous_code: str, error_text: str, notes, diagnosis_text: str):
    code_tail, err_tail, guidance_text = truncate_for_reflection(previous_code, error_text, notes)
    failure_summary = summarize_failure(error_text, notes, previous_code)
    if not guidance_text:
        guidance_text = "Fix the error with minimal changes and keep the interface unchanged."

    diagnosis_tail = "\n".join(diagnosis_text.splitlines()[-30:]).strip()
    if not diagnosis_tail:
        diagnosis_tail = "[Failure Stage]\nunknown\n[Root Cause]\nunknown\n[Interface Family]\nunknown\n[Repair Plan]\nProduce a self-contained repair.\n[Do Not]\nDo not change the interface."

    return (
        original_prompt
        + "\n\nYou are repairing a previously generated Verilog module."
        + "\nReuse the exact same module name and ports. Return only Verilog code."
        + f"\nFailure stage: {failure_summary['stage']}"
        + f"\nDetected interface family: {failure_summary['family']}"
        + "\nUse the structured diagnosis below as the repair specification."
        + "\nIf helper modules are not guaranteed to exist, prefer a self-contained synthesizable implementation."
        + "\nPreserve the declared 1-based indexing convention in ports and slices."
        + "\n\nStructured diagnosis:\n"
        + diagnosis_tail
        + "\n\nPrevious candidate excerpt:\n"
        + code_tail
        + "\n\nCompiler / checker errors:\n"
        + err_tail
        + "\n\nAdditional guidance:\n"
        + guidance_text
    )


def parse_diagnosis_text(diagnosis_text: str):
    fields = {}
    current = None
    for raw_line in diagnosis_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip().lower().replace(" ", "_")
            fields[current] = []
            continue
        if current:
            fields[current].append(line)

    normalized = {k: " ".join(v).strip() for k, v in fields.items()}
    preferred = normalized.get("preferred_action", "").lower()
    fallback = normalized.get("fallback_action", "").lower()
    confidence_text = normalized.get("confidence", "").lower()
    confidence = "medium"
    if "high" in confidence_text:
        confidence = "high"
    elif "low" in confidence_text:
        confidence = "low"

    preferred = normalize_action(preferred, diagnosis_text)
    fallback = normalize_action(fallback, diagnosis_text, default="patch")
    interface_family = normalized.get("interface_family", "")
    if preferred == "regenerate_self_contained" and interface_family in {
        "signed_multi_input_sum",
        "unsigned_multi_input_sum",
        "unsigned_multiplier",
        "signed_mac",
        "two_input_adder",
        "cla_family",
    }:
        preferred = "patch_template"
    if fallback == "regenerate_self_contained" and interface_family in {
        "signed_multi_input_sum",
        "unsigned_multi_input_sum",
        "unsigned_multiplier",
        "signed_mac",
        "two_input_adder",
        "cla_family",
    }:
        fallback = "patch_template"

    return {
        "preferred_action": preferred,
        "fallback_action": fallback,
        "confidence": confidence,
        "root_cause": normalized.get("root_cause", ""),
        "repair_plan": normalized.get("repair_plan", ""),
        "interface_family": interface_family,
    }


def normalize_action(action_text: str, full_text: str, default: str = "regenerate"):
    text = f"{action_text} {full_text}".lower()
    if "patch_template" in text or ("template" in text and "patch" in text):
        return "patch_template"
    if re.search(r"\bpatch\b", text):
        return "patch"
    if "self-contained" in text or "self contained" in text:
        return "regenerate_self_contained"
    if "regenerate" in text or "rewrite" in text or "rebuild" in text or "implement" in text:
        return "regenerate"
    return default


def should_apply_patch(diagnosis: dict):
    preferred = diagnosis.get("preferred_action", "")
    confidence = diagnosis.get("confidence", "medium")
    if preferred in {"patch", "patch_template"}:
        return True
    if preferred == "regenerate_self_contained" and confidence == "high":
        return True
    return False


def apply_patch_library(candidate: str, error_text: str):
    patched = candidate

    if "cannot be driven by primitives or continuous assignment" in error_text:
        patched = re.sub(r"\boutput\s+reg\b", "output wire", patched)

    patched = re.sub(r"endmodule\s+endmodule", "endmodule", patched)

    module_name = extract_module_name(patched) or "GeneratedModule"

    adder_error_markers = [
        "xxxxxxxx",
        "pass rate: 0.00%",
        "part selects straddling the start of signal",
        "out of range",
        "error(s) during elaboration",
        "indefinite width",
    ]
    if detect_adder_interface(patched) and any(marker in error_text.lower() for marker in adder_error_markers):
        width = infer_parameter_width(patched, default="17")
        patched = (
            f"module {module_name} #(\n"
            f"    parameter width={width}\n"
            f") (\n"
            f"    input wire [width:1] A,\n"
            f"    input wire [width:1] B,\n"
            f"    input wire cin,\n"
            f"    output wire [width:1] S,\n"
            f"    output wire cout\n"
            f");\n"
            f"    assign {{cout, S}} = A + B + cin;\n"
            f"endmodule\n"
        )

    if detect_cla_family_interface(patched) and any(marker in error_text.lower() for marker in adder_error_markers + ["errors in port declarations.", "i give up."]):
        width = infer_parameter_width(patched, default="17")
        p_match = re.search(r"output wire (p_1_\d+)", patched)
        g_match = re.search(r"output wire (g_1_\d+)", patched)
        p_name = p_match.group(1) if p_match else "p_1_width"
        g_name = g_match.group(1) if g_match else "g_1_width"
        patched = (
            f"module {module_name} #(\n"
            f"    parameter width={width}\n"
            f") (\n"
            f"    input wire [width:1] A,\n"
            f"    input wire [width:1] B,\n"
            f"    input wire cin,\n"
            f"    output wire [width:1] S,\n"
            f"    output wire cout,\n"
            f"    output wire {p_name},\n"
            f"    output wire {g_name}\n"
            f");\n"
            f"    assign {{cout, S}} = A + B + cin;\n"
            f"    assign {p_name} = &(A ^ B);\n"
            f"    assign {g_name} = cout;\n"
            f"endmodule\n"
        )

    sum_interface = detect_signed_multi_input_sum_interface(patched)
    if sum_interface and (
        "syntax error" in error_text.lower()
        or "invalid module instantiation" in error_text.lower()
        or "xxxxxxxx" in error_text.lower()
        or "pass rate: 0.00%" in error_text.lower()
    ):
        width = infer_parameter_width(patched, default="17")
        input_names = sum_interface["input_names"]
        extra_bits = sum_interface["extra_bits"]
        input_port_lines = "\n".join([f"    input wire [width:1] {name}," for name in input_names])
        signed_sum_lines = " +\n".join(
            [f"        $signed({{{{{extra_bits}{{{name}[width]}}}}, {name}}})" for name in input_names]
        )
        patched = (
            f"module {module_name} #(\n"
            f"    parameter width={width}\n"
            ")(\n"
            f"{input_port_lines}\n"
            f"    output wire [width + {extra_bits}:1] S\n"
            ");\n"
            f"    wire signed [width + {extra_bits}:1] sum_value;\n"
            "    assign sum_value =\n"
            f"{signed_sum_lines};\n"
            "    assign S = sum_value;\n"
            "endmodule\n"
        )

    unsigned_sum_interface = detect_unsigned_multi_input_sum_interface(patched)
    if unsigned_sum_interface and (
        "syntax error" in error_text.lower()
        or "invalid module instantiation" in error_text.lower()
        or "unknown module type" in error_text.lower()
        or "pass rate: 0.00%" in error_text.lower()
        or "xxxxxxxx" in error_text.lower()
    ):
        width = infer_parameter_width(patched, default="17")
        input_names = unsigned_sum_interface["input_names"]
        extra_bits = unsigned_sum_interface["extra_bits"]
        input_port_lines = "\n".join([f"    input wire [width:1] {name}," for name in input_names])
        sum_lines = " +\n".join([f"        {{{extra_bits}'b0, {name}}}" for name in input_names])
        patched = (
            f"module {module_name} #(\n"
            f"    parameter width={width}\n"
            ")(\n"
            f"{input_port_lines}\n"
            f"    output wire [width + {extra_bits}:1] S\n"
            ");\n"
            f"    wire [width + {extra_bits}:1] sum_value;\n"
            "    assign sum_value =\n"
            f"{sum_lines};\n"
            "    assign S = sum_value;\n"
            "endmodule\n"
        )

    if detect_unsigned_multiplier_interface(patched) and (
        "unknown module type" in error_text.lower()
        or "wrong number of ports" in error_text.lower()
        or "is not a port of" in error_text.lower()
        or "invalid module instantiation" in error_text.lower()
        or "pass rate: 0.00%" in error_text.lower()
        or "xxxxxxxx" in error_text.lower()
    ):
        width = infer_parameter_width(patched, default="6")
        patched = (
            f"module {module_name} #(\n"
            f"    parameter width={width}\n"
            f")(\n"
            f"    input wire [width:1] A,\n"
            f"    input wire [width:1] B,\n"
            f"    output wire [width*2:1] P\n"
            f");\n"
            f"    assign P = A * B;\n"
            f"endmodule\n"
        )

    if detect_mac_signed_interface(patched) and (
        "pass rate: 0.00%" in error_text.lower()
        or "xxxxxxxx" in error_text.lower()
        or "unknown module type" in error_text.lower()
        or "invalid module instantiation" in error_text.lower()
        or "is not a port of" in error_text.lower()
    ):
        width = infer_parameter_width(patched, default="22")
        patched = (
            f"module {module_name}#(\n"
            f"    parameter width={width}\n"
            "    )(\n"
            "    input wire signed [width:1] A,\n"
            "    input wire signed [width:1] B,\n"
            "    input wire signed [width*2:1] acc_in,\n"
            "    output wire signed [width*2+1:1] result\n"
            "    );\n"
            "    assign result = $signed(A) * $signed(B) + $signed(acc_in);\n"
            "endmodule\n"
        )

    return patched


def append_reflection_record(log_path: Path, record: dict):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def flatten_reflections(notes):
    return [item for note in notes for item in note.reflection]


def diagnose_failure(
    *,
    prompt: str,
    candidate: str,
    error_text: str,
    notes,
    result_dir: Path,
    iteration: int,
):
    diagnosis_prompt = build_diagnosis_prompt(prompt, candidate, error_text, notes)
    diagnosis_text = generate_text_only_with_limits(diagnosis_prompt, max_new_tokens=384)
    (result_dir / f"iter_{iteration}_diagnosis.txt").write_text(diagnosis_text, encoding="utf-8")
    return diagnosis_text, parse_diagnosis_text(diagnosis_text)


def run_single_case(
    base_path: Path,
    module_id: int,
    test_id: int,
    max_iters: int,
    rag_path: Path | None,
    reflection_log: Path,
    seed: int | None = None,
    enable_patch: bool = True,
    result_dir_name: str = "reflect_result",
):
    case_dir, code_dir, prompt, image_path = load_case(base_path, module_id, test_id)
    result_dir = case_dir / result_dir_name
    result_dir.mkdir(parents=True, exist_ok=True)

    raw_output = ""
    candidate = ""
    diagnosis_text = ""
    diagnosis = {}
    history = []
    syntax_error = ""
    compile_detail = ""
    notes = []

    for iteration in range(max_iters):
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        iter_seed = None if seed is None else seed + iteration
        set_generation_seed(iter_seed)

        if iteration == 0:
            raw_output = generate_code_with_limits(prompt, str(image_path))
        else:
            reflection_prompt = build_reflection_prompt(prompt, candidate, syntax_error or compile_detail, notes, diagnosis_text)
            raw_output = generate_code_with_limits(reflection_prompt, str(image_path), max_new_tokens=1024)

        candidate = clean_generated_code(raw_output)
        syntax_ok, syntax_error = syntax_check(candidate, code_dir)
        syntax_source = "raw"
        diagnosis_excerpt = ""
        preferred_action = ""
        diagnosis_confidence = ""
        diagnosis_family = ""
        patch_attempted = False
        if not syntax_ok:
            notes = retrieve_error_notes(syntax_error, rag_path)
            diagnosis_text, diagnosis = diagnose_failure(
                prompt=prompt,
                candidate=candidate,
                error_text=syntax_error,
                notes=notes,
                result_dir=result_dir,
                iteration=iteration,
            )
            diagnosis_excerpt = "\n".join(diagnosis_text.splitlines()[:12])
            preferred_action = diagnosis.get("preferred_action", "")
            diagnosis_confidence = diagnosis.get("confidence", "")
            diagnosis_family = diagnosis.get("interface_family", "") or detect_interface_family(candidate)
            if enable_patch and should_apply_patch(diagnosis):
                patched_candidate = apply_patch_library(candidate, syntax_error)
                patch_attempted = patched_candidate != candidate
                if patch_attempted:
                    candidate = clean_generated_code(patched_candidate)
                    syntax_ok, syntax_error = syntax_check(candidate, code_dir)
                    syntax_source = "patched"

        history.append(
            {
                "iteration": iteration,
                "diagnosis_excerpt": diagnosis_excerpt,
                "preferred_action": preferred_action,
                "diagnosis_confidence": diagnosis_confidence,
                "diagnosis_family": diagnosis_family,
                "patch_attempted": patch_attempted,
                "syntax_ok": syntax_ok,
                "syntax_error": syntax_error,
                "syntax_source": syntax_source,
            }
        )

        candidate_path = result_dir / f"iter_{iteration}.v"
        candidate_path.write_text(candidate, encoding="utf-8")
        (result_dir / f"iter_{iteration}.txt").write_text(raw_output, encoding="utf-8")

        if not syntax_ok:
            append_reflection_record(
                reflection_log,
                {
                    "module_id": module_id,
                    "test_id": test_id,
                    "iteration": iteration,
                    "stage": "syntax",
                    "event": "syntax_fail",
                    "error_signature": syntax_error.splitlines()[0] if syntax_error else "",
                    "category": notes[0].category if notes else "unclassified",
                    "reflection": flatten_reflections(notes),
                    "preferred_action": preferred_action,
                    "confidence": diagnosis_confidence,
                    "patch_attempted": patch_attempted,
                },
            )
            continue

        compile_ok, compile_detail = functional_check(candidate_path, code_dir, result_dir / f"iter_{iteration}")
        final_source = "raw"
        if not compile_ok:
            notes = retrieve_error_notes(compile_detail, rag_path)
            diagnosis_text, diagnosis = diagnose_failure(
                prompt=prompt,
                candidate=candidate,
                error_text=compile_detail,
                notes=notes,
                result_dir=result_dir,
                iteration=iteration,
            )
            diagnosis_excerpt = "\n".join(diagnosis_text.splitlines()[:12])
            preferred_action = diagnosis.get("preferred_action", "")
            diagnosis_confidence = diagnosis.get("confidence", "")
            diagnosis_family = diagnosis.get("interface_family", "") or detect_interface_family(candidate)
            patch_attempted = False
            if enable_patch and should_apply_patch(diagnosis):
                patched_candidate = apply_patch_library(candidate, compile_detail)
                patch_attempted = patched_candidate != candidate
                if patch_attempted:
                    candidate = clean_generated_code(patched_candidate)
                    candidate_path.write_text(candidate, encoding="utf-8")
                    compile_ok, compile_detail = functional_check(candidate_path, code_dir, result_dir / f"iter_{iteration}_patched")
                    final_source = "patched"
            history[-1]["diagnosis_excerpt"] = diagnosis_excerpt
            history[-1]["preferred_action"] = preferred_action
            history[-1]["diagnosis_confidence"] = diagnosis_confidence
            history[-1]["diagnosis_family"] = diagnosis_family
            history[-1]["patch_attempted"] = patch_attempted
        history[-1]["functional_ok"] = compile_ok
        history[-1]["functional_detail"] = compile_detail
        history[-1]["functional_source"] = final_source
        if compile_ok:
            success_notes = notes if patch_attempted or final_source == "patched" else []
            append_reflection_record(
                reflection_log,
                {
                    "module_id": module_id,
                    "test_id": test_id,
                    "iteration": iteration,
                    "stage": "functional",
                    "event": "patch_success" if final_source == "patched" else "direct_success",
                    "error_signature": compile_detail.splitlines()[0] if compile_detail else "",
                    "category": success_notes[0].category if success_notes else ("patched_success" if final_source == "patched" else "direct_success"),
                    "reflection": flatten_reflections(success_notes),
                    "source": final_source,
                    "syntax_source": history[-1]["syntax_source"],
                    "preferred_action": preferred_action,
                    "confidence": diagnosis_confidence,
                    "patch_attempted": patch_attempted,
                },
            )
            break
        append_reflection_record(
            reflection_log,
            {
                "module_id": module_id,
                "test_id": test_id,
                "iteration": iteration,
                "stage": "functional",
                "event": "functional_fail",
                "error_signature": compile_detail.splitlines()[0] if compile_detail else "",
                "category": notes[0].category if notes else "unclassified",
                "reflection": flatten_reflections(notes),
                "source": final_source,
                "syntax_source": history[-1]["syntax_source"],
                "preferred_action": preferred_action,
                "confidence": diagnosis_confidence,
                "patch_attempted": patch_attempted,
            },
        )

    (result_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return history


def main():
    parser = argparse.ArgumentParser(description="Single-agent iterative Verilog generation with error reflection.")
    parser.add_argument("--base-directory", default="..", help="Dataset root relative to this script.")
    parser.add_argument("--module-id", type=int, required=True, help="Target module id.")
    parser.add_argument("--test-id", type=int, required=True, help="Target test id.")
    parser.add_argument("--max-iters", type=int, default=3, help="Maximum reflection iterations.")
    parser.add_argument("--rag-path", default="", help="Optional JSONL reflection database path.")
    parser.add_argument(
        "--reflection-log",
        default="./error_reflection_db.jsonl",
        help="Where to append reflection samples. Default: ./error_reflection_db.jsonl",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Base random seed for deterministic sampling.")
    parser.add_argument("--no-patch", action="store_true", help="Disable patch execution and use regenerate-only routing.")
    parser.add_argument(
        "--result-dir-name",
        default="reflect_result",
        help="Per-case output directory name. Default: reflect_result",
    )
    args = parser.parse_args()

    base_path = (SCRIPT_DIR / args.base_directory).resolve()
    rag_path = Path(args.rag_path).resolve() if args.rag_path else None
    reflection_log = Path(args.reflection_log).resolve()

    start = time.time()
    history = run_single_case(
        base_path,
        args.module_id,
        args.test_id,
        args.max_iters,
        rag_path,
        reflection_log,
        seed=args.seed,
        enable_patch=not args.no_patch,
        result_dir_name=args.result_dir_name,
    )
    end = time.time()

    print(json.dumps(history, ensure_ascii=False, indent=2))
    print(f"\n[Total Runtime] {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
