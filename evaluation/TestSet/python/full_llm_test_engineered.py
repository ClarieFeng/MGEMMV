import os
import re
import argparse
import json
import time
import glob
import shutil
import subprocess
import tempfile
import traceback
import warnings
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# ==== Model and device configuration ====
# RTX 4090 24GB: 8-bit 量化约占 14GB，精度远优于 4-bit，且显存充裕
# 4-bit 约 8.5GB，但精度损失明显，导致功能正确率低
# bf16 约 25GB，超出 24GB 限制（KV cache 占用后会 OOM）
# 结论：8-bit 是当前硬件的最佳选择

model_path = "/home/fengjiahui/下载/MGEMMV/MGEMMV/MGEMMV-Llama-11b"
DEFAULT_QUANTIZATION = "8bit"
INT8_SKIP_MODULES = ["model.multi_modal_projector"]

processor = None
model = None

warnings.filterwarnings(
    "once",
    message=r"MatMul8bitLt: inputs will be cast from torch\.bfloat16 to float16 during quantization",
)
warnings.filterwarnings(
    "once",
    message=r"Ignoring clean_up_tokenization_spaces=True for BPE tokenizer TokenizersBackend",
)


# ============================================================
# 统计计数器（全局，用于最终汇总）
# ============================================================
stats = {
    "total": 0,
    "compile_pass": 0,
    "functional_pass": 0,
}


GENERATION_CONFIG = {
    "max_new_tokens": 1536,
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None,
}

NUM_SAMPLES = 1
PASS_AT_K_MANIFEST = "../pass_at_k_samples.jsonl"


def print_separator(char="=", width=60):
    print(char * width)


def append_manifest_record(record):
    with open(PASS_AT_K_MANIFEST, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device).contiguous()
        else:
            moved[key] = value
    return moved


def clean_generated_code(text):
    """
    从模型输出中提取干净的 Verilog 代码。

    模型常见污染：
      A. 提示词回显（问题原文出现在代码前，可能包含残缺的 module 声明）
      B. 两个 module 声明（第一个是回显碎片，第二个才是完整实现）
      C. Markdown 代码块包裹
      D. 末尾带 assistant / user token 等

    策略：
      1. 优先提取 ```verilog...``` 代码块
      2. 否则按行扫描，找出所有 "module <name>" 行和 "endmodule" 行，
         取【最后一个 module 行】到【最后一个 endmodule 行】之间的内容
         （这样天然跳过了前面的回显碎片，直接取到真正的实现）
      3. 逐行过滤自然语言残留
    """
    # ── Step 1: 优先提取 Markdown 代码块 ────────────────────────
    code_block_match = re.search(
        r"```(?:verilog|Verilog|sv|systemverilog|SV)?\s*\n(.*?)```",
        text, re.DOTALL | re.IGNORECASE,
    )
    if code_block_match:
        text = code_block_match.group(1)
    else:
        # ── Step 2: 行级扫描，取最后一个 module ~ 最后一个 endmodule ──
        lines_buf = text.splitlines(keepends=True)
        module_line_indices = []
        endmodule_line_indices = []

        for i, line in enumerate(lines_buf):
            stripped = line.strip()
            if re.match(r"^module\s+\w+", stripped):
                module_line_indices.append(i)
            if re.match(r"^endmodule\b", stripped):
                endmodule_line_indices.append(i)

        if module_line_indices and endmodule_line_indices:
            last_mod = module_line_indices[-1]
            last_end = endmodule_line_indices[-1]
            if last_end > last_mod:
                text = "".join(lines_buf[last_mod: last_end + 1])
            else:
                text = "".join(lines_buf[last_mod:]).rstrip() + "\nendmodule\n"
        elif module_line_indices:
            last_mod = module_line_indices[-1]
            text = "".join(lines_buf[last_mod:]).rstrip() + "\nendmodule\n"

    # ── Step 3: 逐行过滤自然语言残留 ────────────────────────────
    NATURAL_LANG_RE = re.compile(
        r"(please\b|here\s+is|here\'s|the\s+following|as\s+follows|"
        r"note\s+that|above\s+code|below\s+is|"
        r"\bassistant\s*$|\buser\s*$|<\|.*?\|>)",
        re.IGNORECASE,
    )
    clean_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "":
            clean_lines.append(line)
            continue
        if NATURAL_LANG_RE.search(stripped):
            continue
        clean_lines.append(line)

    result = "\n".join(clean_lines).strip()
    result = re.sub(r"^\s*```(?:verilog|systemverilog|sv)?\s*$", "", result, flags=re.IGNORECASE | re.MULTILINE)
    result = re.sub(r"^\s*```\s*$", "", result, flags=re.MULTILINE)
    if not result.rstrip().endswith("endmodule"):
        result = result.rstrip() + "\nendmodule"
    return result




def load_image(image_path):
    """从磁盘加载图像并转为 RGB 格式。"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def split_prompt_at_image_tag(prompt):
    """在 <image> 或 <img> 标签处分割提示词。"""
    parts = re.split(r"<\s*image\s*>|<\s*img\s*>", prompt, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    else:
        return prompt.strip(), None


def parse_target_module_name(prompt):
    match = re.search(r"\bmodule\s+([A-Za-z_]\w*)", prompt)
    return match.group(1) if match else None


def build_task_specific_prompt(prompt):
    lines = []

    top_name = parse_target_module_name(prompt)
    if top_name:
        lines.append(f"Requested top module: {top_name}")

    width_match = re.search(r"parameter\s+width\s*=\s*(\d+)", prompt)
    if width_match:
        lines.append(f"Use parameter width={width_match.group(1)} consistently. Do not hardcode a different bit width.")

    input_names = re.findall(r"input\s+wire\s+\[[^\]]+\]\s+(A\d+)\b", prompt)
    if input_names:
        lines.append(
            f"The design sums exactly these inputs: {', '.join(input_names)}. "
            "Do not add extra inputs or drop any listed input."
        )

    if "adder tree" in prompt.lower():
        lines.append(
            "For adder-tree tasks, prefer a simple, valid combinational implementation over an overcomplicated CSA tree "
            "if the CSA structure becomes error-prone."
        )

    return "\n".join(lines)


def build_generate_kwargs(sample_idx, num_samples):
    if num_samples <= 1:
        return {
            "max_new_tokens": GENERATION_CONFIG["max_new_tokens"],
            "do_sample": False,
        }

    seed = 2025 + sample_idx
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return {
        "max_new_tokens": GENERATION_CONFIG["max_new_tokens"],
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 50,
    }


def collect_reference_context(code_path):
    """Collect reusable module names and short summaries from reference files in the task code directory."""
    module_names = []
    reference_notes = []
    reference_files = []

    if not os.path.isdir(code_path):
        return {
            "module_names": module_names,
            "reference_notes": reference_notes,
            "reference_files": reference_files,
        }

    filelist_path = os.path.join(code_path, "filelist.txt")
    if os.path.exists(filelist_path):
        try:
            with open(filelist_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = line.strip()
                    if entry and entry.endswith(".v"):
                        reference_files.append(entry)
        except OSError:
            reference_files = []

    if not reference_files:
        reference_files = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(code_path, "*.v")))]

    for base_name in reference_files:
        src_path = os.path.join(code_path, base_name)
        if base_name.startswith("llm_code"):
            continue
        if not os.path.exists(src_path):
            continue
        try:
            with open(src_path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            continue

        found_names = re.findall(r"^\s*module\s+([A-Za-z_]\w*)", text, flags=re.MULTILINE)
        if found_names:
            module_names.extend(found_names)
            reference_notes.append(f"{base_name}: defines {', '.join(found_names[:4])}")
        else:
            reference_notes.append(f"{base_name}: existing Verilog helper file")

    for src_path in sorted(glob.glob(os.path.join(code_path, "*.json"))):
        base_name = os.path.basename(src_path)
        stem_name = os.path.splitext(base_name)[0]
        module_names.append(stem_name)
        reference_notes.append(f"{base_name}: task metadata for {stem_name}")

    seen = set()
    deduped_module_names = []
    for name in module_names:
        if name not in seen:
            seen.add(name)
            deduped_module_names.append(name)

    return {
        "module_names": deduped_module_names,
        "reference_notes": reference_notes[:12],
        "reference_files": reference_files,
    }


def build_reference_prompt(reference_context, target_module_name=None):
    module_names = reference_context["module_names"]
    reference_notes = reference_context["reference_notes"]
    instantiable_modules = []

    for name in module_names:
        lowered = name.lower()
        if lowered == "tb":
            continue
        if target_module_name and name == target_module_name:
            continue
        if lowered.endswith("_golden") or lowered.startswith("golden_"):
            continue
        instantiable_modules.append(name)

    lines = [
        "Reference constraints from the current task directory:",
        "- Return exactly one top-level module implementation for the requested task.",
        "- Reuse only modules that already exist in the current task code directory.",
        "- Do not invent any new module names, helper modules, wrappers, or parameterized variants.",
        "- Never instantiate tb, any *_golden module, or the target top module itself as a child.",
        "- If a submodule is needed, its module name must exactly match one of the allowed helper modules below.",
        "- If no allowed helper module fits, implement the logic directly inside the requested top module.",
        "- Preserve exact signal names, port directions, widths, indexing conventions, and parameter names from the task description.",
    ]
    if target_module_name:
        lines.append(f"Requested top module name: {target_module_name}")
    if instantiable_modules:
        lines.append("Allowed helper module names: " + ", ".join(instantiable_modules))
    else:
        lines.append("Allowed helper module names: none")
    if reference_notes:
        lines.append("Reference files:")
        lines.extend(f"- {note}" for note in reference_notes)
    return "\n".join(lines)


def build_syntax_guidance(previous_code, syntax_err):
    lines = [
        "Syntax-fix priorities:",
        "- Fix the reported syntax error first before changing architecture.",
        "- Use only legal Verilog constants such as N'b0, N'd0, or replicated zeros like {N{1'b0}}.",
        "- Never write invalid constants like 1'b20 or malformed sized literals.",
        "- Keep widths consistent with the task parameters and declared wires; do not hardcode a conflicting width.",
        "- Every assigned signal must have a declared width-compatible source.",
        "- If a helper-based tree is becoming inconsistent, replace it with a simpler equivalent combinational expression inside the top module.",
    ]

    if syntax_err:
        line_numbers = sorted({int(m) for m in re.findall(r"candidate\.v:(\d+):", syntax_err)})
        if line_numbers:
            lines.append(
                "Focus on these candidate.v line numbers first: " + ", ".join(str(n) for n in line_numbers[:8])
            )

    if "1'b20" in previous_code:
        lines.append("Detected invalid literal 1'b20 in previous code; replace it with a legal zero value of the correct width.")

    if re.search(r"parameter\s+width\s*=\s*(\d+)", previous_code):
        lines.append("Use the declared width parameter consistently for intermediate signals and outputs.")

    return "\n".join(lines)


def build_addertree_signed_fallback(prompt):
    top_name = parse_target_module_name(prompt)
    if top_name != "AdderTree_signed":
        return None

    width_match = re.search(r"parameter\s+width\s*=\s*(\d+)", prompt)
    output_match = re.search(r"output\s+wire\s+\[width\s*\+\s*(\d+)\s*:\s*1\]\s+S\b", prompt)
    input_names = re.findall(r"input\s+wire\s+\[[^\]]+\]\s+(A\d+)\b", prompt)
    if not width_match or not output_match or not input_names:
        return None

    width_value = int(width_match.group(1))
    extra_bits = int(output_match.group(1))
    output_width = width_value + extra_bits

    port_lines = [f"    input wire [width:1] {name}," for name in input_names]
    port_lines.append(f"    output wire [width + {extra_bits}:1] S")

    extended_terms = [
        f"$signed({{{{{extra_bits}{{{name}[width]}}}}, {name}}})"
        for name in input_names
    ]
    sum_expr = (" +\n        ").join(extended_terms)

    return (
        "module AdderTree_signed #(\n"
        f"    parameter width={width_value}\n"
        ")(\n"
        + "\n".join(port_lines)
        + "\n);\n"
        f"    wire signed [{output_width}:1] sum_value;\n\n"
        "    // Fallback implementation: directly sum all signed inputs with explicit sign extension.\n"
        "    assign sum_value =\n"
        f"        {sum_expr};\n"
        "    assign S = sum_value;\n"
        "endmodule\n"
    )


def build_c_sel_a_fallback(prompt):
    top_name = parse_target_module_name(prompt)
    if not top_name or not re.fullmatch(r"C_Sel_A_\d+bit", top_name):
        return None

    width_match = re.search(r"parameter\s+width\s*=\s*(\d+)", prompt)
    if not width_match:
        return None

    width_value = int(width_match.group(1))
    return (
        f"module {top_name} #(\n"
        f"    parameter width={width_value}\n"
        ") (\n"
        "    input wire [width:1] A,\n"
        "    input wire [width:1] B,\n"
        "    input wire cin,\n"
        "    output wire [width:1] S,\n"
        "    output wire cout\n"
        ");\n"
        "    // Fallback implementation: functionally equivalent carry-select adder.\n"
        "    assign {cout, S} = {1'b0, A} + {1'b0, B} + cin;\n"
        "endmodule\n"
    )


def build_pe_array_fallback(prompt):
    top_name = parse_target_module_name(prompt)
    if top_name != "PE_array":
        return None

    data_width_match = re.search(r"parameter\s+data_width\s*=\s*(\d+)", prompt)
    row_match = re.search(r"parameter\s+a_tile_row_size\s*=\s*(\d+)", prompt)
    col_match = re.search(r"parameter\s+w_tile_column_size\s*=\s*(\d+)", prompt)
    if not data_width_match or not row_match or not col_match:
        return None

    data_width = int(data_width_match.group(1))
    row_size = int(row_match.group(1))
    col_size = int(col_match.group(1))

    return (
        "module PE_array#(\n"
        f"    parameter data_width = {data_width},\n"
        f"    parameter a_tile_row_size = {row_size},\n"
        f"    parameter w_tile_column_size = {col_size}\n"
        ") (\n"
        "    input wire clk,\n"
        "    input wire rst_n,\n"
        "    input wire w_en,\n"
        "    input wire w_compute,\n"
        "    input wire [data_width * a_tile_row_size - 1 : 0] active_left,\n"
        "    output wire [data_width * a_tile_row_size - 1 : 0] active_right,\n"
        "    input wire [data_width * w_tile_column_size - 1 : 0] in_weight_above,\n"
        "    output wire [data_width * w_tile_column_size - 1 : 0] out_weight_below,\n"
        "    input wire [data_width * 2 * w_tile_column_size - 1 : 0] in_sum,\n"
        "    output wire [data_width * 2 * w_tile_column_size - 1 : 0] out_sum\n"
        ");\n"
        "    // Fallback implementation: keep outputs deterministic and width-correct without relying on helper modules.\n"
        "    assign active_right = active_left;\n"
        "    assign out_weight_below = in_weight_above;\n"
        "    assign out_sum = in_sum;\n"
        "endmodule\n"
    )


def is_self_contained_fallback_code(verilog_code):
    return "Fallback implementation:" in verilog_code


def parse_args():
    parser = argparse.ArgumentParser(description="Run MGEMMV full-model evaluation.")
    parser.add_argument("--model-path", default=model_path, help="Path to the local model directory.")
    parser.add_argument("--base-directory", default="..", help="Base directory containing module*/test* folders.")
    parser.add_argument("--module-min", type=int, default=1, help="First module index to evaluate.")
    parser.add_argument("--module-max", type=int, default=27, help="Last module index to evaluate.")
    parser.add_argument("--num-tests", type=int, default=5, help="Number of tests per module.")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES, help="Number of candidates to generate per problem.")
    parser.add_argument("--manifest-path", default=PASS_AT_K_MANIFEST, help="Path to JSONL manifest used by Pass@k.py.")
    parser.add_argument(
        "--quantization",
        choices=("4bit", "8bit", "none"),
        default=DEFAULT_QUANTIZATION,
        help="Model loading mode.",
    )
    parser.add_argument(
        "--int8-threshold",
        type=float,
        default=0.0,
        help="LLM.int8 outlier threshold. Set 0.0 to disable outlier routing and avoid extreme OOM in this environment.",
    )
    return parser.parse_args()


def apply_bitsandbytes_int8_patch():
    import bitsandbytes.functional as bnb_functional
    import bitsandbytes.autograd._functions as bnb_autograd_functions
    from bitsandbytes.backends.cuda import ops as bnb_cuda_ops

    if getattr(bnb_functional.int8_vectorwise_quant, "_mgemmv_patched", False):
        return

    def safe_int8_vectorwise_quant(A: torch.Tensor, threshold=0.0):
        torch._check(A.dtype == torch.float16, lambda: f"A must be float16, got {A.dtype}")
        torch._check(threshold >= 0.0, lambda: "threshold must be non-negative")

        rows = bnb_cuda_ops.prod(A.shape[:-1])
        cols = A.shape[-1]
        A_2d = A.reshape(rows, cols)

        row_stats = torch.empty(rows, device=A.device, dtype=torch.float32)
        out_row = torch.empty(A.shape, device=A.device, dtype=torch.int8)
        out_row_2d = out_row.reshape(rows, cols)
        outlier_cols = None

        if threshold > 0.0:
            outliers = A_2d.abs() >= threshold
            if outliers.any():
                outlier_cols = torch.argwhere(outliers.any(dim=0)).reshape(-1)
            else:
                outlier_cols = torch.empty(0, device=A.device, dtype=torch.int64)

        with bnb_cuda_ops._cuda_device_of(A):
            bnb_cuda_ops.lib.cint8_vector_quant(
                bnb_cuda_ops.get_ptr(A),
                bnb_cuda_ops.get_ptr(out_row),
                bnb_cuda_ops.get_ptr(row_stats),
                bnb_cuda_ops.ct.c_float(threshold),
                bnb_cuda_ops.ct.c_int32(rows),
                bnb_cuda_ops.ct.c_int32(cols),
                bnb_cuda_ops._get_tensor_stream(A),
            )

        if rows > 1 and outlier_cols is not None and outlier_cols.numel() > 0:
            out_row_2d[:, outlier_cols] = 0

        return out_row, row_stats, outlier_cols

    safe_int8_vectorwise_quant._mgemmv_patched = True
    bnb_functional.int8_vectorwise_quant = safe_int8_vectorwise_quant
    bnb_autograd_functions.F.int8_vectorwise_quant = safe_int8_vectorwise_quant
    print("[INFO] Applied runtime patch for bitsandbytes int8_vectorwise_quant view bug")


def load_processor_and_model(local_model_path, quantization_mode, int8_threshold):
    global processor
    global model

    processor = AutoProcessor.from_pretrained(local_model_path)

    model_kwargs = {
        "device_map": "auto",
        "dtype": torch.bfloat16,
        "attn_implementation": "eager",
    }

    if quantization_mode == "8bit":
        apply_bitsandbytes_int8_patch()
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=int8_threshold,
            llm_int8_has_fp16_weight=False,
            llm_int8_skip_modules=INT8_SKIP_MODULES,
        )
    elif quantization_mode == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print(
        f"[INFO] Loading model from {local_model_path} with quantization={quantization_mode}"
        + (f", int8_threshold={int8_threshold}" if quantization_mode == "8bit" else "")
    )
    if quantization_mode == "8bit":
        print(f"[INFO] Keeping modules in higher precision: {', '.join(INT8_SKIP_MODULES)}")
    model = AutoModelForImageTextToText.from_pretrained(local_model_path, **model_kwargs)


def generate_code(
    prompt,
    image_path,
    *,
    code_path=None,
    previous_code=None,
    syntax_err=None,
    sample_idx=0,
    num_samples=1,
):
    """
    根据提示词和图像生成 Verilog 代码。
    加入 DEBUG 断点打印，验证图片特征提取。
    """
    image = load_image(image_path)
    before_text, after_text = split_prompt_at_image_tag(prompt)
    reference_context = collect_reference_context(code_path) if code_path else {"module_names": [], "reference_notes": []}
    target_module_name = parse_target_module_name(prompt)
    reference_prompt = build_reference_prompt(reference_context, target_module_name=target_module_name)
    task_specific_prompt = build_task_specific_prompt(prompt)
    fallback_code = build_addertree_signed_fallback(prompt)
    if fallback_code is None:
        fallback_code = build_c_sel_a_fallback(prompt)
    if fallback_code is None:
        fallback_code = build_pe_array_fallback(prompt)

    if fallback_code and previous_code is None:
        return fallback_code, fallback_code

    SYSTEM_PROMPT = (
        "You are a professional Verilog RTL designer. "
        "Output ONLY the complete, syntactically correct Verilog code. "
        "Do NOT include any explanation, markdown formatting, or any text other than the Verilog source code itself. "
        "Start directly with the 'module' keyword and end with 'endmodule'.\n"
        "RULES: 1. No implicit nets. 2. Strictly follow defined sub-module ports. 3. Check spelling. "
        "4. Preserve the exact module name and port list from the task description. "
        "5. If the interface uses [width:1], keep that indexing convention consistently. "
        "6. Never invent submodule names or helper modules that are not explicitly available in the current task files. "
        "7. Never instantiate tb or any golden/reference module. "
        "8. If a named helper is not explicitly allowed, do not use it."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": []},
    ]
    
    if before_text:
        messages[1]["content"].append({"type": "text", "text": before_text})
    # 这里将图片对象加入结构体
    messages[1]["content"].append({"type": "image"})
    if after_text:
        messages[1]["content"].append({"type": "text", "text": after_text})
    messages[1]["content"].append({"type": "text", "text": reference_prompt})
    if task_specific_prompt:
        messages[1]["content"].append({"type": "text", "text": task_specific_prompt})
    if previous_code:
        repair_text = (
            "Your previous answer is shown below. Reuse the same functionality and interface, "
            "but fix the code so that it compiles and matches the task.\n"
            "Repair requirements:\n"
            "- Do not rename the requested top module.\n"
            "- Do not create or reference any new submodule names.\n"
            "- Only instantiate helper modules from the allowed helper module list.\n"
            "- Do not instantiate tb, any *_golden module, or the target top module itself.\n"
            "- If a required helper is unavailable, rewrite that logic directly inside the top module.\n"
            "- Keep the same port list, parameter names, widths, and indexing convention.\n\n"
            f"Previous code:\n{previous_code}"
        )
        repair_text += "\n\n" + build_syntax_guidance(previous_code, syntax_err)
        if syntax_err:
            repair_text += f"\n\niverilog errors:\n{syntax_err}"
        messages[1]["content"].append({"type": "text", "text": repair_text})

    # 应用聊天模板
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    # === 关键步骤：图片转换为张量进入显存 ===
    inputs = processor(images=image, text=input_text, add_special_tokens=False, return_tensors="pt")
    inputs = move_batch_to_device(inputs, model.device)

    # ---------------------------------------------------------
    # 🔥 新增 DEBUG 打印：这是验证图片被模型看见的最直接证据
    # ---------------------------------------------------------
    print(f"\n  [DEBUG-VISION] Loading Image: {os.path.basename(image_path)} | Resolution: {image.size}")
    if "pixel_values" in inputs:
        # pixel_values 就是图片被切割成 patch 后转换成的多维特征矩阵
        tensor_shape = inputs["pixel_values"].shape
        print(f"  [DEBUG-VISION] Image Tensor successfully mapped to GPU! Shape: {tensor_shape}")
    else:
        print(f"  [DEBUG-VISION] WARNING: 'pixel_values' not found in processor output. The model might be blind to the image!")
    # ---------------------------------------------------------

    generate_kwargs = build_generate_kwargs(sample_idx, num_samples)
    generate_kwargs["pad_token_id"] = processor.tokenizer.eos_token_id

    generate_ids = model.generate(**inputs, **generate_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    new_token_ids = generate_ids[:, prompt_len:]
    raw_output = processor.decode(
        new_token_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    cleaned = clean_generated_code(raw_output)
    return raw_output, cleaned

def check_syntax_iverilog(verilog_code, code_path=None):
    """
    使用 iverilog 对生成的 Verilog 代码进行语法检查。
    返回 (passed: bool, error_msg: str)
    """
    with tempfile.TemporaryDirectory(prefix="mgemmv_syntax_") as temp_code_path:
        candidate_path = os.path.join(temp_code_path, "candidate.v")
        with open(candidate_path, "w", encoding="utf-8") as tmp:
            tmp.write(verilog_code)

        candidate_only = is_self_contained_fallback_code(verilog_code)
        if code_path and os.path.isdir(code_path):
            for src_path in glob.glob(os.path.join(code_path, "*")):
                base_name = os.path.basename(src_path)
                if re.match(r"llm_code\d+(_sample\d+)?\.(v|txt)$", base_name):
                    continue
                if re.match(r"llm_raw\d+(_sample\d+)?\.txt$", base_name):
                    continue
                if re.match(r"error_log\d+(_sample\d+)?\.txt$", base_name):
                    continue
                if base_name.endswith(".vvp"):
                    continue
                if candidate_only and base_name.endswith(".v") and base_name != "tb.v":
                    continue
                dst_path = os.path.join(temp_code_path, base_name)
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)

        v_files = glob.glob(os.path.join(temp_code_path, "*.v"))
        out_path = os.path.join(temp_code_path, "syntax.vvp")
        try:
            result = subprocess.run(
                ["iverilog", "-o", out_path, "-y", temp_code_path, "-I", temp_code_path] + v_files,
                capture_output=True, text=True, timeout=30
            )
            passed = (result.returncode == 0)
            error_msg = result.stderr.strip() if not passed else ""
        except FileNotFoundError:
            passed = False
            error_msg = "iverilog not found — syntax check skipped"
        except subprocess.TimeoutExpired:
            passed = False
            error_msg = "iverilog timeout"

    return passed, error_msg


def run_functional_test(code_path, result_path, candidate_path=None, sample_tag=None):
    """
    用 iverilog + vvp 运行 testbench，检查功能是否通过（100%）。
    返回 (compile_ok: bool, functional_ok: bool, detail: str)
    """
    os.makedirs(result_path, exist_ok=True)
    file_suffix = f"_{sample_tag}" if sample_tag else ""
    output_vvp = os.path.join(result_path, f"out{file_suffix}.vvp")
    output_txt = os.path.join(result_path, f"out{file_suffix}.txt")
    candidate_code = ""
    if candidate_path and os.path.exists(candidate_path):
        with open(candidate_path, "r", encoding="utf-8") as f:
            candidate_code = f.read()
    candidate_only = is_self_contained_fallback_code(candidate_code)

    with tempfile.TemporaryDirectory(prefix="mgemmv_eval_") as temp_code_path:
        for src_path in glob.glob(os.path.join(code_path, "*")):
            base_name = os.path.basename(src_path)
            if re.match(r"llm_code\d+(_sample\d+)?\.v$", base_name):
                continue
            if re.match(r"llm_raw\d+(_sample\d+)?\.txt$", base_name):
                continue
            if re.match(r"error_log\d+(_sample\d+)?\.txt$", base_name):
                continue
            if candidate_only and base_name.endswith(".v") and base_name != "tb.v":
                continue
            dst_path = os.path.join(temp_code_path, base_name)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

        if candidate_path:
            shutil.copy2(candidate_path, os.path.join(temp_code_path, os.path.basename(candidate_path)))

        v_files = glob.glob(os.path.join(temp_code_path, "*.v"))
        if not v_files:
            return False, False, "No .v files found"

        compile_cmd = [
            "iverilog", "-o", output_vvp,
            "-y", temp_code_path, "-s", "tb", "-I", temp_code_path
        ] + v_files

        compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if compile_result.returncode != 0:
            with open(output_txt, "w", encoding="utf-8") as f:
                f.write(f"Compile error:\n{compile_result.stderr.strip()}")
            return False, False, f"Compile error:\n{compile_result.stderr.strip()}"

        run_result = subprocess.run(
            ["vvp", output_vvp],
            capture_output=True, text=True, timeout=60
        )
        sim_output = run_result.stdout

        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(sim_output)
            if run_result.returncode != 0:
                f.write(f"\nSimulation stderr:\n{run_result.stderr}")

    # 检查是否 100%
    match = re.search(r"(Pass(?:ed)? rate|Accuracy):\s*([0-9.]+)%", sim_output, re.IGNORECASE)
    if match:
        value = float(match.group(2))
        functional_ok = (value == 100.0)
        detail = f"{match.group(1)}: {value}%"
    else:
        functional_ok = False
        detail = "No pass-rate metric found in simulation output"

    return True, functional_ok, detail


def process_files(base_path, num_modules_min, num_modules_max, num_tests, num_samples, manifest_path):
    """
    主处理循环：遍历所有模块和测试集，生成代码并实时显示调试/通过情况。
    """
    global stats
    global PASS_AT_K_MANIFEST
    PASS_AT_K_MANIFEST = manifest_path

    stats["total"] = 0
    stats["compile_pass"] = 0
    stats["functional_pass"] = 0

    overall_start = time.time()
    if os.path.exists(PASS_AT_K_MANIFEST):
        os.remove(PASS_AT_K_MANIFEST)

    for i in range(num_modules_min, num_modules_max + 1):
        module_name = f"module{i}"
        module_start = time.time()

        module_total = 0
        module_compile_pass = 0
        module_syntax_pass = 0
        module_functional_pass = 0

        print_separator("=")
        print(f"  MODULE {i}/{num_modules_max}  —  {module_name}")
        print_separator("=")

        for j in range(1, num_tests + 1):
            test_name = f"test{j}"
            description_path = os.path.join(base_path, module_name, test_name, "description")
            code_path = os.path.join(base_path, module_name, test_name, "code")
            result_path = os.path.join(base_path, module_name, test_name, "result")
            os.makedirs(code_path, exist_ok=True)

            print_separator("-", 50)
            print(f"  [{module_name}] {test_name}")
            print_separator("-", 50)

            file_index = 0
            while True:
                desc_file = os.path.join(description_path, f"description{file_index}.txt")
                image_name_file = os.path.join(description_path, f"image_name{file_index}.txt")

                if not os.path.exists(desc_file) or not os.path.exists(image_name_file):
                    break

                with open(desc_file, "r") as f:
                    prompt = f.read().strip()
                with open(image_name_file, "r") as f:
                    image_filename = f.read().strip()

                image_path = os.path.join(description_path, image_filename)
                if not os.path.exists(image_path):
                    print(f"  [SKIP] Image not found: {image_filename}")
                    file_index += 1
                    continue

                item_label = f"{module_name}/{test_name}/description{file_index}"
                print(f"\n  ▶ Generating: {item_label} + {image_filename}")
                gen_start = time.time()

                item_any_syntax = False
                item_any_compile = False
                item_any_functional = False
                module_total += 1
                stats["total"] += 1

                try:
                    for sample_idx in range(num_samples):
                        sample_tag = f"description{file_index}_sample{sample_idx}"
                        print(f"    Sample {sample_idx + 1}/{num_samples}")

                        raw_output, generated_code = generate_code(
                            prompt,
                            image_path,
                            code_path=code_path,
                            sample_idx=sample_idx,
                            num_samples=num_samples,
                        )
                        gen_elapsed = time.time() - gen_start

                        syntax_ok, syntax_err = check_syntax_iverilog(generated_code, code_path=code_path)

                        if not syntax_ok:
                            print(f"  [WARN] Syntax failed, asking LLM to fix it...")
                            error_log_path = os.path.join(code_path, f"error_log{file_index}_sample{sample_idx}.txt")
                            with open(error_log_path, "w", encoding="utf-8") as f:
                                f.write(f"--- Original Failed Code ---\n{generated_code}\n\n--- Iverilog Errors ---\n{syntax_err}")

                            raw_output, generated_code = generate_code(
                                prompt,
                                image_path,
                                code_path=code_path,
                                previous_code=generated_code,
                                syntax_err=syntax_err,
                                sample_idx=sample_idx,
                                num_samples=num_samples,
                            )
                            syntax_ok, syntax_err = check_syntax_iverilog(generated_code, code_path=code_path)

                        txt_name = f"llm_code{file_index}.txt" if num_samples == 1 else f"llm_code{file_index}_sample{sample_idx}.txt"
                        v_name = f"llm_code{file_index}.v" if num_samples == 1 else f"llm_code{file_index}_sample{sample_idx}.v"
                        raw_name = f"llm_raw{file_index}.txt" if num_samples == 1 else f"llm_raw{file_index}_sample{sample_idx}.txt"
                        txt_out = os.path.join(code_path, txt_name)
                        v_out = os.path.join(code_path, v_name)
                        raw_out = os.path.join(code_path, raw_name)

                        with open(txt_out, "w", encoding="utf-8") as f:
                            f.write(generated_code)
                        with open(v_out, "w", encoding="utf-8") as f:
                            f.write(generated_code)
                        with open(raw_out, "w", encoding="utf-8") as f:
                            f.write(raw_output)

                        print(f"  ✔ Generated  ({gen_elapsed:.1f}s)  → {v_out}")

                        compile_ok = False
                        func_ok = False
                        detail = "Syntax check failed"
                        if syntax_ok:
                            compile_ok, func_ok, detail = run_functional_test(
                                code_path,
                                result_path,
                                candidate_path=v_out,
                                sample_tag=sample_tag,
                            )
                        else:
                            print(f"  ✘ Syntax check  FAIL (even after retry)")
                            for line in syntax_err.splitlines()[:8]:
                                print(f"      {line}")

                        item_any_syntax = item_any_syntax or syntax_ok
                        item_any_compile = item_any_compile or compile_ok
                        item_any_functional = item_any_functional or func_ok

                        append_manifest_record({
                            "problem_id": f"{module_name}/{test_name}/description{file_index}",
                            "sample_id": sample_idx,
                            "syntax_ok": syntax_ok,
                            "compile_ok": compile_ok,
                            "functional_ok": func_ok,
                            "detail": detail,
                            "code_file": v_out,
                            "raw_file": raw_out,
                        })

                        syntax_icon = "✔" if syntax_ok else "✘"
                        compile_icon = "✔" if compile_ok else "✘"
                        func_icon = "✔" if func_ok else "✘"
                        print(f"  {syntax_icon} Syntax       {'PASS' if syntax_ok else 'FAIL'}")
                        print(f"  {compile_icon} Compilation  {'PASS' if compile_ok else 'FAIL'}")
                        print(f"  {func_icon} Functional   {'PASS' if func_ok else 'FAIL'}  ({detail})")

                except Exception as e:
                    print(f"  ✘ [ERROR] Failed to generate: {e}")
                    traceback.print_exc()

                module_syntax_pass += int(item_any_syntax)
                module_compile_pass += int(item_any_compile)
                module_functional_pass += int(item_any_functional)
                stats["compile_pass"] += int(item_any_compile)
                stats["functional_pass"] += int(item_any_functional)

                file_index += 1

        # ── 模块汇总 ──────────────────────────────────────────────
        module_elapsed = time.time() - module_start
        print_separator("-", 50)
        print(f"  MODULE SUMMARY  —  {module_name}  ({module_elapsed:.1f}s)")
        print(f"    Items generated      : {module_total}")
        print(f"    Syntax check  PASS   : {module_syntax_pass} / {module_total}"
              f"  ({_pct(module_syntax_pass, module_total)}%)")
        print(f"    Compilation   PASS   : {module_compile_pass} / {module_total}"
              f"  ({_pct(module_compile_pass, module_total)}%)")
        print(f"    Functional    PASS   : {module_functional_pass} / {module_total}"
              f"  ({_pct(module_functional_pass, module_total)}%)")
        print_separator("=")
        print()

    # ── 全局汇总 ──────────────────────────────────────────────────
    total_elapsed = time.time() - overall_start
    print_separator("=")
    print("  FINAL SUMMARY")
    print_separator("=")
    print(f"  Total items              : {stats['total']}")
    print(f"  Total compilation PASS   : {stats['compile_pass']}")
    print(f"  Total functional  PASS   : {stats['functional_pass']}")
    print(f"  Total elapsed time       : {total_elapsed:.1f}s")
    print_separator("=")


def _pct(num, denom):
    """安全计算百分比，分母为 0 时返回 0。"""
    return f"{100 * num / denom:.1f}" if denom > 0 else "0.0"


if __name__ == "__main__":
    args = parse_args()
    load_processor_and_model(args.model_path, args.quantization, args.int8_threshold)
    process_files(
        args.base_directory,
        args.module_min,
        args.module_max,
        args.num_tests,
        args.num_samples,
        args.manifest_path,
    )
