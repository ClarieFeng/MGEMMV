import os
import re
import time
import glob
import shutil
import subprocess
import tempfile

from PIL import Image

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)

# ============================================================
# Model configuration
# ============================================================

model_path = "/home/fengjiahui/下载/MGEMMV/MGEMMV/MGEMMV-Llama-11b"
INT8_SKIP_MODULES = ["model.multi_modal_projector"]

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    llm_int8_skip_modules=INT8_SKIP_MODULES,
)

processor = AutoProcessor.from_pretrained(model_path)

model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto",
    dtype=torch.bfloat16,
)

model.eval()

# ============================================================
# Global statistics
# ============================================================

stats = {
    "total": 0,
    "syntax_pass": 0,
    "compile_pass": 0,
    "functional_pass": 0,
}

# ============================================================
# Utility functions
# ============================================================

def print_separator(char="=", width=70):
    print(char * width)


def safe_remove(path):
    if os.path.exists(path):
        try:
            os.remove(path)
        except:
            pass


def clean_code_directory(code_path):
    patterns = ["llm_code*.v", "llm_code*.txt", "llm_raw*.txt", "*.vvp", "out.txt"]
    for pattern in patterns:
        for f in glob.glob(os.path.join(code_path, pattern)):
            safe_remove(f)

# ============================================================
# Verilog cleaning
# ============================================================

def clean_generated_code(text):
    if not text:
        return ""

    code_block_match = re.search(r"```\s*(?:verilog|systemverilog|sv)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        text = code_block_match.group(1)

    module_match = re.search(r"\bmodule\b.*?\bendmodule\b", text, re.DOTALL | re.IGNORECASE)
    if module_match:
        text = module_match.group(0)
    else:
        lines = text.splitlines()
        module_indices = [i for i, l in enumerate(lines) if re.match(r"^module\s+\w+", l.strip())]
        endmodule_indices = [i for i, l in enumerate(lines) if re.match(r"^endmodule\b", l.strip())]
        if module_indices:
            start = module_indices[-1]
            end = endmodule_indices[-1] if endmodule_indices and endmodule_indices[-1] > start else len(lines) - 1
            text = "\n".join(lines[start:end+1])

    NATURAL_LANG_RE = re.compile(r"(please\b|here\s+is|here\'s|the\s+following|as\s+follows|note\s+that|below\s+is|\bassistant\s*$|\buser\s*$|<\|.*?\|>|^\s*```)", re.IGNORECASE)
    clean_lines = [line for line in text.splitlines() if not NATURAL_LANG_RE.search(line.strip()) or line.strip() == ""]
    result = "\n".join(clean_lines).strip()

    if result and not result.rstrip().endswith("endmodule"):
        result += "\nendmodule"

    return result.strip()

# ============================================================
# Image loading
# ============================================================

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def split_prompt_at_image_tag(prompt):
    parts = re.split(r"<\s*image\s*>|<\s*img\s*>", prompt, flags=re.IGNORECASE)
    return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (prompt.strip(), None)

# ============================================================
# Code generation
# ============================================================

def generate_code(prompt, image_path):
    image = load_image(image_path)
    before_text, after_text = split_prompt_at_image_tag(prompt)

    SYSTEM_PROMPT = (
        "You are a professional Verilog RTL designer.\n"
        "Output ONLY syntactically correct Verilog code.\n"
        "Do NOT output markdown.\n"
        "Do NOT output explanation.\n"
        "Do NOT output comments outside Verilog.\n"
        "Start with 'module'.\n"
        "End with 'endmodule'."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": []}]

    if before_text:
        messages[1]["content"].append({"type": "text", "text": before_text})

    messages[1]["content"].append({"type": "image"})

    if after_text:
        messages[1]["content"].append({"type": "text", "text": after_text})

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(images=image, text=input_text, add_special_tokens=False, return_tensors="pt")
    # 强制 reshape 避免 stride 错误
    inputs = {k: (v.to(model.device).reshape(v.size()) if torch.is_tensor(v) else v) for k,v in inputs.items()}

    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    raw_output = processor.decode(generate_ids[0], skip_special_tokens=True)
    cleaned_code = clean_generated_code(raw_output)
    return raw_output, cleaned_code

# ============================================================
# Syntax check
# ============================================================

def check_syntax_iverilog(verilog_code):
    with tempfile.NamedTemporaryFile(suffix=".v", mode="w", delete=False) as tmp:
        tmp.write(verilog_code)
        tmp_path = tmp.name

    out_path = tmp_path.replace(".v", ".vvp")
    try:
        result = subprocess.run(["iverilog", "-o", out_path, tmp_path], capture_output=True, text=True, timeout=30)
        passed = (result.returncode == 0)
        err = result.stderr.strip()
    except FileNotFoundError:
        passed = False
        err = "iverilog not installed"
    except subprocess.TimeoutExpired:
        passed = False
        err = "iverilog timeout"
    finally:
        safe_remove(tmp_path)
        safe_remove(out_path)
    return passed, err

# ============================================================
# Functional simulation
# ============================================================

def run_functional_test(code_path, result_path):
    os.makedirs(result_path, exist_ok=True)
    output_vvp = os.path.join(result_path, "out.vvp")
    output_txt = os.path.join(result_path, "out.txt")
    v_files = glob.glob(os.path.join(code_path, "*.v"))

    if not v_files:
        return False, False, "No .v files found"

    compile_cmd = ["iverilog", "-o", output_vvp, "-y", code_path, "-s", "tb", "-I", code_path] + v_files
    compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)

    if compile_result.returncode != 0:
        return False, False, compile_result.stderr.strip()

    run_result = subprocess.run(["vvp", output_vvp], capture_output=True, text=True, timeout=60)
    sim_output = run_result.stdout

    with open(output_txt, "w") as f:
        f.write(sim_output)

    match = re.search(r"(Pass(?:ed)? rate|Accuracy):\s*([0-9.]+)%", sim_output, re.IGNORECASE)
    if match:
        value = float(match.group(2))
        functional_ok = (value == 100.0)
        detail = f"{match.group(1)}: {value}%"
    else:
        functional_ok = False
        detail = "No pass-rate metric found"

    return True, functional_ok, detail

# ============================================================
# Main process loop
# ============================================================

def process_files(base_path, num_modules_min, num_modules_max, num_tests):
    global stats
    overall_start = time.time()

    for i in range(num_modules_min, num_modules_max + 1):
        module_name = f"module{i}"
        print_separator("=")
        print(f"MODULE: {module_name}")
        print_separator("=")

        module_total = module_syntax_pass = module_compile_pass = module_functional_pass = 0

        for j in range(1, num_tests + 1):
            test_name = f"test{j}"
            print_separator("-")
            print(f"{module_name} / {test_name}")
            print_separator("-")

            description_path = os.path.join(base_path, module_name, test_name, "description")
            code_path = os.path.join(base_path, module_name, test_name, "code")
            result_path = os.path.join(base_path, module_name, test_name, "result")
            os.makedirs(code_path, exist_ok=True)

            clean_code_directory(code_path)
            generated_anything = False
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
                print(f"\nGenerating: {module_name}/{test_name}/{file_index}")

                try:
                    raw_output, generated_code = generate_code(prompt, image_path)
                    if not generated_code.strip():
                        print("✘ Empty generated code")
                        file_index += 1
                        continue

                    txt_out = os.path.join(code_path, f"llm_code{file_index}.txt")
                    v_out = os.path.join(code_path, f"llm_code{file_index}.v")
                    raw_out = os.path.join(code_path, f"llm_raw{file_index}.txt")

                    with open(txt_out, "w") as f: f.write(generated_code)
                    with open(v_out, "w") as f: f.write(generated_code)
                    with open(raw_out, "w") as f: f.write(raw_output)

                    generated_anything = True
                    print(f"✔ Saved: {v_out}")

                    syntax_ok, syntax_err = check_syntax_iverilog(generated_code)
                    module_total += 1
                    stats["total"] += 1

                    if syntax_ok:
                        module_syntax_pass += 1
                        stats["syntax_pass"] += 1
                        print("✔ Syntax PASS")
                    else:
                        print("✘ Syntax FAIL")
                        print_separator("!")
                        print(syntax_err)
                        print_separator("!")

                except Exception as e:
                    print(f"✘ Generation FAILED: {e}")

                file_index += 1

            if not generated_anything:
                print("⚠ Skip simulation (nothing generated)")
                continue

            tb_file = os.path.join(code_path, "tb.v")
            if not os.path.exists(tb_file):
                print("⚠ No tb.v found")
                continue

            print("\nRunning simulation...")
            try:
                compile_ok, functional_ok, detail = run_functional_test(code_path, result_path)

                if compile_ok:
                    module_compile_pass += 1
                    stats["compile_pass"] += 1
                    print("✔ Compilation PASS")
                else:
                    print("✘ Compilation FAIL")

                if functional_ok:
                    module_functional_pass += 1
                    stats["functional_pass"] += 1
                    print(f"✔ Functional PASS ({detail})")
                else:
                    print(f"✘ Functional FAIL ({detail})")

            except Exception as e:
                print(f"✘ Simulation FAILED: {e}")

        # Module summary
        print_separator("=")
        print(f"SUMMARY: {module_name}")
        print(f"Generated Items : {module_total}")
        print(f"Syntax PASS     : {module_syntax_pass}/{module_total}")
        print(f"Compile PASS    : {module_compile_pass}/{num_tests}")
        print(f"Functional PASS : {module_functional_pass}/{num_tests}")
        print_separator("=")
        print()

    elapsed = time.time() - overall_start
    print_separator("=")
    print("FINAL SUMMARY")
    print_separator("=")
    print(f"Total Generated     : {stats['total']}")
    print(f"Syntax PASS         : {stats['syntax_pass']}")
    print(f"Compilation PASS    : {stats['compile_pass']}")
    print(f"Functional PASS     : {stats['functional_pass']}")
    print(f"Elapsed Time        : {elapsed:.1f}s")
    print_separator("=")

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    base_directory = ".."
    num_modules_min = 1
    num_modules_max = 27
    num_tests = 5

    process_files(base_directory, num_modules_min, num_modules_max, num_tests)
