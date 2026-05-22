import os
import re
import time
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import AutoModelForVision2Seq as AutoVisionModel
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoVisionModel

# ==== Model and device configuration ====
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = REPO_ROOT / "MGEMMV-Llama-11b"
INT8_SKIP_MODULES = ["model.multi_modal_projector"]


def apply_bitsandbytes_int8_patch():
    try:
        import bitsandbytes.autograd._functions as bnb_autograd_functions
        import bitsandbytes.functional as bnb_functional
        import bitsandbytes.libbitsandbytes_cpu as bnb_cuda_ops
    except Exception:
        return

    if getattr(bnb_functional.int8_vectorwise_quant, "_mgemmv_patched", False):
        return

    def safe_int8_vectorwise_quant(A, threshold=0.0):
        rows = A.shape[0]
        cols = A.shape[1]
        row_stats = torch.empty(rows, device=A.device, dtype=torch.float32)
        out_row = torch.empty(A.shape, device=A.device, dtype=torch.int8)
        outlier_cols = None

        A_row_major = A.contiguous()
        out_row_2d = out_row.view(rows, cols)

        bnb_cuda_ops.cint8_vector_quant(
            get_ptr(A_row_major),
            get_ptr(out_row_2d),
            get_ptr(row_stats),
            ct.c_float(threshold),
            ct.c_int32(rows),
            ct.c_int32(cols),
            bnb_cuda_ops._get_tensor_stream(A),
        )

        if rows > 1 and outlier_cols is not None and outlier_cols.numel() > 0:
            out_row_2d[:, outlier_cols] = 0

        return out_row, row_stats, outlier_cols

    from bitsandbytes.cextension import get_ptr
    import ctypes as ct

    safe_int8_vectorwise_quant._mgemmv_patched = True
    bnb_functional.int8_vectorwise_quant = safe_int8_vectorwise_quant
    bnb_autograd_functions.F.int8_vectorwise_quant = safe_int8_vectorwise_quant

processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
apply_bitsandbytes_int8_patch()
model = AutoVisionModel.from_pretrained(
    str(model_path),
    device_map="auto",
    dtype=torch.bfloat16,
    attn_implementation="eager",
    quantization_config=BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        llm_int8_skip_modules=INT8_SKIP_MODULES,
    ),
    trust_remote_code=True,
)
model.eval()

def clean_generated_code(text):
    """
    Extract the final Verilog module from model output and ensure it ends with endmodule.
    This removes echoed prompt text such as 'user' / 'assistant' headers.
    """
    code_block_match = re.search(
        r"```(?:verilog|Verilog|sv|systemverilog|SV)?\s*\n(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if code_block_match:
        text = code_block_match.group(1)

    module_matches = list(re.finditer(r"^\s*module\s+[A-Za-z_][A-Za-z0-9_$]*\b", text, re.MULTILINE))
    endmodule_matches = list(re.finditer(r"^\s*endmodule\b", text, re.MULTILINE))

    if module_matches:
        start = module_matches[-1].start()
        if endmodule_matches:
            end = endmodule_matches[-1].end()
            if end > start:
                return text[start:end].strip() + "\n"
        return text[start:].strip() + "\nendmodule\n"

    truncated = re.split(r"endmodule", text, maxsplit=1)[0]
    return truncated.strip() + "\nendmodule"

def load_image(image_path):
    """Load an image from disk and convert it to RGB format."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")

def generate_code(prompt, image_path):
    """
    Generate Verilog code based on the input prompt and image.

    Steps:
    1. Load the image from disk.
    2. Parse the prompt into text before and after the image tag.
    3. Format the input in chat style with image and text.
    4. Preprocess using processor and move tensors to the target device.
    5. Use the model to generate code tokens.
    6. Decode and clean the result.
    """
    image = load_image(image_path)
    before_text, after_text = split_prompt_at_image_tag(prompt)

    # Construct chat-style input
    messages = [{"role": "user", "content": []}]
    if before_text:
        messages[0]["content"].append({"type": "text", "text": before_text})
    messages[0]["content"].append({"type": "image"})
    if after_text:
        messages[0]["content"].append({"type": "text", "text": after_text})

    # Convert messages to input string for model
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Prepare model inputs (image and text) as tensors
    inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt").to(device)

    # Generate output token IDs using sampling
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=8192,
        temperature=0.8,
        top_p=0.9,
        do_sample=True
    )

    # Decode token IDs to readable text
    output_text = processor.decode(generate_ids[0], skip_special_tokens=True)
    return clean_generated_code(output_text)

def split_prompt_at_image_tag(prompt):
    """Split the prompt at <image> or <img> tag (case-insensitive)."""
    parts = re.split(r"<\s*image\s*>|<\s*img\s*>", prompt, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    else:
        return prompt.strip(), None

def process_files(base_path, num_modules_min, num_modules_max, num_tests):
    """
    Main processing loop:
    Traverse through all modules and test sets, read the description and image,
    call the model, and write the generated Verilog code.
    """
    for i in range(num_modules_min, num_modules_max + 1):
        module_name = f'module{i}'
        for j in range(1, num_tests + 1):
            test_name = f'test{j}'
            description_path = os.path.join(base_path, module_name, test_name, 'description')
            code_path = os.path.join(base_path, module_name, test_name, 'code')
            os.makedirs(code_path, exist_ok=True)

            file_index = 0
            while True:
                desc_file = os.path.join(description_path, f"description{file_index}.txt")
                image_name_file = os.path.join(description_path, f"image_name{file_index}.txt")

                if not os.path.exists(desc_file) or not os.path.exists(image_name_file):
                    break  # No more description-image pairs

                with open(desc_file, "r") as f:
                    prompt = f.read().strip()
                with open(image_name_file, "r") as f:
                    image_filename = f.read().strip()

                image_path = os.path.join(description_path, image_filename)
                if not os.path.exists(image_path):
                    print(f"[Skipping] Image not found: {image_filename}")
                    file_index += 1
                    continue

                print(f"[Processing] {module_name}/{test_name}/description{file_index}.txt + {image_filename}")

                try:
                    generated_code = generate_code(prompt, image_path)
                    cleaned_code = clean_generated_code(generated_code)

                    txt_out_path = os.path.join(code_path, f"llm_code{file_index}.txt")
                    verilog_out_path = os.path.join(code_path, f"llm_code{file_index}.v")

                    with open(txt_out_path, "w") as f:
                        f.write(generated_code)
                    with open(verilog_out_path, "w") as f:
                        f.write(cleaned_code)

                except Exception as e:
                    print(f"[Error] Failed to process {desc_file}: {e}")

                file_index += 1

            print(f"[Done] {module_name}/{test_name}")
        print(f"[Completed] All tests in {module_name}")

if __name__ == "__main__":
    base_directory = ".."
    num_modules_min = 1
    num_modules_max = 27
    num_tests = 5

    start_time = time.time()
    process_files(base_directory, num_modules_min, num_modules_max, num_tests)
    end_time = time.time()

    print(f"\n[Total Runtime] {end_time - start_time:.2f} seconds")
