import json
import re
import os
'''
windows

def clean_paramod_names(input_folder, output_folder):
    # Regex 1: handle $paramod\MyAdder\width=...
    pattern_normal = re.compile(r"\$paramod\\([^\\]+)\\[^\\]+")
    # Regex 2: handle $paramod$<hash>\CSA_3to2_18_18_18
    pattern_hash = re.compile(r"\$paramod\$[a-f0-9]+\\([^\\]+)")

    def replace_paramod(value):
        if isinstance(value, str):
            new_value = pattern_normal.sub(r"\1", value)
            new_value = pattern_hash.sub(r"\1", new_value)
            return new_value
        elif isinstance(value, dict):
            return {replace_paramod(k): replace_paramod(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [replace_paramod(v) for v in value]
        return value

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".json"):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, input_folder)  # relative path
                output_path = os.path.join(output_folder, rel_path)   # output file path

                # Skip if the target file already exists
                if os.path.exists(output_path):
                    print(f"Skip! File already exists: {output_path}")
                    continue

                try:
                    with open(input_path, 'r', encoding='utf-8') as f:
                        original_data = json.load(f)

                    cleaned_data = replace_paramod(original_data)

                    # Create intermediate directories in the output path if not exist
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(cleaned_data, f, indent=2)

                    print(f"Saved cleaned file to: {output_path}")
                except Exception as e:
                    print(f"Error! Failed to process {input_path} -- {str(e)}")

# Example usage
clean_paramod_names(".\\json", ".\\cleaned_json")
'''


def clean_paramod_names(input_folder, output_folder):
    # Regex 1: handle $paramod\MyAdder\width=...
    # 注意：在 Linux 路径中通常也是用 / 或者 \ 取决于 Yosys 生成时的设置，
    # 但 Yosys JSON 内部通常使用 \ 作为层级分隔符，无论操作系统如何。
    # 所以这里的正则保持 \ 是正确的，因为它是 JSON 内容的一部分，不是文件路径。
    pattern_normal = re.compile(r"\$paramod\\([^\\]+)\\[^\\]+")
    
    # Regex 2: handle $paramod$<hash>\CSA_3to2_18_18_18
    pattern_hash = re.compile(r"\$paramod\$[a-f0-9]+\\([^\\]+)")

    def replace_paramod(value):
        if isinstance(value, str):
            new_value = pattern_normal.sub(r"\1", value)
            new_value = pattern_hash.sub(r"\1", new_value)
            return new_value
        elif isinstance(value, dict):
            # 注意：修改 Key 需要谨慎。如果 Key 是标准 JSON 结构键（如 "modules"），不应被修改。
            # 目前的正则只匹配 $paramod 开头，所以相对安全。
            return {replace_paramod(k): replace_paramod(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [replace_paramod(v) for v in value]
        return value

    # 确保输入文件夹存在
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".json"):
                input_path = os.path.join(root, file)
                
                # 计算相对路径，以便在输出文件夹中保持相同的目录结构
                rel_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, rel_path)

                # Skip if the target file already exists
                if os.path.exists(output_path):
                    print(f"Skip! File already exists: {output_path}")
                    continue

                try:
                    with open(input_path, 'r', encoding='utf-8') as f:
                        original_data = json.load(f)

                    cleaned_data = replace_paramod(original_data)

                    # Create intermediate directories in the output path if not exist
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(cleaned_data, f, indent=2)

                    print(f"Saved cleaned file to: {output_path}")
                except Exception as e:
                    print(f"Error! Failed to process {input_path} -- {str(e)}")

if __name__ == "__main__":
    # 【修改点】将 "../" 改为 "./" ，确保与另外两个脚本的路径在同一层级
    input_dir = "./json" 
    output_dir = "./cleaned_json"
    
    print(f"Starting cleaning from {input_dir} to {output_dir}...")
    clean_paramod_names(input_dir, output_dir)
    print("Done.")