import os
import shutil

def generate_yosys_script(my_toplevel, json_name, address):
    # 将路径中的反斜杠替换为正斜杠，确保在 Ubuntu/Linux 下路径规范
    linux_address = address.replace("\\", "/")
    json_path = os.path.join('.', 'json', f'{json_name}.json').replace("\\", "/")
    
    script_content = f"""
#**************************************************/
#* Compile Script for yosys                       */
#* */
#* yosys -s sys.ys                               */
#**************************************************/

# 使用标准的 read_verilog 命令通过通配符直接读取代码
read_verilog {linux_address}/code/*.v

hierarchy -check -top {my_toplevel}
#synth -top {my_toplevel}
proc
#opt
#fsm
#opt
#memory
#opt
write_json {json_path}
"""
    # Write the script to a file
    file_path = os.path.join(address, 'sys.ys')
    with open(file_path, 'w') as script_file:
        script_file.write(script_content)

def create_verilog_filelist(directory):
    """
    Scans the given directory for all .v and .sv files and writes their relative paths into filelist.txt.
    
    Parameters:
    directory (str): The directory to search for Verilog files.
    """
    scan_path = os.path.join(directory, 'code')
    filelist_path = os.path.join(directory, 'code', "filelist.txt")
    # Collect all .v and .sv files
    verilog_files = []
    for root, _, files in os.walk(scan_path):
        for file in files:
            if file.endswith(".v") or file.endswith(".sv"):
                rel_path = os.path.relpath(os.path.join(root, file), scan_path)
                verilog_files.append(rel_path)
    
    # Write to filelist.txt
    with open(filelist_path, "w", encoding="utf-8") as f:
        f.write("\n".join(verilog_files) + "\n")
    
    print(f"File list created at: {filelist_path}")
    
# Example usage
# create_file_list("./code")    
def create_prompt_file(name, directory):
    """
    Creates a prompt.txt file in the specified directory with a Verilog design request.
    增加提示词信息密度，并强制插入 <image> 标签，确保视觉与文本对齐。
    """
    os.makedirs(directory, exist_ok=True)
    
    # ======== 1. 扩充的高密度提示词 ========
    prompt_content = (
        f"You are a Senior IC Design Engineer. Your task is to implement the exact RTL code for the module named '{name}'.\n\n"
        f"Below is the netlist circuit schematic for this module:\n"
        f"<image>\n\n"
        f"CRITICAL INSTRUCTIONS:\n"
        f"1. VISUAL ANALYSIS: Carefully trace the logic gates, wires, and sub-modules shown in the schematic above.\n"
        f"2. EXACT MATCH: Do not guess or invent logic. The Verilog code must strictly match the visual structure.\n"
        f"3. DECLARE EVERYTHING: All internal connecting wires MUST be declared explicitly (e.g., `wire internal_net;`) before being used in assignments or instantiations.\n"
    )
    
    # ======== 2. 动态注入底层接口上下文 ========
    interface_injection = ""
    name_lower = name.lower()
    
    if "cla" in name_lower or "addertree" in name_lower:
        interface_injection = (
            "\nATTENTION: This design contains the 'CLA_20bit' sub-module. "
            "You MUST use this exact interface, no other ports exist:\n"
            "module CLA_20bit (\n"
            "    input wire [19:0] A,\n"
            "    input wire [19:0] B,\n"
            "    input wire Cin,\n"
            "    output wire [19:0] S,\n"
            "    output wire Cout\n"
            ");\n"
        )
    
    prompt_content += interface_injection

    # 写入文件
    file_path = os.path.join(directory, "prompt.txt") 
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(prompt_content)
    
    print(f"Prompt file created at: {file_path}")
    
# Example usage
# create_prompt_file("FA_1bit", "./output_directory")


def generate_DataReOrganize_code(width, a_tile_column_size):
    # Verilog header
    verilog_code = f"""
module DataReOrganize#(
    parameter data_width = {width},
    parameter  a_tile_column_size = {a_tile_column_size}
    )(
    // interface to system
    input wire clk,                         
    input wire rst_n,                   
    input wire en, 
    input wire [data_width * a_tile_column_size - 1 : 0] din,
    output reg [data_width * a_tile_column_size - 1 : 0] dout
    );
"""

    # Register declarations
    for a in range(a_tile_column_size):
        if a == 0:
            verilog_code += f"    reg [data_width - 1 : 0] row{a}_delay;\n"
        else:
            verilog_code += f"    reg [data_width * {a+1} - 1 : 0] row{a}_delay;\n"

    verilog_code += "\n// Always block for reset and functionality\n"
    verilog_code += "    always @(negedge rst_n or posedge clk) begin\n"
    verilog_code += "        if (~rst_n) begin\n"

    # Reset block
    for a in range(a_tile_column_size):
        verilog_code += f"            row{a}_delay <= 'd0;\n"

    verilog_code += "        end else if (en) begin\n"
    verilog_code += "            // input -> array\n"
    for a in range(a_tile_column_size):
        low = f"data_width * {a + 1} - 1"
        high = f"data_width * {a}"
        verilog_code += f"            row{a}_delay [data_width - 1 : 0] <= din [{low} : {high}];\n"

    verilog_code += "\n            // array flow\n"
    for a in range(1, a_tile_column_size):
        dst_hi = f"data_width * {a + 1} - 1"
        dst_lo = f"data_width"
        src_hi = f"data_width * {a} - 1"
        src_lo = f"0"
        verilog_code += f"            row{a}_delay [{dst_hi} : {dst_lo}] <= row{a-1}_delay [data_width * {a} - 1 : 0];\n"

    verilog_code += "\n            // array -> output\n"
    for a in range(a_tile_column_size):
        low = f"data_width * {a + 1} - 1"
        high = f"data_width * {a}"
        verilog_code += f"            dout [{low} : {high}] <= row{a}_delay [{low} : {high}];\n"

    verilog_code += "        end\n"
    verilog_code += "    end\n"
    verilog_code += "endmodule\n"

    return verilog_code

def generate_DataReOrganize_tb_files(min_width, max_width):
    base_path = '..'  # Root directory for file storage
    for DataReOrganize_type in range(21, 22):  # Outer loop for multi_type from 1 to 7
        if os.path.exists(os.path.join(base_path, f'module{DataReOrganize_type}')):
            shutil.rmtree(os.path.join(base_path, f'module{DataReOrganize_type}'))
        for test_num in range(min_width, max_width + 1):  # Loop num_tests times for each multi_type
            # Generate a random width between min_width and max_width
            width = test_num
            a_tile_column_size = 16
            for column in range(4, a_tile_column_size + 1):
                # Define the file path based on multi_type and test_num
                
                code_path = os.path.join(base_path, f'module{DataReOrganize_type}', f'column{column}', f'width{test_num}')
                code_tb_path = os.path.join(base_path, f'module{DataReOrganize_type}', f'column{column}', f'width{test_num}', 'code')
                if os.path.exists(code_path):
                    shutil.rmtree(code_path)
                os.makedirs(code_tb_path, exist_ok=True)
                generate_yosys_script(f'DataReOrganize', f'DataReOrganize_{column}_{width}bit', code_path)
                code = generate_DataReOrganize_code(width, column)
                code_file_path = os.path.join(code_tb_path, f'top_code.v')
                with open(code_file_path, 'w') as code_file:
                    code_file.write(code)
                create_verilog_filelist(code_path)
        print(f"DataReOrganize file (Type 21) has been saved to ../module{DataReOrganize_type}.")


generate_DataReOrganize_tb_files(1, 16)
