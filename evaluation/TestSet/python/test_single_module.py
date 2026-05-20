import os
import re
import subprocess
import glob

def test_single_module(target_module_id, num_tests=5):
    """
    只针对指定的单个模块 (例如 module1) 进行编译和仿真测试。
    """
    base_path = '..'
    module_name = f'module{target_module_id}'
    
    print(f"========== 开始测试单一模块: {module_name} ==========")
    compilation_correct = 0
    functional_correct = 0

    for j in range(1, num_tests + 1):
        test_name = f'test{j}'
        code_path = os.path.join(base_path, module_name, test_name, 'code')
        result_path = os.path.join(base_path, module_name, test_name, 'result')
        os.makedirs(result_path, exist_ok=True)

        output_vvp = os.path.join(result_path, 'out.vvp')
        output_txt = os.path.join(result_path, 'out.txt')
        
        # 获取该测试用例下的所有 .v (或 .sv) 文件
        v_files = glob.glob(os.path.join(code_path, '*.v'))
        if target_module_id > 31: # 处理可能存在的 sv 文件
            v_files = glob.glob(os.path.join(code_path, '*.sv'))
            
        if not v_files:
            print(f"[{test_name}] [跳过] 未找到任何代码文件。")
            continue

        compile_command = [
            'iverilog', '-o', output_vvp, '-y', code_path, '-s', 'tb', '-I', code_path
        ] + v_files

        # 1. 执行编译
        compile_result = subprocess.run(compile_command, check=False, text=True, capture_output=True)

        if compile_result.returncode != 0:
            print(f"\n❌ [{test_name}] 编译失败! 错误日志如下:")
            print("-" * 40)
            print(compile_result.stderr.strip())  # 直接在屏幕上打印错误，方便调试！
            print("-" * 40)
            with open(output_txt, "w") as outfile:
                outfile.write(f"Compilation failed.\n{compile_result.stderr}")
            continue  
            
        print(f"✅ [{test_name}] 编译成功!")
        compilation_correct += 1

        # 2. 执行仿真
        run_command = ['vvp', output_vvp]
        with open(output_txt, "w") as outfile:  
            run_result = subprocess.run(run_command, check=False, text=True, stdout=outfile, stderr=subprocess.PIPE)

        if run_result.returncode != 0:
            print(f"❌ [{test_name}] 仿真运行失败!")
            continue

        # 3. 检查正确率
        with open(output_txt, "r") as file:
            content = file.read()
            match = re.search(r"(Pass(?:ed)? rate|Accuracy):\s*([0-9.]+)%", content, re.IGNORECASE)
            if match and float(match.group(2)) == 100.0:
                functional_correct += 1
                print(f"🎉 [{test_name}] 功能验证 100% 通过!")
            else:
                print(f"⚠️ [{test_name}] 功能验证未达到 100%。")

    print(f"\n========== {module_name} 最终成绩 ==========")
    print(f"总测试数: {num_tests}")
    print(f"编译通过: {compilation_correct} 个")
    print(f"功能全对: {functional_correct} 个")

if __name__ == "__main__":
    # 在这里修改你想测试的模块编号！比如想测 module1 就写 1，想测 module6 就写 6。
    TARGET_MODULE = 1 
    
    test_single_module(target_module_id=TARGET_MODULE)