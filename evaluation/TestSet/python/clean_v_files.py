import os
import glob
import re

def clean_existing_verilog_files():
    """
    终极版清洗脚本：无视任何 Markdown 和自然语言，
    精准定位到第一个真正的 'module' 关键字，一直截取到文件末尾。
    """
    base_path = '..'
    
    # 搜索所有的生成的 .v 文件
    search_pattern = os.path.join(base_path, 'module*', 'test*', 'code', 'llm_code*.v')
    v_files = glob.glob(search_pattern)
    
    if not v_files:
        print("[错误] 没有找到任何 llm_code*.v 文件，请确认是否在 python 目录下运行此脚本。")
        return

    cleaned_count = 0
    for f in v_files:
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 终极正则匹配：寻找独立的单词 'module'，并匹配其后的所有内容 (re.DOTALL允许匹配换行符)
        match = re.search(r'\bmodule\b.*', content, re.DOTALL)

        if match:
            clean_code = match.group(0)
            
            # 防御性清洗：如果代码中还不小心残存了 ``` 符号，直接把它删掉
            clean_code = clean_code.replace('```', '')
            
            # 写回原文件
            with open(f, 'w', encoding='utf-8') as file:
                file.write(clean_code.strip() + "\n")
            cleaned_count += 1
        else:
            print(f"⚠️ [跳过] 文件 {f} 中连 'module' 关键字都没有，大模型可能完全生成失败了。")
            
    print(f"\n🎉 [大功告成] 成功清洗了 {cleaned_count} 个 Verilog 文件！")

if __name__ == "__main__":
    clean_existing_verilog_files()