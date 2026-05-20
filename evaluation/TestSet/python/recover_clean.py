import os
import glob
import re

def recover_and_clean():
    base_path = '..'
    # 读取原始的、未经破坏的 txt 备份文件
    txt_files = glob.glob(os.path.join(base_path, 'module*', 'test*', 'code', 'llm_code*.txt'))
    
    cleaned_count = 0
    for f in txt_files:
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 1. 以 'assistant' 为界限，丢弃前面所有的 Prompt 上下文
        parts = re.split(r'assistant', content, flags=re.IGNORECASE)
        generated_text = parts[-1]  # 只取最后大模型真正生成的内容
        
        # 【修复 2】提前初始化变量，防止未匹配时报错
        clean_code = None 
        
        # 2. 提取从 module 到 endmodule 之间的代码
        match = re.search(r'(\bmodule\b.*?\bendmodule\b)', generated_text, re.DOTALL)
        
        # 【修复 1】把错误的 md_match 改为正确的 match
        if match:
            clean_code = match.group(1)
        else:
            # 策略 B：严格匹配 Verilog 语法规则
            strict_match = re.search(r'(\bmodule\s+[a-zA-Z_]\w*\b.*?\bendmodule\b)', generated_text, re.DOTALL)
            if strict_match:
                clean_code = strict_match.group(1)
        
        # 3. 写入最终的 .v 文件
        if clean_code:
            # 最后做一次防御，防止有残留的 Markdown 符号（虽然正则已经去掉了外面，但防止意外嵌套）
            clean_code = clean_code.replace('```verilog', '').replace('
```', '')
            
            v_file = f.replace('.txt', '.v')
            with open(v_file, 'w', encoding='utf-8') as file:
                file.write(clean_code.strip() + "\n")
            cleaned_count += 1
        else:
            print(f"⚠️ [跳过] 文件 {f} 中未找到任何有效的代码。")
            
    print(f"\n🎉 [满血复活] 成功用防弹策略清洗了 {cleaned_count} 个 Verilog 文件！")

if __name__ == "__main__":
    recover_and_clean()