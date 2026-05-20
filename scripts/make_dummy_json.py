import os
import json

# 读取你刚才生成的图片文件夹
input_dir = 'jpg_150dpi'
fake_data = []

# 为每一张图片生成一个假的 JSON 条目
if os.path.exists(input_dir):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.jpg'):
            # 伪造的数据结构，只需满足 process_image.py 和 compare 的读取需求即可
            fake_data.append({"images": [filename]})

# 生成那个缺失的文件
with open('combined_descriptions.json', 'w', encoding='utf-8') as f:
    json.dump(fake_data, f, indent=4)

print("✅ 成功生成占位符 combined_descriptions.json！")