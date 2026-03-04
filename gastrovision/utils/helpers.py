input_file = 'D:/codes/work-projects/Gastrovision_models/data/test_mul.txt'  # 你的原始文件路径
output_file = 'D:/codes/work-projects/Gastrovision_models/data/test1.txt' # 你想保存的新文件路径

with open(input_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:
    
    for line in f_in:
        line = line.strip()
        if not line: continue
        
        # 核心逻辑：按空格分割
        parts = line.split('.jpg')
        
        image_path = parts[0] + '.jpg'
        first_label = parts[1].split()[0]
        
        # 重新组合并写入新文件
        new_line = f"{image_path} {first_label}\n"
        f_out.write(new_line)

print("处理完成，已生成单标签文件。")