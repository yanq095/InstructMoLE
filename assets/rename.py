import os
import shutil

split = [("live_subject/animal", "animal"), ("object", "object")]

# 定义目录路径
caption_dir_base = './data/DreamBench_plus/captions'
image_dir_base = './data/DreamBench_plus/images'
new_image_dir_base = './data/XVerseBench_rename'

for s, ts in split:
    caption_dir = os.path.join(caption_dir_base, s)
    image_dir = os.path.join(image_dir_base, s)
    new_image_dir = os.path.join(new_image_dir_base, ts)

    # 创建新的目标目录（如果不存在）
    if not os.path.exists(new_image_dir):
        os.makedirs(new_image_dir)

    # 获取所有 caption 文件
    caption_files = sorted([f for f in os.listdir(caption_dir) if f.endswith('.txt')])

    for caption_file in caption_files:
        # 提取索引
        index = os.path.splitext(caption_file)[0]
        # 构建 caption 文件完整路径
        caption_file_path = os.path.join(caption_dir, caption_file)
        # 构建对应的图片文件路径
        image_file_name = f'{index}.jpg'
        image_file_path = os.path.join(image_dir, image_file_name)

        # 检查图片文件是否存在
        if os.path.exists(image_file_path):
            # 读取 caption 文件内容
            with open(caption_file_path, 'r', encoding='utf-8') as f:
                caption = f.read().split('\n')[0].strip()

            # 生成新的文件名
            new_file_name = f'{index}_{caption}.jpg'
            new_file_path_in_new_dir = os.path.join(new_image_dir, new_file_name)

            # 移动并重命名文件
            shutil.copy2(image_file_path, new_file_path_in_new_dir)
            print(f'文件 {image_file_path} 已移动并重命名为 {new_file_path_in_new_dir}')
        else:
            print(f'未找到对应的图片文件: {image_file_path}')


old_human_index = ['00', '05', '06', '09', '12', '13', '14', '16', '17']

# 新增的文件映射
new_files = [
    "object/65_anime space ranger.jpg", "object/66_anime girl.jpg", "object/67_pixelated warrior.jpg",
    "object/68_anime girl.jpg", "object/69_anime samurai.jpg", "object/70_anime girl.jpg",
    "object/71_anime Spider-Man.jpg", "object/72_Avatar.jpg", "object/73_anime man.jpg"
]

# 新增复制文件的代码
for old_human_index, new_file in zip(old_human_index, new_files):
    # 构建原始图片文件路径
    original_image_path = os.path.join(image_dir_base, "live_subject/human", f"{old_human_index}.jpg")
    # 构建新的图片文件路径
    new_image_path = os.path.join(new_image_dir_base, new_file)
    
    # 创建新文件的目录（如果不存在）
    new_image_dir = os.path.dirname(new_image_path)
    if not os.path.exists(new_image_dir):
        os.makedirs(new_image_dir)
    
    # 检查原始图片文件是否存在
    if os.path.exists(original_image_path):
        # 复制文件
        shutil.copy2(original_image_path, new_image_path)
        print(f'文件 {original_image_path} 已复制到 {new_image_path}')
    else:
        print(f'未找到对应的图片文件: {original_image_path}')