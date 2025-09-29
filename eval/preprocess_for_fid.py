# preprocess_for_fid.py
import os
import argparse
from PIL import Image
from tqdm.auto import tqdm


def preprocess_images(input_dir, output_dir, size=299):
    """
    遍历输入目录，将所有图像调整到指定尺寸并保存到输出目录。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(f"正在处理 {len(files)} 张图片，将尺寸统一为 {size}x{size}...")

    for filename in tqdm(files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            with Image.open(input_path) as img:
                # 确保是 RGB 格式
                img = img.convert("RGB")
                # 使用高质量的 LANCZOS 插值方法进行缩放
                img_resized = img.resize((size, size), Image.Resampling.LANCZOS)
                img_resized.save(output_path)
        except Exception as e:
            print(f"处理文件 {filename} 时出错，已跳过: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为 FID 计算预处理图像，统一尺寸。")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="包含原始图像的文件夹。"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="保存处理后图像的输出文件夹。"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=299,
        help="所有图像的目标尺寸 (size x size)。299 是 InceptionV3 的标准尺寸。",
    )

    args = parser.parse_args()
    preprocess_images(args.input_dir, args.output_dir, args.size)
    print("预处理完成！")
