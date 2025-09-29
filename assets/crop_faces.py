import os
import face_recognition
from PIL import Image, ImageOps
import numpy as np

def detect_and_crop_faces(input_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace('.png', '.jpg'))

            # 加载图像并处理透明背景
            image = Image.open(input_path).convert("RGBA")
            background = Image.new("RGBA", image.size, "WHITE")
            alpha_composite = Image.alpha_composite(background, image).convert("RGB")

            # 添加白色边缘，这里 padding 设为 10 像素，可按需调整
            padded_image = ImageOps.expand(alpha_composite, border=10, fill='white')

            # 尝试不同尺度的图像检测
            scales = [0.6, 0.4, 0.2]
            face_locations = []
            for scale in scales:
                resized_image = padded_image.resize((int(padded_image.width * scale), int(padded_image.height * scale)), Image.LANCZOS)
                image_np = np.array(resized_image)
                # Use the cnn model for detection
                face_locations = face_recognition.face_locations(image_np, model="cnn")
                if face_locations:
                    # Adjust the detected face positions to the original image size
                    face_locations = [(int(top / scale), int(right / scale), int(bottom / scale), int(left / scale)) for top, right, bottom, left in face_locations]
                    break

            if face_locations:
                # 假设第一个检测到的人脸是需要裁剪的
                top, right, bottom, left = face_locations[0]
                height = bottom - top
                width = right - left

                # 计算扩充后的区域
                new_top = max(0, int(top - height * 0.3))
                new_bottom = min(np.array(padded_image).shape[0], int(bottom + height * 0.3))
                new_left = max(0, int(left - width * 0.3))
                new_right = min(np.array(padded_image).shape[1], int(right + width * 0.3))

                face_image = np.array(padded_image)[new_top:new_bottom, new_left:new_right]
                # 将 NumPy 数组转换为 PIL 图像
                face_pil = Image.fromarray(face_image)
                # 保存裁剪后的人脸图像
                face_pil.save(output_path)
                print(f"已裁剪并保存: {output_path}")
            else:
                print(f"未在 {input_path} 中检测到人脸")

if __name__ == "__main__":
    input_directory = "/mnt/bn/yg-butterfly-algo/personal/sunhm/code/XVerse/assets/XVerseBench_seg/human_seg"
    output_directory = "/mnt/bn/yg-butterfly-algo/personal/sunhm/code/XVerse/assets/XVerseBench_seg/human"
    detect_and_crop_faces(input_directory, output_directory)
