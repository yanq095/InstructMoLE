from src.utils.data_utils import get_train_config, image_grid, pil2tensor, json_dump, pad_to_square, cv2pil, merge_bboxes
from eval.tools.florence_sam import ObjectDetector
import torch
import os
from PIL import Image  # 补充导入 Image 模块
import numpy as np

def merge_instances(orig_img, indices, ins_bboxes, ins_images):
    orig_image_width, orig_image_height = orig_img.width, orig_img.height
    final_img = Image.new("RGB", (orig_image_width, orig_image_height), color=(255, 255, 255))
    bboxes = []
    for i in indices:
        bbox = np.array(ins_bboxes[i], dtype=int).tolist()
        bboxes.append(bbox)
        
        img = cv2pil(ins_images[i])
        mask = (np.array(img)[..., :3] != 255).any(axis=-1)
        mask = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
        final_img.paste(img, (bbox[0], bbox[1]), mask)
    
    bbox = merge_bboxes(bboxes)
    img = final_img.crop(bbox)
    return img, bbox

dtype = torch.bfloat16
device = "cuda"
detector = ObjectDetector(device)
def det_seg_img(image, label):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    instance_result_dict = detector.get_multiple_instances(image, label, min_size=image.size[0]//20)
    indices = list(range(len(instance_result_dict["instance_images"])))
    ins, bbox = merge_instances(image, indices, instance_result_dict["instance_bboxes"], instance_result_dict["instance_images"])
    return ins

def segment_images_in_folder(input_folder, output_folder):
    """
    对输入文件夹内所有图像进行分割，并将结果保存到输出文件夹。

    :param input_folder: 输入图像文件夹路径
    :param output_folder: 输出分割结果的文件夹路径
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹及其子文件夹内的所有文件
    for root, _, filenames in os.walk(input_folder):
        for filename in filenames:
            # 检查是否为图像文件
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, filename)
                try:
                    # 从文件名中提取标签，假设文件名格式为 "数字_标签.png"
                    label = filename.split('_')[-1].rsplit('.', 1)[0].strip()
                    # 进行图像分割
                    segmentation_result = det_seg_img(file_path, label)
                    # 构建输出文件路径，保持原文件名
                    relative_path = os.path.relpath(root, input_folder)
                    output_subfolder = os.path.join(output_folder, relative_path)
                    os.makedirs(output_subfolder, exist_ok=True)
                    output_path = os.path.join(output_subfolder, filename)
                    # 保存分割结果
                    if isinstance(segmentation_result, Image.Image):
                        segmentation_result.save(output_path)
                    else:
                        # 假设 segmentation_result 是可转换为 PIL Image 的对象
                        Image.fromarray(segmentation_result).save(output_path)
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")


# 使用示例
if __name__ == "__main__":
    input_folder = "./assets/XverseBench_rename"
    output_folder = "./assets/XVerseBench"
    segment_images_in_folder(input_folder, output_folder)
