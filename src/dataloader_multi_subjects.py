import os
from PIL import Image, ImageFilter, ImageDraw
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
from datasets import load_dataset, concatenate_datasets
import json
from transformers import pipeline
from src.adaptive_resize import AdaptiveResizeMultipleOf
import io
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler


def extract_subject_keyword(description: str) -> str:
    """从长描述中提取核心名词短语作为关键词。"""
    return description.split(',')[0].strip()

def join_references_naturally(references: list) -> str:
    """将指代短语列表用自然语言连接起来。"""
    if not references: return ""
    if len(references) == 1: return references[0]
    if len(references) == 2: return f"{references[0]} and {references[1]}"
    return ", ".join(references[:-1]) + f", and {references[-1]}"


FLEXIBLE_PROMPT_TEMPLATES = {
    # 等级1: 明确指代 (当 subject 描述存在时使用)
    "explicit": [
        "Based on {subject_references}, generate an image of the following scene: {target_scene}",
        "Create a single scene with {subject_references}. In this scene, {target_scene}.",
        "Using {subject_references}, create a picture where {target_scene}."
    ],
    # 等级2: 模糊指代 (只说参考，不描述具体内容)
    "vague": [
        "Using the subjects from the provided images, create a scene where {target_scene}.",
        "Combine the subjects from the source images into a single picture. The scene is: {target_scene}.",
        "Reference the given images to generate a new picture of this scene: {target_scene}."
    ],
    # 等级3: 完全隐式 (只描述目标，什么都不参考)
    "implicit": [
        "Generate a scene described as: {target_scene}",
        "Create an image of the following: {target_scene}",
        "{target_scene}" # 最简单直接的指令，就是目标场景本身
    ]
}

def parse_and_clean_data(data: dict) -> tuple[list, list]:
    subjects = [v for k, v in data.items() if k.startswith("subject_")]
    cleanup_pattern = "remove image borders and unify all subjects into a single scene. "
    cleaned_scenes = [v.replace(cleanup_pattern, "").strip() for k, v in data.items() if k.startswith("scene_")]
    return subjects, cleaned_scenes

def generate_flexible_prompt(subjects: list, target_scene: str) -> str:
    """
    根据输入信息，随机生成一个明确、模糊或隐式的指令。
    Args:
        subjects (list): 一个包含所有主体完整描述的列表。如果为空，则代表信息缺失。
        target_scene (str): 描述最终目标场景的单个字符串。
    Returns:
        str: 一条随机选择明确度的、用于训练的指令。
    """
    # 如果没有提供任何 subject 描述，只能生成“模糊”或“隐式”指令
    if not subjects:
        chosen_level = random.choice(["vague", "implicit"])
    else:
        # 如果有 subject 描述，我们可以从三个级别中随机选一个
        # 通过权重可以调整各类指令的生成比例，这里我们让明确指令占比更高
        levels = ["explicit", "vague", "implicit"]
        weights = [0.7, 0.2, 0.1] 
        chosen_level = random.choices(levels, weights=weights, k=1)[0]
    # 根据选择的级别，获取相应的模板
    template = random.choice(FLEXIBLE_PROMPT_TEMPLATES[chosen_level])
    # 准备填充模板的参数
    format_args = {"target_scene": target_scene}
    # 只有在需要“明确指代”时，才构建指代短语
    if chosen_level == "explicit":
        ordinal_words = ["first", "second", "third", "fourth", "fifth"]
        reference_phrases = []
        for i, desc in enumerate(subjects):
            keyword = extract_subject_keyword(desc)
            position = ordinal_words[i] if i < len(ordinal_words) else f"image {i+1}"
            reference_phrases.append(f"the {keyword} from the {position} image")
        format_args["subject_references"] = join_references_naturally(reference_phrases)
    return template.format(**format_args)


image_transform = T.Compose(
    [
        AdaptiveResizeMultipleOf(max_size=512, multiple_of=16),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]
)

class MultiSubsDataset(Dataset):
    def __init__(
        self,
        cond_cnt
    ):
        # NOTE: You may need to adjust the dataset path
        dataset = []
        paths = [
            "dataset/kontext_subjects/batch_1",
            "dataset/kontext_subjects/batch_2",
            "dataset/kontext_subjects/batch_3",
            "dataset/kontext_subjects/batch_4"]
        self.cond_cnt = cond_cnt
        def filter_func(item: dict) -> bool:
            subjects, _ = parse_and_clean_data(json.loads(item['prompt']))
            sub_cnt = len(subjects)
            if sub_cnt == cond_cnt:
                return True 
            else:
                return False
        
        for path in paths:
            dataset.append(load_dataset(
                path,
                split="train",
                cache_dir=f"cache/3subs",
            ))
        dataset = concatenate_datasets(dataset).shuffle()
        data_valid = dataset.filter(
            filter_func,
            num_proc=8,
            cache_file_name=f"cache/3subs/{cond_cnt}_subs.arrow",
        )
        self.dataset = data_valid

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        subjects, cleaned_scenes = parse_and_clean_data(json.loads(item['prompt']))
        prompt = generate_flexible_prompt(subjects, cleaned_scenes[0])
        sub_cnt = len(subjects)
        img = Image.open(io.BytesIO(item['stiched_image'])).convert("RGB")
        width, height = img.size
        mid_x = width // 2
        if sub_cnt == 2:
            mid_y = height
        else:
            mid_y = height // 2
        crop_boxes = [(0, 0, mid_x, mid_y), 
                      (mid_x, 0, width, mid_y), 
                      (0, mid_y, mid_x, height),
                      (mid_x, mid_y, width, height)]
        conds = []
        for i in range(sub_cnt):
            img_crop = img.crop(crop_boxes[i])
            # print(img_crop.size)
            img_crop = image_transform(img_crop)
            conds.append(img_crop)
        conds = torch.stack(conds, dim=0)
        image = Image.open(io.BytesIO(item['scene_images'][0])).convert("RGB")
        # --- 4. Return the updated dictionary structure ---
        return {
            "pixel_values": image_transform(image),
            "condition_latents": conds,
            "condition_types": "subject",  
            "descriptions": prompt, 
            # "items": description, 
            }


# def pad_collate_fn(batch):
#     keys = ['pixel_values', 'condition_latents']
#     batch_out = {}
#     # 先处理需要pad和stack的字段
#     for key in keys:
#         images = [item[key] for item in batch]
#         max_h = max(img.shape[1] for img in images)
#         max_w = max(img.shape[2] for img in images)
#         padded_images = []
#         for img in images:
#             c, h, w = img.shape
#             pad_h = max_h - h
#             pad_w = max_w - w
#             img = F.pad(img, (0, pad_w, 0, pad_h), value=0.0)
#             padded_images.append(img)
#         batch_out[key] = torch.stack(padded_images)
#     # 处理其他字段（比如text、id_embed等），保持为list
#     for k in batch[0]:
#         if k not in keys:
#             batch_out[k] = [d[k] for d in batch]
#     return batch_out

def prepare_multisub_dataloader(cond_num, args, accelerator):
    train_dataset = MultiSubsDataset(cond_cnt=cond_num)
    bsz = max(1, args.train_batch_size//cond_num)
    train_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        drop_last=True,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        sampler=train_sampler,
        batch_size=bsz,
        num_workers=args.dataloader_num_workers,
        pin_memory=True
    )
    return train_dataloader

# train_dataset = ThreeSubsDataset()
# train_dataloader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=2,
#         shuffle=True,
#         num_workers=0,
#     )
# first_batch = next(iter(train_dataloader))
# print(len(train_dataloader))
# print(first_batch['descriptions'])