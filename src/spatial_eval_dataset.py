import argparse
import random
import cv2
import numpy as np
from PIL import Image
from datasets import load_dataset, VerificationMode

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as T
from src.dataloader_pose import generate_control_prompt

# --- 配置信息 ---
DATASET_CONFIGS = {
    "depth": {
        "hf_path": "dataset/limingcv/MultiGen-20M_depth_eval",
        "split": "validation",
        "image_col": "image",
        "condition_col": "control_depth",
        "text_col": "text",
    },
    "canny": {
        "hf_path": "dataset/limingcv/MultiGen-20M_canny_eval",
        "split": "validation",
        "image_col": "image",
        "condition_col": None,
        "text_col": "text",
    },
    "pose": {
        "hf_path": "dataset/limingcv/Captioned_COCOPose",
        "split": "validation",
        "image_col": "image",
        "condition_col": "control_pose",
        "text_col": "caption",
        "data_files": {"validation": "data/validation-*.parquet"},
        "verification_mode": VerificationMode.NO_CHECKS,
    },
}


# --- 辅助函数 ---
def get_canny_edge(
    img: Image.Image, low_threshold: int = 100, high_threshold: int = 200
) -> Image.Image:
    """从输入的PIL图像生成Canny边缘图。"""
    img_np = np.array(img.convert("RGB"))
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, low_threshold, high_threshold)
    return Image.fromarray(edges).convert("RGB")


class SpatialAlignmentEvalDataset(Dataset):
    """
    一个用于空间对齐评估的数据集，保持图像的原始尺寸。
    """

    def __init__(self, task: str, hf_cache_dir: str = None, max_samples: int = None):
        if task not in DATASET_CONFIGS:
            raise ValueError(
                f"无效的任务类型 '{task}'. 可选任务为: {list(DATASET_CONFIGS.keys())}"
            )

        self.task = task
        config = DATASET_CONFIGS[task]

        print(f"--- 正在加载数据集: {config['hf_path']} ---")

        # ** THE FIX **: Use the 'data_files' key if it exists in the config.
        dataset_full = load_dataset(
            config["hf_path"],
            split=config["split"],
            cache_dir=hf_cache_dir,
            data_files=config.get("data_files", None),
            verification_mode=config.get(
                "verification_mode", VerificationMode.BASIC_CHECKS
            ),
        )
        if max_samples is not None:
            num_to_select = min(max_samples, len(dataset_full))
            self.dataset = dataset_full.select(range(num_to_select))
            print(f"--- 数据集已截断，仅使用前 {len(self.dataset)} 个样本 ---")
        else:
            self.dataset = dataset_full
        print("--- 数据集加载完成 ---")

        self.image_col = config["image_col"]
        self.condition_col = config["condition_col"]
        self.text_col = config["text_col"]

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        target_image = item[self.image_col].convert("RGB")

        if self.task == "canny":
            condition_image = get_canny_edge(target_image)
        else:
            condition_image = item[self.condition_col].convert("RGB")

        prompt = item[self.text_col]
        if isinstance(prompt, list):
            prompt = prompt[0]

        image_tensor = self.transform(target_image)
        condition_tensor = self.transform(condition_image)

        return {
            "image": image_tensor,
            "condition": condition_tensor,
            "prompt": generate_control_prompt(self.task, prompt),
            "text": prompt,
        }


def prepare_spatial_eval_dataloader(task, accelerator):
    eval_dataset = SpatialAlignmentEvalDataset(
        hf_cache_dir="cachespatial_eval_cache", task=task, max_samples=500
    )

    eval_sampler = DistributedSampler(
        dataset=eval_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        sampler=eval_sampler,
        batch_size=1,
        num_workers=2,
    )
    return eval_dataloader
