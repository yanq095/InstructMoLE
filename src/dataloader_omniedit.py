import os
from zoneinfo import available_timezones
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
from datasets import load_dataset
import json
from src.condition import *
from src.adaptive_resize import AdaptiveResizeMultipleOf
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
import io

image_transform = T.Compose(
    [
        AdaptiveResizeMultipleOf(max_size=512, multiple_of=16),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]
)


class OmniEditDataset(Dataset):
    def __init__(
        self,
    ):
        # NOTE: You may need to adjust the dataset path
        super().__init__()
        dataset = load_dataset("dataset/omniedit_40k", cache_dir="cache/omniedit_40m")
        self.base_dataset = dataset["train"]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        src_img = item["src_img"].convert("RGB")
        edited_img = item["edited_img"].convert("RGB")
        prompt = item["edited_prompt_list"]
        if len(prompt) == 0:
            return None
        elif len(prompt) == 1:
            prompt = prompt[0]
        else:
            if random.random() < 0.3:
                prompt = prompt[0]
            else:
                prompt = prompt[1]

        task = item["task"]
        # 图片转换
        src_img = image_transform(src_img)
        edited_img = image_transform(edited_img)

        return {
            "src_img": src_img,
            "edited_img": edited_img,
            "prompt": prompt,
            "task": task,
        }


def pad_collate_fn(batch):
    keys = ["src_img", "edited_img"]
    batch_out = {}
    # 先处理需要pad和stack的字段
    for key in keys:
        images = [item[key] for item in batch]
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        padded_images = []
        for img in images:
            c, h, w = img.shape
            pad_h = max_h - h
            pad_w = max_w - w
            img = F.pad(img, (0, pad_w, 0, pad_h), value=0.0)
            padded_images.append(img)
        batch_out[key] = torch.stack(padded_images)
    # 处理其他字段（比如text、id_embed等），保持为list
    for k in batch[0]:
        if k not in keys:
            batch_out[k] = [d[k] for d in batch]
    return batch_out


def prepare_omniedit_dataloader(args, accelerator):
    train_dataset = OmniEditDataset()

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
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=pad_collate_fn,
    )
    return train_dataloader


# train_dataset = OmniEditDataset()
# for t in train_dataset:
#     print(t)
#     break
