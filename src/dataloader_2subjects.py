from accelerate.logging import get_logger
import torch
import io

logger = get_logger(__name__)
from PIL import Image
from .condition import Condition
from diffusers.image_processor import VaeImageProcessor
from datasets import load_dataset, concatenate_datasets
import random
import re
from torch.utils.data import DataLoader, DistributedSampler

def rephrase_descriptions_with_tokens(short_desc, long_desc):
    # 1. 从 'short' 描述中解析出主体
    # 我们用 " and " 作为分隔符，并处理首尾空格
    subjects = [s.strip() for s in short_desc.split(' and ')]
    
    # 如果解析出的主体不足两个，则无法进行替换，直接返回原始数据
    if len(subjects) < 2:
        print(f"警告: 'short' 描述 '{short_desc}' 中未能解析出两个主体。")
        return short_desc, long_desc

    # 为了演示，我们只取前两个主体
    subject1, subject2 = subjects[0], subjects[1]

    # --- 开始改造 ---
    
    # 2. 随机化介词
    prepositions = ['of', 'from', 'in']
    prep1 = random.choice(prepositions)
    prep2 = random.choice(prepositions)
    
    # 3. 创建带Token的新主体短语
    if random.random() < 0.5:
        new_subject1_phrase = f"{subject1} {prep1} the first image"
    else:
        new_subject1_phrase = f"{subject1} {prep1} the left image"
    if random.random() < 0.5:
        new_subject2_phrase = f"{subject2} {prep2} the second image"
    else:
        new_subject2_phrase = f"{subject2} {prep2} the right image"
    
    # 4. 改造 'short' 描述
    if random.random() < 0.5:
        rephrased_short = f"{new_subject1_phrase} and {new_subject2_phrase}"
    else:
        rephrased_short = f"{new_subject2_phrase} and {new_subject1_phrase}"
    
    # 5. 改造 'long' 描述 (这是最关键的一步)
    # 我们需要用一种对大小写不敏感的方式来替换
    # re.sub 的一个技巧是使用一个函数作为替换参数
    
    rephrased_long = long_desc
    
    # 替换第一个主体 (e.g., "calculator")
    # `re.IGNORECASE` 使得替换不区分大小写
    # `\b` 是单词边界符，确保我们只替换完整的单词 (例如，不会替换 "calculators" 中的 "calculator")
    rephrased_long = re.sub(r'\b' + re.escape(subject1) + r'\b', new_subject1_phrase, rephrased_long, flags=re.IGNORECASE, count=1)

    # 替换第二个主体 (e.g., "mirror")
    rephrased_long = re.sub(r'\b' + re.escape(subject2) + r'\b', new_subject2_phrase, rephrased_long, flags=re.IGNORECASE, count=1)
    
    # 检查是否还有未被替换的主体，这可能发生在同一主体出现多次的情况下
    # 我们可以选择将后续出现的也替换掉，或者只替换第一次出现的（通常只替换第一次更稳健）
    # (上面的 `count=1` 参数确保了只替换第一次出现)
    return rephrased_short, rephrased_long

def get_dataset():
    dataset = []
    dataset_name = ["dataset/batch042825_pq", "dataset/batch062425_pq"]
    for name in dataset_name:
        # Downloading and loading a dataset from the hub.
        dataset.append(load_dataset(name, cache_dir='cache/2subs', split="train"))
    dataset = concatenate_datasets(dataset).shuffle()
    return dataset


def prepare_dataset(dataset, accelerator):
    image_processor = VaeImageProcessor(
        do_resize=True, do_convert_rgb=True
    )
    resolution = 512
    def preprocess_conditions(conditions):
        conditioning_tensors = []
        conditions_types = []
        for cond in conditions:
            conditioning_tensors.append(
                image_processor.preprocess(
                    cond.condition, width=resolution, height=resolution
                ).squeeze(0)
            )
            conditions_types.append(cond.condition_type)
        return torch.stack(conditioning_tensors, dim=0), conditions_types

    def preprocess(examples):
        # images = [image_transforms(image) for image in images]
        pixel_values = []
        condition_latents = []
        condition_types = []
        # bboxes = []
        short_dess = []
        long_dess = []
        for image, des in zip(examples["image"], examples["description"]):
            image = Image.open(io.BytesIO(image))
            long_des = des["long"]
            short_des = des["short"]
            short_des, long_des = rephrase_descriptions_with_tokens(short_des, long_des)
            width, height = image.size
            # 检查宽度是否为3，以便可以均匀分割
            if width % 3 != 0:
                raise ValueError(
                    "Image width must be even to split into two equal parts."
                )
            mean_width = width // 3
            left_image = image.crop((0, 0, mean_width, height))
            mid_image = image.crop((mean_width, 0, mean_width * 2, height))
            right_image = image.crop((mean_width * 2, 0, width, height))
            # 应用转换,将分割后的图像添加到列表中
            pixel_values.append(
                image_processor.preprocess(
                    left_image, width=resolution, height=resolution
                ).squeeze(0)
            )
            conditions = []
            # for condition_type in condition_types:
            #     if condition_type == "subject":
            #         conditions.append(Condition("subject", condition=right_image))
            #     elif condition_type == "canny":
            #         conditions.append(
            #             Condition(
            #                 "canny", condition=Image.open(io.BytesIO(canny["bytes"]))
            #             )
            #         )
            #     elif condition_type == "depth":
            #         conditions.append(
            #             Condition(
            #                 "depth", condition=Image.open(io.BytesIO(depth["bytes"]))
            #             )
            #         )
            #     elif condition_type == "fill":
            #         conditions.append(Condition("fill", condition=masked_left_image))
            #     else:
            #         raise ValueError("Only support for subject, canny, depth, fill")
            conditions.append(Condition("subject", condition=mid_image))
            conditions.append(Condition("subject", condition=right_image))
            cond_tensors, cond_types = preprocess_conditions(conditions)
            condition_latents.append(cond_tensors)
            condition_types.append(cond_types)
            short_dess.append(short_des)
            long_dess.append(long_des)
        
        examples["pixel_values"] = pixel_values
        examples["condition_latents"] = condition_latents
        examples["condition_types"] = condition_types
        
        examples["descriptions"] =long_dess
        examples["items"] = short_dess
        # examples["bbox"] = None
        return examples

    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess)

    return dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    condition_latents = torch.stack(
        [example["condition_latents"] for example in examples]
    )
    condition_latents = condition_latents.to(
        memory_format=torch.contiguous_format
    ).float()
    # bboxes = [example["bbox"] for example in examples]
    condition_types = [example["condition_types"] for example in examples]
    descriptions = [example["descriptions"] for example in examples]
    items = [example["items"] for example in examples]
    return {
        "pixel_values": pixel_values,
        "condition_latents": condition_latents,
        "condition_types": condition_types,
        "descriptions": descriptions,
        # "bboxes": bboxes,
        "items": items,
    }


def prepare_2subs_dataloader(accelerator, args):
    train_dataset = prepare_dataset(
        get_dataset(), accelerator,
    )
    train_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        drop_last=True,
    )
    train_dataloader_sub = DataLoader(
        train_dataset,
        shuffle=False,
        sampler=train_sampler,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    return train_dataloader_sub