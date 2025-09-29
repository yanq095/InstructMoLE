from PIL import Image
import io
from src.data.dataset_square import KVDataset
import json
import bson
from src.adaptive_resize import AdaptiveResizeMultipleOf, PasteToCenterCanvas
import torch
import torchvision.transforms as T
from dataloader import KVReader
import random
import torch.nn.functional as F
from typing import List, Optional

prompt_templates = [
                        # === 详细指令 (Detailed Instructions) ===
                        # Direct & Technical
                        "Using the person of the image as a identity reference, render a scene of {desc}",
                        "Using the people of the image as a identity reference to recreate the image.",
                        "Generate an image of the person from the reference face as {desc}",
                        "The person in the reference image, now depicted as {desc}",
                        "Same person from the photo, new scene: {desc}",
                        "Reference the face of the image, new scene: {desc}",
                        "Generate an image of the person from the reference face.",
                        "{desc}"
                    ]


def generate_prompt(prompt):
    chosen_template = random.choice(prompt_templates)
    if random.random() < 0.5:
        parts = prompt.split('.', 1) 
        prompt = parts[0].strip() + '.' if parts else prompt
    text = chosen_template.format(desc=prompt)
    return text


class SampleDecoder:
    def __call__(self, item):
        try:
            image = Image.open(io.BytesIO(item["image"])).convert("RGB")
            if random.random() < 0.7:
                face = Image.open(io.BytesIO(item["base"])).convert("RGB")
            else:
                face = Image.open(io.BytesIO(item["base_face_image"])).convert("RGB")
            return {
                "text": generate_prompt(item["prompt"]),
                "image": image,
                "ref_face": face,
            }
        except Exception as e:
            print(f"SampleDecoder error: {e}")
            return None


class SPMVDataset(KVDataset):

    def __init__(
        self,
        paths=[
            " hdfs://harunafr/home/byte_voyager_fr/user/jinqi.xiao/data/spmv_format_filter",
        ],
        rank=0,
        world_size=1,
        shuffle=False,
        image_transform=T.Compose(
            [
                AdaptiveResizeMultipleOf(max_size=512, multiple_of=16),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        ),
    ):
        super().__init__(paths, rank, world_size, shuffle)
        self.sample_decoder = SampleDecoder()
        self.image_transform = image_transform
        self._length = 2034624
        
        # for filepath in self.filepaths:
        #     try:
        #         reader = KVReader(filepath)
        #         self._length += len(reader.list_keys())
        #     except Exception as ex:
        #         print(f"Error counting keys in {filepath}: {ex}")
        #         continue
        # print(f"Dataset length: {self._length}")

    def __len__(self):
        return self._length

    def __iter__(self):
        for item in super().__iter__():
            try:
                try:
                    item = bson.loads(item)
                except:
                    item = json.loads(item)
                sample = self.sample_decoder(item)
                if sample is None:
                    continue

                if self.image_transform:
                    sample["image"] = self.image_transform(sample["image"])
                    sample["ref_face"] = self.image_transform(sample["ref_face"])

                yield sample
            except Exception as ex:
                print(f"Error: {ex}")
                continue


def pad_collate_fn(batch):
    keys = ["image", "ref_face"]
    # if "mask" in batch[0]:
    #     keys.append("mask")
    batch_out = {}

    for key in keys:
        images = [item[key] for item in batch]
        max_h = max(img.shape[-2] for img in images)
        max_w = max(img.shape[-1] for img in images)
        padded_images = []
        for img in images:
            h, w = img.shape[-2:]
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


def prepare_spmv_dataloader(args, accelerator):
    train_dataset = SPMVDataset(
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    bsz = args.train_batch_size
    # if cond_num > 10:
    #     bsz = 1
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=bsz,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=pad_collate_fn,
    )
    return train_dataloader


# dataset = SPMVDataset()

# for sample in dataset:
#     print(sample["text"])
#     print(sample["ref_face"].shape)
