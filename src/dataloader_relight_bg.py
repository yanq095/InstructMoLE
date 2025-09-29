from PIL import Image
import io
from src.data.dataset_square import KVDataset
import json
import bson
from src.adaptive_resize import AdaptiveResizeMultipleOf
import torch
import torchvision.transforms as T
from dataloader import KVReader
import random
import torch.nn.functional as F


SUBJECT_PHRASES = {
    "simple": [
        "the subject",
        "the foreground",
    ],
    "explicit": [
        "the subject in the first image",
        "the subject from image 1",
        "the foreground element from the first image",
    ]
}
# Define phrases for the {background} placeholder
BACKGROUND_PHRASES = {
    "simple": [
        "the background",
        "the new background",
        "the scene",
    ],
    "explicit": [
        "the background of the second image",
        "the background from image 2",
        "the new background provided in the second image",
    ]
}
# --- Prompt Template Library (with Placeholders) ---
# Templates for Direct Background Replacement
BG_REPLACE_TEMPLATES = [
    "Replace {background}.",
    "Change {background}.",
    "Put the {subject} on the new {background}.",
    "Composite the {subject} onto {background}.",
    "Swap {background}.",
    "Place the {subject} into the new {background}.",
]
# Templates for Background Replacement with Relighting
BG_RELIGHT_TEMPLATES = [
    "Relight the {subject} with {background}.",
    "Relight the {subject} using {background}.",
    "Blend the {subject} naturally into the new {background}.",
    "Match the {subject} lighting to the {background}.",
    "Harmonize the lighting and color between the {subject} and the new {background}.",
    "Integrate the {subject} into {background} with realistic lighting.",
    "Change {background} and adjust the {subject} to match.",
]
def generate_prompt(task_type: str, explicitness_prob: float = 0.5) -> str:
    # 1. Select the appropriate template list
    if task_type == 'replace':
        template = random.choice(BG_REPLACE_TEMPLATES)
    elif task_type == 'relight':
        template = random.choice(BG_RELIGHT_TEMPLATES)
    else:
        raise ValueError("Invalid task_type. Please choose 'replace' or 'relight'.")
    # 2. Decide whether to use simple or explicit phrases
    if random.random() < explicitness_prob:
        subject_text = random.choice(SUBJECT_PHRASES["explicit"])
        background_text = random.choice(BACKGROUND_PHRASES["explicit"])
    else:
        subject_text = random.choice(SUBJECT_PHRASES["simple"])
        background_text = random.choice(BACKGROUND_PHRASES["simple"])
    # 3. Format the template with the chosen phrases
    # The .format() method gracefully handles templates that might only use one
    # of the placeholders (e.g., "Change {background}.")
    prompt = template.format(subject=subject_text, background=background_text)
    return prompt

class SampleDecoder:
    def __call__(self, item):
        try:
            image_ori = "image_ori"
            if random.random() < 0.5:
                image_ori = "image_rmbg"
            image_ori = Image.open(io.BytesIO(item[image_ori])).convert("RGB")
            bg_img = Image.open(io.BytesIO(item["bg"])).convert("RGB")
            if random.random() < 0.5:
                prompt = generate_prompt("replace")
                image = Image.open(io.BytesIO(item['image_bg'])).convert("RGB")
            else:
                prompt = generate_prompt("relight")
                image = Image.open(io.BytesIO(item['image_bg_relight'])).convert("RGB")
            return {
                "prompt": prompt,
                "relight_image": image,
                "ori_img": image_ori,
                "bg_img": bg_img
            }
        except Exception as e:
            print(f"SampleDecoder error: {e}")
            return None

class RelightDataset(KVDataset):
    def __init__(
        self,
        paths=[
            "hdfs://harunafr/home/byte_voyager_fr/user/jinqi.xiao/data/relighting_bg/cosmicmanhq_l3single",
            "hdfs://harunafr/home/byte_voyager_fr/user/jinqi.xiao/data/relighting_bg/subject200k", 
           ],
        rank=0,
        world_size=1,
        shuffle=False,
        sample_decoder=SampleDecoder(),
        image_transform=T.Compose([
            AdaptiveResizeMultipleOf(max_size=512, multiple_of=16),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]),
    ):
        super().__init__(paths, rank, world_size, shuffle)
        self.sample_decoder = sample_decoder
        self.image_transform = image_transform
        self._length = 119129 
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
                # print(sample["relight_image"].size, sample["ori_img"].size,sample["bg_img"].size,)
                if self.image_transform:
                    sample["relight_image"] = self.image_transform(sample["relight_image"])
                    sample["ori_img"] = self.image_transform(sample["ori_img"])
                    sample["bg_img"] = self.image_transform(sample["bg_img"])
                yield sample
            except Exception as ex:
                print(f"Error: {ex}")
                continue


def prepare_relight_bg_dataloader(args, accelerator):
    train_dataset = RelightDataset(
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size//2,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=pad_collate_fn,
    )
    return train_dataloader

def pad_collate_fn(batch):
    keys = ['relight_image', 'ori_img', "bg_img"]
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

# dataset = RelightDataset()

# for sample in dataset:
#     print(sample["prompt"])
#     import pdb
#     pdb.set_trace()

# train_dataset = RelightDataset(
#         rank=0,
#         world_size=2,
#     )
# train_dataloader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=2,
#         num_workers=2,
#         pin_memory=True,
#         collate_fn=pad_collate_fn,
#     )
# # train_dataloader_relit_bg = prepare_relight_bg_dataloader(args, accelerator)
# bg = next(iter(train_dataloader))
# import pdb
# pdb.set_trace()