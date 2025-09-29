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

instruct = [
    "Use the reference face images only for the identity of the people. ",
    "Generate an image using the face from the reference images. ",
    "Keep identities from the face images; follow the text prompt for everything else."
    "Given the reference face images, generate an image."
]

def generate_prompt(prompt):
    return random.choice(instruct) + prompt


class SampleDecoder:

    def __init__(self, cond_num=2) -> None:
        self.cond_num = cond_num

    def __call__(self, item):
        # if self.cond_num <= 3:
        #     if item["person_cnt"] != self.cond_num:
        #         return None
        # else:
        if item["person_cnt"] < self.cond_num:
            return None
        persons = []
        try:
            image = Image.open(io.BytesIO(item["image"])).convert("RGB")
            # for p in item["persons"]:
            for p in item["faces"]:
                persons.append(Image.open(io.BytesIO(p)).convert("RGB"))
            if random.random() < 0.5:
                prompt = item["blip2_opt"]
            else:
                prompt = item["InternVL2_caption"]
            return {
                "text": generate_prompt(prompt),
                "image": image,
                "persons": persons[: self.cond_num],
            }
        except Exception as e:
            print(f"SampleDecoder error: {e}")
            return None


class MultiPersonDataset(KVDataset):

    def __init__(
        self,
        paths=[
            "hdfs://harunafr/home/byte_voyager_fr/user/jinqi.xiao/data/multi_person",
        ],
        rank=0,
        world_size=1,
        shuffle=False,
        cond_num=2,
        image_transform=T.Compose(
            [
                AdaptiveResizeMultipleOf(max_size=512, multiple_of=16),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        ),
    ):
        super().__init__(paths, rank, world_size, shuffle)
        self.sample_decoder = SampleDecoder(cond_num)
        self.image_transform = image_transform
        self._length = 1007934
        self.cond_num = cond_num
        self.face_transform = T.Compose(
            [
                AdaptiveResizeMultipleOf(max_size=128, multiple_of=16),
                PasteToCenterCanvas(canvas_size=(128, 128), fill=(0, 0, 0)),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )

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
                    persons = sample["persons"]
                    conds = []
                    for p in persons:
                        conds.append(self.face_transform(p))
                    conds = torch.stack(conds, dim=0)
                    sample["persons"] = conds
                yield sample
            except Exception as ex:
                print(f"Error: {ex}")
                continue


def pad_collate_fn(batch):
    keys = ["image", "persons"]
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


def prepare_multi_person_dataloader(args, cond_num, accelerator):
    train_dataset = MultiPersonDataset(
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        cond_num=cond_num,
    )
    bsz = max(1, args.train_batch_size // cond_num)
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


# dataset = MultiPersonDataset(cond_num=4)

# for sample in dataset:
#     print(sample["text"])
#     print(sample["persons"].shape)
