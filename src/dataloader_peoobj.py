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

prompt_templates = [
    "Add the {obj_key} of the second image to the first image, so that {action}.",
    "With the person from image 1 and the {obj_key} from image 2, generate a new image where {action}.",
    # "The goal is to show the subject from the first photo {action}, using the {obj_key} provided in the second photo.",
    "Take the {obj_key} in the second picture and put it in the first one. {action}.",
    "A photo of the person from the first image, now {action} with the {obj_key} from the second image.",
    "Person from image 1, {obj_key} from image 2. Action: {action}.",
    # "Task: Edit the primary image. Subject: the person. Object: the {obj_key} from the secondary image. Desired result: {action}.",
    # "Make the person in the first image {action}. The object they should use is the {obj_key} from the second image.",
    "Referring to the person in the first image and the {obj_key} of the second image: show them {action}.",
]


class SampleDecoder:
    def __call__(self, item):
        try:
            image = Image.open(io.BytesIO(item["image"])).convert("RGB")
            person = Image.open(io.BytesIO(item["person"])).convert("RGB")
            obj = (
                Image.open(io.BytesIO(item["object_front_view"]))
                .resize((512, 512))
                .convert("RGB")
            )
            obj_key = item["object_keyword"]
            prompt = item.get("caption")

            if random.random() < 0.35:
                prompt = item.get("InternVL_26B_caption")
                if random.random() < 0.5:
                    replacement_text = f"{obj_key} of the second image"
                    prompt = prompt.replace("{obj_key}", replacement_text)
            else:
                template = random.choice(prompt_templates)
                prompt = template.format(obj_key=obj_key, action=prompt)
            return {
                "prompt": prompt,
                "image": image,
                "person": person,
                "object": obj,
            }
        except Exception as e:
            print(f"SampleDecoder error: {e}")
            return None


class PeoObjDataset(KVDataset):
    def __init__(
        self,
        paths=[
            "hdfs://harunafr/home/byte_voyager_fr/user/jinqi.xiao/data/hoi/alamy_l3single_internvl2_blip2_arcface_id_embed_bbox_buckets_diff_view_v3"
        ],
        rank=0,
        world_size=1,
        shuffle=False,
        sample_decoder=SampleDecoder(),
        image_transform=T.Compose(
            [
                AdaptiveResizeMultipleOf(max_size=512, multiple_of=16),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        ),
    ):
        super().__init__(paths, rank, world_size, shuffle)
        self.sample_decoder = sample_decoder
        self.image_transform = image_transform
        self._length = 867243
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
                    sample["person"] = self.image_transform(sample["person"])
                    sample["object"] = self.image_transform(sample["object"])
                yield sample
            except Exception as ex:
                print(f"Error: {ex}")
                continue


def pad_collate_fn(batch):
    keys = ["image", "person", "object"]
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


def prepare_po_dataloader(args, accelerator):
    train_dataset = PeoObjDataset(
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


# dataset = PeoObjDataset()

# for sample in dataset:
#     print(sample["prompt"], sample["image"].shape, sample["object"].shape)
#     import pdb
#     pdb.set_trace()
