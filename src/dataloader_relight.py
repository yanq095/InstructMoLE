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

def generate_relighting_prompt(style_prompt: str, source_direction: str) -> str:
    templates = [
        "Relight with: {style}",
        "Lighting style: {style}",
        "Apply '{style}' lighting",
        "Use '{style}' lighting on this picture",
        "Adjust the lighting of this image to create a '{style}' atmosphere",
        "Change the lighting to match this description: {style}",
        "Simulate a '{style}' lighting environment for this image",
        "Relight this image to evoke a '{style}' mood"
    ]
    # --- 2. 选择并填充模板 ---
    template = random.choice(templates)
    base_prompt = template.format(style=style_prompt)
    # --- 3. 定义并选择方向描述 ---
    direction_text = ""
    if source_direction != "None":
        # 长句描述
        long_phrases = {
            "Left Light": [", with the main light source coming from the left", ", where the light shines from the left"],
            "Right Light": [", with the main light source coming from the right", ", where the light shines from the right"],
            "Top Light": [", using top-down lighting", ", with the light source positioned above"],
            "Bottom Light": [", with the light illuminating from below", ", using bottom-up lighting"]
        }
        # 短关键词描述
        short_phrases = {
            "Left Light": [", left light", ", light source: left", " (left light)"],
            "Right Light": [", right light", ", light source: right", " (right light)"],
            "Top Light": [", top light", ", light source: top", " (top light)"],
            "Bottom Light": [", bottom light", ", light source: bottom", " (bottom light)"]
        }
        # 随机决定使用长描述还是短描述 (这里设为 50/50 的概率)
        if random.random() < 0.5:
            direction_text = random.choice(short_phrases[source_direction])
        else:
            direction_text = random.choice(long_phrases[source_direction])
    # --- 4. 组合最终指令并添加句号 ---
    final_prompt = base_prompt + direction_text + "."
    return final_prompt


class SampleDecoder:
    def __call__(self, item):
        try:
            image = Image.open(io.BytesIO(item['image'])).convert("RGB")
            cond_img = "image_ori"
            if random.random() < 0.5:
                cond_img = "image_rmbg"
            cond_img = Image.open(io.BytesIO(item[cond_img])).convert("RGB")
            prompt = generate_relighting_prompt(item["relighting_prompt"], item["light_source"])
            
            return {
                "prompt": prompt,
                "relight_image": image,
                "ori_img": cond_img,
            }
        except Exception as e:
            print(f"SampleDecoder error: {e}")
            return None

class RelightDataset(KVDataset):
    def __init__(
        self,
        paths=[
            "hdfs://harunafr/home/byte_voyager_fr/user/jinqi.xiao/data/relighting/subject200k_v2", 
            "hdfs://harunafr/home/byte_voyager_fr/user/jinqi.xiao/data/relighting/cosmicmanhq_l3single_v2"],
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
        self._length = 100000 
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
                    sample["relight_image"] = self.image_transform(sample["relight_image"])
                    sample["ori_img"] = self.image_transform(sample["ori_img"])
                yield sample
            except Exception as ex:
                print(f"Error: {ex}")
                continue


def prepare_relight_dataloader(args, accelerator):
    train_dataset = RelightDataset(
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=pad_collate_fn,
    )
    return train_dataloader

def pad_collate_fn(batch):
    keys = ['relight_image', 'ori_img']
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