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
from typing import List, Optional


def _format_item_descriptions_to_string(full_descriptions: List[str]) -> str:
    """
    一个辅助函数，用于将完整的物品描述列表格式化为自然的英文短语。
    """
    if not full_descriptions:
        return ""
    # 为每个物品描述加上 "the"，除非它已经有了冠词 (a/an/the)
    formatted_items = []
    for desc in full_descriptions:
        first_word = desc.split()[0].lower()
        if first_word in ["a", "an", "the"]:
            formatted_items.append(desc)
        else:
            formatted_items.append(f"the {desc}")
    if len(formatted_items) == 1:
        return formatted_items[0]
    elif len(formatted_items) == 2:
        return " and ".join(formatted_items)
    else:
        return ", ".join(formatted_items[:-1]) + ", and " + formatted_items[-1]


def generate_prompt(
    item_names: Optional[List[str]] = None, source_probability: float = 0.5
):
    """
    根据一个纯物品名称的列表，为虚拟试穿生成多样化、直接的 prompt。
    - 人物固定在第一张图。
    - 物品从第二张图开始按顺序分配。
    - 如果 `item_names` 为空或包含5个及以上物品，则生成“穿上所有服装”的通用 prompt。
    - 对于1-4件物品，会以一定的概率为每件物品添加图片来源。
    - 安全网：对于多件物品的 prompt，保证至少有一件会显示来源，以防歧义。
    :param item_names: 一个可选的字符串列表，只包含物品名称。例如: ["shoes", "necklace"]
    :param source_probability: 一个0到1之间的浮点数，表示为物品添加来源的概率。默认为 0.5 (50%)。
    :return: 随机一个 prompt 。
    """
    target_person = "the person from the first image"
    target_person_capitalized = "The person from the first image"
    # --- 场景一: 没有指定物品名，或物品数量过多(>=5)，触发通用模式 ---
    if not item_names or len(item_names) >= 5:
        generic_term = "the complete outfit from the other images"
        prompts = [
            f"Dress {target_person} in {generic_term}.",
            f"Put all the clothing items from the other images on {target_person}.",
            f"{target_person_capitalized} wearing the full set of clothes from the other pictures.",
            f"Virtual Try-On: Model is {target_person}, garments are everything from the other images.",
            f"Combine all garments from the secondary images and fit them onto {target_person}.",
            f"The subject from the first image wearing the entire outfit from the other photos.",
        ]
        return random.choice(prompts)
    # --- 场景二: 指定了1-4件具体物品 ---
    chosen_descriptions = []
    full_source_descriptions = []  # 用于安全网
    image_map = {2: "second", 3: "third", 4: "fourth", 5: "fifth"}
    # 遍历物品名，并根据概率决定是否添加来源
    add_source = random.random() < source_probability
    for i, name in enumerate(item_names, start=2):
        if add_source:
            full_desc = f"{name} from the {image_map[i]} image"
        else:
            full_desc = name
        full_source_descriptions.append(full_desc)
        if add_source:
            chosen_descriptions.append(full_desc)
        else:
            chosen_descriptions.append(name)
    # 安全网：如果物品多于一个，且随机后没有任何一个物品指定来源，则强制为第一个物品指定来源
    has_source = any("from the" in desc for desc in chosen_descriptions)
    if len(chosen_descriptions) > 1 and not has_source:
        chosen_descriptions[0] = full_source_descriptions[0]
    # 将物品描述列表格式化为单个字符串
    if add_source and random.random() < 0.3:
        random.shuffle(chosen_descriptions)
    items_string = _format_item_descriptions_to_string(chosen_descriptions)
    # 定义一组直接、陈述性的 prompt 模板
    prompts = [
        f"Put {items_string} on {target_person}.",
        f"Dress {target_person} with {items_string}.",
        f"{target_person_capitalized} wearing {items_string}.",
        f"An image of {target_person} wearing {items_string}.",
        f"A depiction of {target_person}, adorned with {items_string}.",
        f"Virtual Try-On: Model is {target_person}. Garment(s): {items_string}.",
        f"Render {target_person} featuring {items_string}.",
        f"The subject from the first image, now wearing {items_string}.",
    ]
    return random.choice(prompts)


class SampleDecoder:

    def __init__(self, cond_num=1) -> None:
        self.cond_num = cond_num

    def __call__(self, item):
        cloths = []
        name = []
        mask = None
        try:
            openpose_img = Image.open(io.BytesIO(item["pose_image"])).convert("RGB")
            if self.cond_num == 1:
                # use original data
                if random.random() < 0.3:
                    image = Image.open(io.BytesIO(item["ori_image"])).convert("RGB")
                    cloth = Image.open(io.BytesIO(item["ori_cloth"])).convert("RGB")
                    model = Image.open(io.BytesIO(item["try_on_images"][0])).convert(
                        "RGB"
                    )
                else:
                    # use generate data
                    idx = random.randint(0, len(item["try_on_images"]) - 1)
                    # if random.random() < 0.5:
                    #     mask = Image.open(io.BytesIO(item["mask_images"][idx])).convert(
                    #         "RGB"
                    #     )
                    if idx == 0:
                        image = Image.open(
                            io.BytesIO(item["try_on_images"][0])
                        ).convert("RGB")
                        model = Image.open(io.BytesIO(item["ori_image"])).convert("RGB")
                        cloth = Image.open(io.BytesIO(item["cloth_images"][0])).convert(
                            "RGB"
                        )
                    else:
                        name = [item["cloth_name"][idx]]
                        image = Image.open(
                            io.BytesIO(item["try_on_images"][idx])
                        ).convert("RGB")
                        model = Image.open(
                            io.BytesIO(item["try_on_images"][idx - 1])
                        ).convert("RGB")
                        cloth = Image.open(
                            io.BytesIO(item["cloth_images"][idx])
                        ).convert("RGB")
                cloths.append(cloth)
            else:
                image = Image.open(
                    io.BytesIO(item["try_on_images"][self.cond_num - 1])
                ).convert("RGB")
                model = Image.open(io.BytesIO(item["ori_image"])).convert("RGB")
                for i in range(self.cond_num):
                    cloth = Image.open(io.BytesIO(item["cloth_images"][i])).convert(
                        "RGB"
                    )
                    cloths.append(cloth)
                    name.append(item["cloth_name"][i])
            # if mask is not None:
            #     return {
            #         "text": generate_prompt(name),
            #         "image": image,
            #         "model": model,
            #         "cloth": cloths,
            #         "openpose_img": openpose_img,
            #         "mask": mask,
            #     }
            # else:
            return {
                "text": generate_prompt(name),
                "image": image,
                "model": model,
                "cloth": cloths,
                "openpose_img": openpose_img,
            }
        except Exception as e:
            print(f"SampleDecoder error: {e}")
            return None


class TryonDataset(KVDataset):

    def __init__(
        self,
        paths=[
            "hdfs://harunafr/home/byte_voyager_fr/user/jinqi.xiao/data/tryon/tryon_part1",
            "hdfs://harunafr/home/byte_voyager_fr/user/jinqi.xiao/data/tryon/tryon_part2",
        ],
        rank=0,
        world_size=1,
        shuffle=False,
        cond_num=1,
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
        self._length = 20034
        self.cond_num = cond_num
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
                    sample["model"] = self.image_transform(sample["model"])
                    cloths = sample["cloth"]
                    conds = []
                    for cloth in cloths:
                        conds.append(self.image_transform(cloth))
                    conds = torch.stack(conds, dim=0)

                    if "mask" in sample:
                        mask = self.image_transform(sample["mask"])
                        mask = (mask > 0.5).float()
                        sample["mask"] = mask
                    sample["openpose_img"] = self.image_transform(
                        sample["openpose_img"]
                    )
                    sample["cloth"] = conds
                yield sample
            except Exception as ex:
                print(f"Error: {ex}")
                continue


def pad_collate_fn(batch):
    keys = ["image", "model", "cloth", "openpose_img"]
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


def prepare_tryon_dataloader(args, cond_num, accelerator):
    train_dataset = TryonDataset(
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        cond_num=cond_num,
    )
    bsz = max(1, args.train_batch_size//cond_num)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=bsz,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=pad_collate_fn,
    )
    return train_dataloader


# dataset = TryonDataset(cond_num=4)

# for sample in dataset:
#     print(sample["text"])
#     print(sample["cloth"].shape)
