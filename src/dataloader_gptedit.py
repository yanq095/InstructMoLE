from PIL import Image
import io
from src.data.dataset_square import KVDataset
import json
import bson
from src.adaptive_resize import AdaptiveResizeMultipleOf
import torch
import torchvision.transforms as T
from dataloader import KVReader
import torch.nn.functional as F

class SampleDecoder:
    def __call__(self, item):
        try:
            src_img = Image.open(io.BytesIO(item['input'])).convert("RGB")
            edited_img = Image.open(io.BytesIO(item['output'])).convert("RGB")

            return {
                "prompt": item.get("instruction"),
                "task": item.get("task"),
                "src_img": src_img,
                "edited_img": edited_img,
            }
        except Exception as e:
            print(f"SampleDecoder error: {e}")
            return None


class GptEditDataset(KVDataset):
    def __init__(
        self,
        paths=["hdfs://harunafr/home/byte_voyager_fr/user/jinqi.xiao/data/gpt-edit-kv/hqedit", "hdfs://harunafr/home/byte_voyager_fr/user/jinqi.xiao/data/gpt-edit-kv/ultraedit", "hdfs://harunafr/home/byte_voyager_fr/user/jinqi.xiao/data/gpt-edit-kv/omniedit"],
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
        self._length = 930000
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
                    sample["src_img"] = self.image_transform(sample["src_img"])
                    sample["edited_img"] = self.image_transform(sample["edited_img"])
                yield sample
            except Exception as ex:
                print(f"Error: {ex}")
                continue

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

def prepare_gptedit_dataloader(args, accelerator):
    train_dataset_style = GptEditDataset(
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_style,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=pad_collate_fn,
    )
    return train_dataloader


# dataset = GptEditDataset()

# for sample in dataset:
#     print(sample)
#     break