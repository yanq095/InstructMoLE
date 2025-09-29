from PIL import Image
import io
from src.data.dataset_square import KVDataset
import json
import bson
from src.adaptive_resize import AdaptiveResizeMultipleOf
import torch
import torchvision.transforms as T
from dataloader import KVReader

class SampleDecoder:
    def __call__(self, item):
        objs = []
        try:
            image = Image.open(io.BytesIO(item["result_image"])).convert("RGB")
            # for p in item["persons"]:
            for p in item["subjects"]:
                objs.append(Image.open(io.BytesIO(p)).convert("RGB"))
            return {
                "text": item['scene_prompt'],
                "image": image,
                "objects": objs,
            }
        except Exception as e:
            print(f"SampleDecoder error: {e}")
            return None


class MultiObjDataset(KVDataset):

    def __init__(
        self,
        paths=[
            "hdfs://harunafr/home/byte_voyager_fr/user/jinqi.xiao/data/nano_banana",
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
        self._length = 62311

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
                    conds = []
                    for p in sample["objects"]:
                        conds.append(self.image_transform(p))
                    sample["objects"] = torch.stack(conds, dim=0)
                yield sample
            except Exception as ex:
                print(f"Error: {ex}")
                continue


def prepare_multi_obj_dataloader(args, accelerator):
    train_dataset = MultiObjDataset(
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    # if cond_num > 10:
    #     bsz = 1
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    return train_dataloader


# dataset = MultiPersonDataset()

# for sample in dataset:
#     print(sample["text"])
#     print(sample["objects"].shape)
