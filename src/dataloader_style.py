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
        try:
            image = Image.open(io.BytesIO(item['image_after']))
            image2 = Image.open(io.BytesIO(item['image2_after']))
            image_before = Image.open(io.BytesIO(item['image_before']))
            image2_before = Image.open(io.BytesIO(item['image2_before']))
            return {
                "prompt": item.get("prompt"),
                "prompt2": item.get("prompt2"),
                "image": image,
                "image2": image2,
                "image_before": image_before,
                "image2_before": image2_before,
                "style": item.get("style")
            }
        except Exception as e:
            print(f"SampleDecoder error: {e}")
            return None


class StyleDataset(KVDataset):
    def __init__(
        self,
        paths=["hdfs://harunafr/home/byte_voyager_fr/user/jinqi.xiao/data/style_aligned_char_paired"],
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
        self._length = 459000
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
                    sample["image2"] = self.image_transform(sample["image2"])
                    sample["image_before"] = self.image_transform(sample["image_before"])
                    sample["image2_before"] = self.image_transform(sample["image2_before"])
                yield sample
            except Exception as ex:
                print(f"Error: {ex}")
                continue


def prepare_style_dataloader(args, accelerator):
    train_dataset_style = StyleDataset(
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_style,
        batch_size=args.train_batch_size//2,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    return train_dataloader


# dataset = StyleDataset(
#     paths=["hdfs://harunasg/home/byte_pokemon_sg/user/minjin.chong/flux_data_gen/style_aligned"],
#     shuffle=True,
# )

# for sample in dataset:
#     print(sample["prompt"], sample["style"], sample["image"].shape, sample["image2"].shape)
#     break