import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from datasets import load_dataset
from src.adaptive_resize import AdaptiveResizeMultipleOf
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F

image_transform = T.Compose(
    [
        AdaptiveResizeMultipleOf(max_size=512, multiple_of=16),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]
)


class MultiSubDataset(Dataset):
    def __init__(
        self,
    ):
        super().__init__()
        dataset = load_dataset("guozinan/MUSAR-Gen")
        self.base_dataset = dataset["train"]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        img = image_transform(item["tgt_img"].convert("RGB"))
        conds = [image_transform(item["cond_img_0"].convert("RGB")), image_transform(item["cond_img_1"].convert("RGB"))]
        prompt = item["prompt"]
        conds = torch.stack(conds, dim=0)
        return {
            "pixel_values": img,
            "condition_latents": conds,
            "descriptions": prompt,
            "condition_types": "subject"
        }


def pad_collate_fn(batch):
    keys = ["pixel_values", "condition_latents"]
    batch_out = {}
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
    for k in batch[0]:
        if k not in keys:
            batch_out[k] = [d[k] for d in batch]
    return batch_out


def prepare_multi_sub_dataloader(args, accelerator):
    train_dataset = MultiSubDataset()

    train_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        drop_last=True,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        # collate_fn=pad_collate_fn,
    )
    return train_dataloader


