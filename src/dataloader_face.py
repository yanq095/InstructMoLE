import torch
import torch.nn.functional as F
import torchvision.transforms as T

# from torch.distributed.optim import ZeroRedundancyOptimizer
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from diffusers.image_processor import VaeImageProcessor
import PIL
from torch.utils.data import default_collate
from torch.utils.data._utils.collate import default_collate_fn_map
import random
import numpy as np

def collate_customtype_fn(batch, *, collate_fn_map=None):
    return batch


default_collate_fn_map.update({PIL.Image.Image: collate_customtype_fn})
from src.configs.data.bucket.bucket_instantid import create_dataset as create_dataset_bk
from torchvision.transforms import InterpolationMode

# dict_keys(['image', 'face_kps_image', 'id_face_image', 'id_embed', 'text', 'blip2_opt_t5_bf16', 'index_file', 'key', 'original_size', 'crop_coords_top_left', 'target_size'])
from PIL import Image
from src.adaptive_resize import AdaptiveResizeMultipleOf, PasteToCenterCanvas

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def prepare_face_dataloader(args, accelerator, use_spmv=False):
    if not use_spmv:
        train_dataset_bk = create_dataset_bk(
            paths=[
                "hdfs://harunafr/home/byte_voyager_fr/user/jinqi.xiao/data/cosmicmanhq_l3single_internvl2_blip2_arcface_id_embed_train_buckets"
            ],
            image_transform=T.Compose(
                [
                    AdaptiveResizeMultipleOf(max_size=512, multiple_of=16),
                    T.ToTensor(),
                    T.Normalize([0.5], [0.5]),
                ]
            ),
            bucket_sharding=True,
            use_resize_random_crop=True,
            batch_size=args.train_batch_size,
            embedding_keys=["id_embed"],
            kps_keys=["kps"],
            short_keys=["prompt"],
            long_keys=["blip2_opt", "InternVL_26B_caption"],
            long_prompt_prob=0.75,
            skip_normalization=True,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
        )
    else:
        # spmv
        train_dataset_bk = create_dataset_bk(
            paths=[
                "hdfs://harunafr/home/byte_voyager_fr/user/jinqi.xiao/data/spmv_format_filter"
            ],
            image_transform=T.Compose(
                [
                    AdaptiveResizeMultipleOf(max_size=512, multiple_of=16),
                    T.ToTensor(),
                    T.Normalize([0.5], [0.5]),
                ]
            ),
            image_embeds_processor=T.Compose(
                [
                    AdaptiveResizeMultipleOf(max_size=512, multiple_of=16),
                    PasteToCenterCanvas(canvas_size=(512, 512), fill=(0, 0, 0)),
                    T.ToTensor(),
                    T.Normalize([0.5], [0.5]),
                ]
            ),
            bucket_sharding=True,
            use_resize_random_crop=True,
            batch_size=args.train_batch_size,
            embedding_keys=["id_embed"],
            kps_keys=["kps"],
            short_keys=["prompt"],
            long_keys=["prompt"],
            long_prompt_prob=0.75,
            skip_normalization=True,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
        )

    g = torch.Generator()
    base_seed = getattr(args, 'seed', 0)
    g.manual_seed(base_seed + accelerator.process_index)

    train_dataloader_bk = torch.utils.data.DataLoader(
        train_dataset_bk,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=default_collate,
        worker_init_fn=seed_worker,
        generator=g, 
    )
    # train_dataloader_iter_bk = iter(train_dataloader_bk)
    return train_dataloader_bk
