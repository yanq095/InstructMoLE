# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time
from tqdm import tqdm
import argparse
import math
import random

import torch
import torch.distributed as dist
from diffusers import FluxKontextPipeline
from PIL import Image
import os
from src.pipeline_flux_kontext import FluxKontextPipeline as MoePipeline
from src.transformer_flux_kontext import FluxTransformer2DModel
from src.lora_moe import (
    convert_to_lora_moe,
    load_experts_weights,
)
from eval.data_utils import json_load, image_grid, get_rank_and_worldsize
import json
from train_kontext import parse_target_modules
# ckpt = "/opt/tiger/efficient_ai/UniCombine/output/train_result/moe_V2.5_typeS_tokenD/checkpoint-20000/"
ckpt = os.getenv("CKPT")


def get_lora_pipe():
    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path="models/black-forest-labs/FLUX.1-Kontext-dev",
        subfolder="transformer",
    ).to(dtype=torch.bfloat16)
    transformer.load_lora_adapter(
        os.path.join(ckpt, "pytorch_lora_weights.safetensors"),
        use_safetensors=True,
    )
    pipe = MoePipeline.from_pretrained(
        "models/black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
        transformer=transformer,
    ).to(dtype=torch.bfloat16)
    return pipe


def get_moe_pipe():
    with open(os.path.join(ckpt, 'train_config.json'), 'r') as f:
        config = json.load(f)
    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path="models/black-forest-labs/FLUX.1-Kontext-dev",
        subfolder="transformer",
    ).to(dtype=torch.bfloat16)

    transformer.load_lora_adapter(
        os.path.join(ckpt, "pytorch_lora_weights.safetensors"),
        use_safetensors=True,
    )
    moe_config = config["moe_config"]
    target_modules = parse_target_modules(moe_config["target_modules"], transformer)
    convert_to_lora_moe(transformer, moe_config, target_modules)

    if os.path.exists(os.path.join(ckpt, "condition_type_embedder.pt")):
        transformer.add_type_embedding(
            os.path.join(ckpt, "condition_type_embedder.pt"),
        )
    load_experts_weights(
        transformer,
        os.path.join(ckpt, "mole_experts.pt"),
    )
    # print(transformer)

    pipeline = MoePipeline.from_pretrained(
        "models/black-forest-labs/FLUX.1-Kontext-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to(dtype=torch.bfloat16)
    return pipeline


import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--condition_size", type=int, default=128)
    parser.add_argument("--save_name", type=str, default="../output/xvers")
    parser.add_argument("--test_list_name", type=str, default="base_test_list_200")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)

    local_rank, global_rank, world_size = get_rank_and_worldsize()
    print(
        f"local_rank={local_rank}, global_rank={global_rank}, world_size={world_size}"
    )
    torch.cuda.set_device(local_rank)

    dtype = torch.bfloat16
    device = "cuda"
    config_path = args.config_name

    num_images = 4
    save_dir = args.save_name

    # pipe = get_lora_pipe().to(device)
    pipe = get_moe_pipe().to(device)
    # pipe = FluxKontextPipeline.from_pretrained("models/black-forest-labs/FLUX.1-Kontext-dev").to(device).to(dtype=torch.bfloat16)
    if "py" in args.test_list_name:
        test_list = globals()[args.test_list_name.split("_py")[0]]
        test_list = test_list[5:11] + test_list[17:23]  # TODO only for debug
    else:
        test_list = json_load(f"eval/tools/{args.test_list_name}.json", "utf-8")

    num_samples = len(test_list)
    num_ranks = world_size
    assert local_rank == global_rank
    if world_size > 1:
        num_per_rank = math.ceil(num_samples / num_ranks)
        test_list_indices = list(range(num_samples))
        random.seed(0)
        random.shuffle(test_list_indices)
        local_test_list_indices = test_list_indices[
            local_rank * num_per_rank : (local_rank + 1) * num_per_rank
        ]
        print(f"[worker {local_rank}] got {len(local_test_list_indices)} local samples")
    os.makedirs(save_dir, exist_ok=True)

    for i in tqdm(local_test_list_indices):
        test_sample = test_list[i]
        prompt_name = test_sample["prompt"][:40].replace(" ", "_")
        save_path = f"{save_dir}/{i}_{prompt_name}.png"
        if os.path.exists(save_path):
            print(f"文件 {save_path} 已存在，跳过保存")
            continue
        inputs = test_sample["modulation"][0]["src_inputs"]
        condition_imgs = []
        prompt = test_sample["prompt"]
        # ids = [" of the first image", " of the second image", " of the third image", " of the fourth image"]
        for i, inp in enumerate(inputs):
            condition_imgs.append(Image.open(inp["image_path"]).convert("RGB"))
        #     prompt.replace(inp["caption"], inp["caption"]+ids[i])
        # image = image_grid(condition_imgs, 1, len(condition_imgs))
        # image = pipe(
        #     image=image,
        #     prompt=prompt,
        #     num_images_per_prompt = num_images,
        #     guidance_scale=2.5,
        #     # num_inference_steps=25,
        # ).images
        image = pipe(
            images=condition_imgs[:1],
            conds=condition_imgs[1:],
            prompt=prompt,
            num_images_per_prompt=num_images,
            guidance_scale=2.5,
            # num_inference_steps=25,
        ).images
        print(f"{test_sample['prompt']}")
        if isinstance(image, list):
            image = image_grid(image, len(image) // 2, 2)
        image.save(save_path)
        print(f"save results {i} to: {save_path}")
        del image


if __name__ == "__main__":
    main()
