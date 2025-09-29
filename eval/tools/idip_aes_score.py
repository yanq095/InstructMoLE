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

from tqdm import tqdm
from glob import glob
import argparse
import math
import random
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from eval.data_utils import get_rank_and_worldsize, json_dump, json_load
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
import shutil
from pathlib import Path
import os
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../examples")
    parser.add_argument("--test_list_name", type=str, default="base_test_list_200")
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    print(args)
    
    local_rank, global_rank, world_size = get_rank_and_worldsize()
    print(f"local_rank={local_rank}, global_rank={global_rank}, world_size={world_size}")
    is_local_main_process = local_rank == 0
    is_main_process = global_rank == 0
    torch.cuda.set_device(local_rank)
    
    dtype = torch.bfloat16
    device = "cuda"
    
    run_name = time.strftime("%m%d_$H")
    model, preprocessor = convert_v2_5_from_siglip(
        predictor_name_or_path="models/aesthetic_predictor_v2_5.pth",
        encoder_model_name = "models/google/siglip-so400m-patch14-384",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = model.to(torch.bfloat16).to(f"cuda:{local_rank}")
    
    test_list = json_load(f"eval/tools/{args.test_list_name}.json", 'utf-8')
    images = list(glob(f"{args.input_dir}/*.png"))
    
    num_samples = min(len(test_list), len(images))
    num_ranks = world_size
    assert local_rank == global_rank
    if world_size > 1:
        num_per_rank = math.ceil(num_samples / num_ranks)
        test_list_indices = list(range(num_samples))
        random.seed(0)
        random.shuffle(test_list_indices)
        local_test_list_indices = test_list_indices[local_rank*num_per_rank:(local_rank+1)*num_per_rank]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank % 8)
        print(f"[worker {local_rank}] got {len(local_test_list_indices)} local samples")

    run_name = time.strftime("%Y%m%d-%H")
    temp_dir = os.path.join("/tmp", f"eval_temp_{run_name}")

    if is_main_process:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

    score_json = {}
    with torch.no_grad():
        for i in tqdm(local_test_list_indices):
            test_sample = test_list[i]
            image_path = list(filter(lambda x: x.split("/")[-1].split("_")[0] == str(i), images))[0]
            
            SAMPLE_IMAGE_PATH = Path(image_path)
            image = Image.open(SAMPLE_IMAGE_PATH).convert("RGB")
            pixel_values = (
                preprocessor(images=image, return_tensors="pt")
                .pixel_values.to(torch.bfloat16)
                .to(f"cuda:{local_rank}")
            )

            with torch.inference_mode():
                score = model(pixel_values).logits.squeeze().float().cpu().numpy()
            
            score_json[i] = float(score)*10

    json_dump(score_json, f"{temp_dir}/scores_{global_rank}.json", "utf-8")

    if is_main_process:
        # 等待所有进程完成文件写入
        all_files_written = False
        max_retries = 10
        retry_count = 0
        while not all_files_written and retry_count < max_retries:
            try:
                if len(glob(f"{temp_dir}/scores_*.json")) == world_size:
                    all_files_written = True
                    time.sleep(5)  # 确保文件写入完成
                else:
                    time.sleep(5)
                    retry_count += 1
            except Exception as e:
                print(f"Error checking files: {e}")
                time.sleep(5)
                retry_count += 1

        if not all_files_written:
            print("Not all score files were written within the timeout.")
            return

        merged_json = {}
        prompt_scores = {}
        scores = []
        for rank_path in glob(f"{temp_dir}/scores_*.json"):
            try:
                rank_json = json_load(rank_path, "utf-8")
                merged_json.update(rank_json)
                for i in rank_json:
                    score = rank_json[i]
                    prompt_scores[i] = score
                    scores.append(score)
            except Exception as e:
                print(f"Error loading file {rank_path}: {e}")

        json_dump(merged_json, f"{args.input_dir}/aes_scores_{run_name}.json", "utf-8")
        if scores:
            dpg_score = np.mean(scores)
            lines_to_write = [
                f"AES Score: {dpg_score:.2f}\n"
            ]
            print(lines_to_write[0])
            for i, score in prompt_scores.items():
                lines_to_write.append(f"{i}: {score:.2f}\n")

            with open(f"{args.input_dir}/aes_scores_{run_name}.txt", "w") as f:
                f.writelines(lines_to_write)
        else:
            print("No scores were collected.")

        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
