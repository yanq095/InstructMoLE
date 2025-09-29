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
from glob import glob
import argparse
import math
import random
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torch.distributed as dist


from eval.data_utils import get_train_config, get_rank_and_worldsize
from eval.data_utils import pad_to_square, pad_to_target, json_dump, json_load, split_grid, image_grid, pil2tensor
import shutil
from eval.tools.florence_sam import ObjectDetector
from eval.tools.dino import DINOScore

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
    detector_model = ObjectDetector(device)
    dino_model = DINOScore(device)
    
    test_list = json_load(f"eval/tools/{args.test_list_name}.json", 'utf-8')
    images = list(glob(f"{args.input_dir}/*.png"))
    print(len(test_list), len(images))
    assert len(test_list) == len(images)
    
    num_samples = len(test_list)
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
    temp_dir = os.path.join("/tmp", f"eval_ip_temp_{run_name}")

    if is_main_process:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

    rank_json = {}
    with torch.no_grad():
        for i in tqdm(local_test_list_indices):
            test_sample = test_list[i]
            real_paths, real_ips, real_names, real_labels = [], [], [], []
            for j, x in enumerate(test_sample["modulation"][0]["src_inputs"]):
                img_path = x["image_path"]
                name = "_".join(img_path.split("/")[-2:])
                label = test_sample["modulation"][0]["use_words"][j][1]
                
                if not name.startswith("human"):
                    real_paths.append(img_path)
                    real_ips.append(Image.open(img_path).convert("RGB"))
                    real_names.append(name)
                    real_labels.append(label)

            gen_img_path = list(filter(lambda x: x.split("/")[-1].split("_")[0] == str(i), images))[0]
            rank_json[i] = []
            
            for j, gen_img in enumerate(split_grid(Image.open(gen_img_path))):
                rank_json[i].append({})
                if len(real_names) > 0:
                    
                    for real_ip, real_name, real_label in zip(real_ips, real_names, real_labels):
                        found_ips = detector_model.get_instances(gen_img, real_label, min_size=gen_img.size[0]//20)[:3]
                        found_ips = [pad_to_square(x) for x in found_ips]
                        score = 0
                        if len(found_ips) > 0:
                            score = max([dino_model(real_ip, ip) for ip in found_ips])
                    
                        rank_json[i][j][real_name] = score
                    
    json_dump(rank_json, f"{temp_dir}/scores_{global_rank}.json", "utf-8")

    if is_main_process:
        while len(glob(f"{temp_dir}/scores_*.json")) < world_size:
            time.sleep(5)
        time.sleep(5) # wait for the file writting to be finished
        merged_json = {}
        ip_scores = defaultdict(list)
        all_scores = []
        for rank_path in glob(f"{temp_dir}/scores_*.json"):
            rank_json = json_load(rank_path, "utf-8")
            merged_json.update(rank_json)
            for i in rank_json:
                grid_json = rank_json[i]
                for img_json in grid_json:
                    for ip_name, ip_score in img_json.items():
                        ip_scores[ip_name].append(ip_score)
        
        for ip_name in ip_scores:
            all_scores += ip_scores[ip_name]
            ip_scores[ip_name] = np.mean(ip_scores[ip_name])
            print(ip_name, ip_scores[ip_name])

        json_dump(merged_json, f"{args.input_dir}/ip_scores_{run_name}.json", "utf-8")
        final_ip_score = np.mean(all_scores)
        lines_to_write = [
            f"IP Score: {final_ip_score:.2f}\n"
        ]
        print(lines_to_write[0])
        for ip_name, score in ip_scores.items():
            lines_to_write.append(f"{ip_name}: {score:.2f}\n")

        with open(f"{args.input_dir}/ip_scores_{run_name}.txt", "w") as f:
            f.writelines(lines_to_write)

        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
