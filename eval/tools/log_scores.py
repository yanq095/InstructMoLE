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

from PIL import Image
import os
from glob import glob
from tqdm import tqdm
import random
import argparse
import time
from eval.data_utils import json_dump, json_load


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../examples")
    parser.add_argument("--test_list_name", type=str, default="base_test_list_200")
    args = parser.parse_args()
    return args

def read_txt_first_line(file_path):
    with open(file_path, "r") as f:
        return f.readline().strip()

def read_txt_second_line(file_path):
    with open(file_path, "r") as f:
        f.readline()
        return f.readline().strip()

if __name__ == '__main__':
    args = parse_args()
    
    final_score = {}

    files = sorted(list(glob(f"{args.input_dir}/dpg_scores_*.txt")))
    if len(files) > 0:
        score = read_txt_first_line(files[-1]).split(":")[-1]
        final_score["dpg"] = score
        
    files = sorted(list(glob(f"{args.input_dir}/id_scores_*.txt")))
    if len(files) > 0:
        score = read_txt_first_line(files[-1]).split(":")[-1]
        final_score["id"] = score
        
    files = sorted(list(glob(f"{args.input_dir}/ip_scores_*.txt")))
    if len(files) > 0:
        score = read_txt_first_line(files[-1]).split(":")[-1]
        final_score["ip"] = score
    
    files = sorted(list(glob(f"{args.input_dir}/clip_scores_*.txt")))
    if len(files) > 0:
        score_i = read_txt_first_line(files[-1]).split(":")[-1]
        score_t = read_txt_second_line(files[-1]).split(":")[-1]
        final_score["clip_i"] = score_i
        final_score["clip_t"] = score_t
    
    files = sorted(list(glob(f"{args.input_dir}/aes_scores_*.txt")))
    if len(files) > 0:
        score = read_txt_first_line(files[-1]).split(":")[-1]
        final_score["aes"] = score
    
    if "dpg" in final_score and "id" in final_score and "ip" in final_score and "aes" in final_score:
        score = (float(final_score["dpg"]) + float(final_score["id"]) + float(final_score["ip"]) + float(final_score["aes"])) / 4
        final_score["avg"] = score
    
    run_name = time.strftime("%Y%m%d-%H%m")
    json_dump(final_score, f"{args.input_dir}/idip_all_scores_{run_name}.json", "utf-8")
    print(final_score)
        

