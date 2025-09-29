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

import torch
from copy import deepcopy
from collections import defaultdict
import numpy as np
import pandas as pd
import os

class MPLUG(torch.nn.Module):
    def __init__(self, ckpt='damo/mplug_visual-question-answering_coco_large_en', device='gpu'):
        super().__init__()
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        self.pipeline_vqa = pipeline(Tasks.visual_question_answering, model=ckpt, device=device)

    def vqa(self, image, question):
        input_vqa = {'image': image, 'question': question}
        result = self.pipeline_vqa(input_vqa)
        return result['text']


class DPGScore:
    def __init__(self, device):
        self.device = device
        ckpt = os.getenv('DPG_VQA_MODEL_PATH', "models/xingjianleng/mplug_visual-question-answering_coco_large_en")
        self.vqa_model = MPLUG(ckpt, device=self.device)

        
    def __call__(self, image, q_dict):
        VQA = self.vqa_model
        qid2tuple, qid2dependency, qid2question = q_dict['qid2tuple'], q_dict['qid2dependency'], q_dict['qid2question']
        qid2answer = {}
        qid2scores = {}

        for id, question in qid2question.items():
            id = str(id)
            answer = VQA.vqa(image, question)
            qid2answer[id] = answer
            qid2scores[id] = float(answer == 'yes')
                
        average_score_without_dep = sum(qid2scores.values()) / len(qid2scores)
            
        qid2validity = {}
        qid2scores_after_filtering = deepcopy(qid2scores)

        for id, parent_ids in qid2dependency.items():
            id = str(id)
            any_parent_answered_no = False
            for parent_id in parent_ids:
                parent_id = str(parent_id)
                if int(parent_id) == 0:
                    continue
                if parent_id in qid2scores:
                    if qid2scores[parent_id] == 0:
                        any_parent_answered_no = True
                        break
            if any_parent_answered_no:
                qid2scores_after_filtering[id] = 0.0
                qid2validity[id] = False
            else:
                qid2validity[id] = True

        average_score_with_dep = sum(qid2scores_after_filtering.values()) / len(qid2scores)
        return {
            'qid2tuple': qid2tuple,
            'qid2dependency': qid2dependency,
            'qid2question': qid2question,
            'qid2answer': qid2answer,
            'qid2scores': qid2scores,
            'qid2validity': qid2validity,
            'average_score_with_dependency': average_score_with_dep * 100.,
            'average_score_without_dependency': average_score_without_dep * 100.
        }


def prepare_dpg_data(csv_path):
    previous_id = ''
    current_id = ''
    question_dict = dict()
    category_count = defaultdict(int)
    data = pd.read_csv(csv_path)
    for i, line in data.iterrows():
        if i == 0:
            continue

        current_id = line.item_id
        qid = str(line.proposition_id)
        dependency_list_str = line.dependency.split(',')
        dependency_list_int = []
        for d in dependency_list_str:
            d_int = str(d.strip())
            dependency_list_int.append(d_int)

        if current_id == previous_id:
            question_dict[current_id]['qid2tuple'][qid] = line.tuple
            question_dict[current_id]['qid2dependency'][qid] = dependency_list_int
            question_dict[current_id]['qid2question'][qid] = line.question_natural_language
        else:
            question_dict[current_id] = dict(
                qid2tuple={qid: line.tuple},
                qid2dependency={qid: dependency_list_int},
                qid2question={qid: line.question_natural_language})
        
        category = line.question_natural_language.split('(')[0].strip()
        category_count[category] += 1
        
        previous_id = current_id
    return question_dict



if __name__ == "__main__":
    import os
    import time
    import shutil
    import argparse
    from PIL import Image
    from tqdm import tqdm
    from src.train.data.data_utils import split_grid, json_load, json_dump
    from src.train.train_utils import get_train_config, get_rank_and_worldsize
    from src.train.data.validation import *

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_dir", type=str, default="")
        args = parser.parse_args()
        return args

    args = parse_args()

    local_rank, global_rank, world_size = get_rank_and_worldsize()
    print(f"local_rank={local_rank}, global_rank={global_rank}, world_size={world_size}")
    is_local_main_process = local_rank == 0
    is_main_process = global_rank == 0

    images = sorted(glob(f"{args.image_dir}/*.png"))

    if world_size > 1:
        num_per_rank = round(len(images) / world_size)
        images = images[global_rank*num_per_rank:(global_rank+1)*num_per_rank]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank % 8)
        print(f"[rank {global_rank}/{world_size}] has {len(images)} prompts to process, using device {torch.cuda.current_device()}")

    run_name = time.strftime("%Y%m%d-%H")
    temp_dir = os.path.join(args.image_dir, f"eval_temp_{run_name}")

    if global_rank == 0:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

    dpg_score_model = DPGScore("cuda")
    q_dicts = prepare_dpg_data(f"eval/dpg/dpg_bench.csv")

    rank_json = {}
    with torch.no_grad():
        for image_path in tqdm(images):
            prompt_name = os.path.splitext(os.path.basename(image_path))[0]
            q_dict = q_dicts[prompt_name]
            images = split_grid(Image.open(image_path))
            rank_json[prompt_name] = []
            for i, img in enumerate(images):
                rank_json[prompt_name].append({})
                result = dpg_score_model(img, q_dict)
                for q_id, question in result["qid2question"].items():
                    answer = result["qid2answer"][q_id]
                    rank_json[prompt_name][i][question] = answer
                rank_json[prompt_name][i]['average_score_with_dependency'] = result['average_score_with_dependency']
                rank_json[prompt_name][i]['average_score_without_dependency'] = result['average_score_without_dependency']

    rank_save_path = os.path.join(temp_dir, f"scores_{global_rank}.json")
    json_dump(rank_json, rank_save_path, "utf-8")

    if global_rank == 0:
        while len(glob(os.path.join(temp_dir, f"scores_*.json"))) < world_size:
            time.sleep(5)
        time.sleep(5) # wait for the file writting to be finished
        merged_json = {}
        prompt_scores = {}
        scores = []
        for rank_path in glob(os.path.join(temp_dir, f"scores_*.json")):
            rank_json = json_load(rank_path, "utf-8")
            merged_json.update(rank_json)
            for prompt_name in rank_json:
                score_list = [x['average_score_with_dependency'] for x in rank_json[prompt_name]]
                prompt_scores[prompt_name] = np.mean(score_list)
                scores += score_list

        json_dump(merged_json, os.path.join(args.image_dir, f"dpg_scores_{run_name}.json"), "utf-8")
        dpg_score = np.mean(scores)
        lines_to_write = [
            f"DPG Score: {dpg_score:.2f}\n"
        ]
        print(lines_to_write[0])
        for prompt_name, score in prompt_scores.items():
            lines_to_write.append(f"{prompt_name}: {score:.2f}\n")

        with open(os.path.join(args.image_dir, f"dpg_scores_{run_name}.txt"), "w") as f:
            f.writelines(lines_to_write)

        shutil.rmtree(temp_dir)