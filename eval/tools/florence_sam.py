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
import torch
import cv2
from PIL import Image
from eval.grounded_sam.grounded_sam2_florence2_autolabel_pipeline import FlorenceSAM

class ObjectDetector:
    def __init__(self, device):
        self.device = torch.device(device)
        self.detector = FlorenceSAM(device)
    
    def get_instances(self, gen_image, label, min_size=64):
        _, instance_result_dict = \
            self.detector.od_grounding_and_segmentation(
                image=gen_image, text_input=label,
            )
        instances = instance_result_dict["instance_images"]
        
        filtered_instances = []
        for img in instances:
            width, height = img.shape[:2]
            if width * height < min_size * min_size or min(width, height) < min_size // 4:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            filtered_instances.append(img)

        return filtered_instances
    
    def get_multiple_instances(self, gen_image, label, min_size=64):
            # self.detector.phrase_grounding_and_segmentation(
        _, instance_result_dict = \
            self.detector.od_grounding_and_segmentation(
                image=gen_image, text_input=label,
            )
        
        return instance_result_dict


if __name__ == "__main__":
    # online demo: https://dun.163.com/trial/face/compare
    from glob import glob
    from tqdm import tqdm
    from eval.data_utils import split_grid, pad_to_square
    from eval.tools.dino import DINOScore

    detector = ObjectDetector("cuda")
    # dino_model = DINOScore("cuda")

    gen_image = Image.open("assets/cat.png").convert("RGB")
    label = "cat"

    save_dir = f"tmp"
    os.makedirs(save_dir, exist_ok=True)

    # for i, img in enumerate(split_grid(gen_image)):
    for i, img in enumerate([gen_image]):
        found_ips = detector.get_instances(img, label, min_size=img.size[0]//20)[:3]
        found_ips = [pad_to_square(x) for x in found_ips]
        for j, ip in enumerate(found_ips):
            # score = dino_model(real_image, ip)
            score = 1
            pad_to_square(ip).save(f"{save_dir}/{label}_{i}_{j}_{score}.png")
            
        

            


