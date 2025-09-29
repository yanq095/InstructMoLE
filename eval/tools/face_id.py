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
import facer
from PIL import Image
from torchvision import transforms
from eval.tools.face_utils.face import tight_warp_face
from eval.tools.face_utils.face_recg import Backbone
import os
from torch.nn import functional as F
from eval.data_utils import pad_to_square, pad_to_target, json_dump, json_load, split_grid


def expand_bounding_box(x_min, y_min, x_max, y_max, factor=1.3):
    # Calculate the center of the bounding box
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # Calculate the width and height of the bounding box
    width = x_max - x_min
    height = y_max - y_min

    # Calculate the new width and height
    new_width = factor * width
    new_height = factor * height

    # Calculate the new bounding box coordinates
    x_min_new = x_center - new_width / 2
    x_max_new = x_center + new_width / 2
    y_min_new = y_center - new_height / 2
    y_max_new = y_center + new_height / 2

    return x_min_new, y_min_new, x_max_new, y_max_new

class FaceID:
    def __init__(self, device):
        self.device = torch.device(device)
        self.T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.detector = facer.face_detector("retinaface/resnet50", device=device, model_path="models/Resnet50_Final.pth")
        face_model_path = "models/model_ir_se50.pth" #os.getenv("FACE_ID_MODEL_PATH")
        self.model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.model.load_state_dict(torch.load(face_model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def detect(self, image, expand_scale=1.3):
        with torch.no_grad():
            faces = self.detector(image.to(self.device))
        bboxes = faces['rects'].detach().cpu().tolist()
        bboxes = [expand_bounding_box(*x, expand_scale) for x in bboxes]
        return bboxes

    def __call__(self, image_x, image_y, normalize=False):
        # NOTE: Only Support One Face Per Image
        
        try:
            warp_x = tight_warp_face(image_x, self.detector)['cropped_face_masked']
            warp_y = tight_warp_face(image_y, self.detector)['cropped_face_masked']
        except:
            # print("[Warning] No face detected!!")
            return 0

        if warp_x is None or warp_y is None:
            # print("[Warning] No face detected!!")
            return 0
        
        feature_x = self.model(self.T(warp_x).unsqueeze(0).to(self.device))[0] # [512]
        feature_y = self.model(self.T(warp_y).unsqueeze(0).to(self.device))[0] # [512]
        
        
        if normalize:
            feature_x = feature_x / feature_x.norm(p=2, dim=-1, keepdim=True)
            feature_y = feature_y / feature_y.norm(p=2, dim=-1, keepdim=True)

        return F.cosine_similarity(feature_x, feature_y, dim=0).item() * 100
    


if __name__ == "__main__":
    # online demo: https://dun.163.com/trial/face/compare
    from src.train.data.data_utils import pil2tensor
    import numpy as np


    faceid = FaceID("cuda")
    real_image_path = "assets/bengio_bengio.png"

    # gen_image_path = "runs/0303-2034_flux100k_mod-t_oc-sblocks_multi-0.5_fs-lora8_cond192_res384_bs48_resume/eval/ckpt/40000/0304_cond192_tar512/1_A man is wearing green headphones standi.png"
    # gen_image_path = "runs/0303-2034_flux100k_mod-t_oc-sblocks_multi-0.5_fs-lora8_cond192_res384_bs48_resume/eval/ckpt/40000/0304_cond192_tar512/7_A man is wearing green headphones standi.png"
    # gen_image_path = "runs/0303-2034_flux100k_mod-t_oc-sblocks_multi-0.5_fs-lora8_cond192_res384_bs48_resume/eval/ckpt/40000/0304_cond192_tar512/198_A man wearing a black white suit and a w.png"
    gen_image_path = "data/tmp/GCG/florence-sam_phrase-grounding_S3L-two-20k_v41_wds/00000/qwen10M_00000-00010_00000_000000008_vis.png"
    if "eval/ckpt" in gen_image_path:
        gen_images = split_grid(Image.open(gen_image_path))
    else:
        gen_images = [Image.open(gen_image_path)]

    for i, gen_img in enumerate(gen_images):
        # img_tensor = torch.from_numpy(np.array(gen_img)).unsqueeze(0).permute(0, 3, 1, 2)
        img_tensor = (pil2tensor(gen_img).unsqueeze(0) * 255).to(torch.uint8)
        bboxes = faceid.detect(img_tensor)
        for j, bbox in enumerate(bboxes):
            face_img = gen_img.crop(bbox)
            face_img.save(f"tmp_{j}.png")
            print(faceid(real_image_path, face_img))
        # print(faceid.detect(img_tensor))
        # break
        # print(faceid(real_image_path, gen_img))


