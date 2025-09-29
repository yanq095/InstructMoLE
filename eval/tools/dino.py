# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) Facebook, Inc. and its affiliates.
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

from transformers import ViTImageProcessor, ViTModel
from torch.nn import functional as F
from PIL import Image
import requests
from torchvision import transforms
import torch, os

class DINOScore:

    def __init__(self, device, use_center_crop=True):
        # https://github.com/facebookresearch/dino/issues/72#issuecomment-932874140
        # https://github.com/facebookresearch/dino/blob/main/eval_linear.py
        # https://gist.github.com/woctezuma/a30ee1de2e5efc1a3beff8e108795374
        # according to this, we should use center crop with class token
        self.device = torch.device(device)
        self.use_center_crop = use_center_crop

        if use_center_crop:
            self.T = transforms.Compose([
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            self.T = transforms.Compose([
                transforms.Resize(224, interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.model = ViTModel.from_pretrained(os.getenv("DINO_MODEL_PATH", "/opt/tiger/efficient_ai/UniCombine/models/facebook/dino-vits16")).to(self.device)

    
    def __call__(self, image_x, image_y, similarity_type="class"):

        inputs = torch.stack([self.T(x) for x in [image_x, image_y]]) # (2, 3, 224, 224). Batchsize = 2
        outputs = self.model(inputs.to(self.device))
        last_hidden_states = outputs.last_hidden_state
        
        assert similarity_type in ["class", "avg"]
        if similarity_type == "class":
            return self.cls_similarity(last_hidden_states[0], last_hidden_states[1])

        return self.avg_similairty(last_hidden_states[0], last_hidden_states[1])


    def avg_similairty(self, x, y):
        return F.cosine_similarity(x.mean(dim=0), y.mean(dim=0), dim=0).item() * 100

    def cls_similarity(self, x, y):
        return F.cosine_similarity(x[0], y[0], dim=0).item() * 100

if __name__ == "__main__":
    # urls = [
    #     'https://github.com/google/dreambooth/blob/main/dataset/rc_car/03.jpg?raw=true', # reference from Fig 11
    #     'https://github.com/google/dreambooth/blob/main/dataset/rc_car/02.jpg?raw=true'# Real Sample from Fig 11
    # ]
    # images = [Image.open(requests.get(url, stream=True).raw) for url in urls]
    # urls = [
    #     "assets/idipbench_base/object/3_pinkbackpack.png",
    #     "tmp/backpack_0.png",
    # ]
    urls = ["02.jpg", "03.jpg"]
    images = [Image.open(url).convert("RGB") for url in urls]

    dino_score_model = DINOScore("cuda", use_center_crop=True)
    print(dino_score_model(images[0], images[1], "class"))
    print(dino_score_model(images[0], images[1], "avg"))