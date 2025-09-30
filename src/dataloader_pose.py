import os
from zoneinfo import available_timezones
from PIL import Image, ImageFilter, ImageDraw
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
from datasets import load_dataset
import json
from torchvision.transforms import InterpolationMode
from transformers import pipeline
from src.adaptive_resize import AdaptiveResizeMultipleOf
from torch.utils.data import DataLoader, DistributedSampler
# depth_pipe = pipeline(
#     task="depth-estimation",
#     model="LiheYoung/depth-anything-small-hf",
#     device="cpu",
# )

PROMPT_TEMPLATES = {
        "canny": [
            "Use the Canny map to create {description}",
            "Guided by the Canny map, generate a picture of {description}",
            "Generate {description}, with its structure defined by the provided Canny map",
        ],
        "depth": [
            "Use the depth map to set the 3D scene for {description}",
            "Generate {description}, following the spatial layout from the depth map",
            "Create {description}, ensuring its perspective matches the provided depth map",
        ],
        "fill": [
            "Inpaint the missing areas of the image to show {description}",
            "Complete the partial image, using the visible context to create {description}",
            "Fill in the black regions of the image to reveal a full scene of {description}",
        ],
        "pose": [
            "Guided by the openpose image, generate a picture of {description}",
            "Generate {description}, using the provided pose reference to define the character's posture.",
            "Create {description}, with the character's pose strictly controlled by the pose reference image.",
            "{description}. The posture is defined by the attached pose guide.",
        ],
        "subject": [
            "Based on the reference image, create a new version showing {description}",
            "Using the input image as a strong visual guide, generate {description}",
            "Re-render the reference image to become {description}",
        ]
    }

def generate_control_prompt(condition: str, description: str) -> str:
    """
    Generates a randomized, formatted prompt for a given control condition.

    Args:
        condition (str): The type of control signal. 
                         Must be one of 'canny', 'depth', 'fill', 'pose'.
        description (str): The text description of the desired image content.

    Returns:
        str: A fully formatted prompt ready to be used by the model.
        
    Raises:
        ValueError: If the condition is not a valid key in PROMPT_TEMPLATES.
    """
    if condition not in PROMPT_TEMPLATES:
        raise ValueError(f"Invalid condition '{condition}'. Available conditions are: {list(PROMPT_TEMPLATES.keys())}")
    
    # Randomly choose a template from the list for the given condition
    chosen_template = random.choice(PROMPT_TEMPLATES[condition])
    
    # Format the chosen template with the user's description
    return chosen_template.format(description=description)

def get_canny_edge(img):
    img_np = np.array(img)
    low_threshold = random.randint(50, 150)
    high_threshold = random.randint(int(low_threshold * 2.0), int(low_threshold * 2.5))
    high_threshold = min(high_threshold, 255)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # edges = cv2.Canny(img_gray, 100, 200)
    edges = cv2.Canny(img_gray, low_threshold, high_threshold)
    return Image.fromarray(edges).convert("RGB")


# available_conds = ["canny", "coloring", "deblurring", "depth", "fill"]
available_conds = ["canny", "depth", "fill"]

_depth_pipe = None


def get_depth_pipe():
    global _depth_pipe
    if _depth_pipe is None:
        _depth_pipe = pipeline(
            task="depth-estimation",
            model="models/depth-anything-small-hf",
            device="cpu",
        )
    return _depth_pipe


def get_condition(image, cond_type):
    if cond_type not in condition_dict:
        raise ValueError(f"Condition type {cond_type} not implemented")
    if cond_type == "canny":
        condition_img = get_canny_edge(image)
    elif cond_type == "coloring":
        condition_img = image.convert("L").convert("RGB")
    elif cond_type == "deblurring":
        blur_radius = random.randint(1, 10)
        condition_img = (
            image.convert("RGB")
            .filter(ImageFilter.GaussianBlur(blur_radius))
            .convert("RGB")
        )
    elif cond_type == "depth":
        depth_pipe = get_depth_pipe()
        condition_img = depth_pipe(image)["depth"].convert("RGB")
    elif cond_type == "fill":
        condition_img = image.convert("RGB")
        w, h = image.size
        x1, x2 = sorted([random.randint(0, w), random.randint(int(w*0.2), w)])
        y1, y2 = sorted([random.randint(0, h), random.randint(int(h*0.2), h)])
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([x1, y1, x2, y2], fill=255)
        if random.random() > 0.5:
            mask = Image.eval(mask, lambda a: 255 - a)
        condition_img = Image.composite(
            image, Image.new("RGB", image.size, (0, 0, 0)), mask
        )
    else:
        raise ValueError(f"Condition type {cond_type} not implemented")
    return toTensor(condition_img)



image_transform = T.Compose(
    [
        AdaptiveResizeMultipleOf(max_size=512, multiple_of=16),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]
)

toTensor = T.Compose(
    [
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]
)

class OpenposeDataset(Dataset):
    def __init__(
        self,
        condition_size: int = 512,
        target_size: int = 512,
        image_size: int = 512,
        drop_text_prob: float = 0.1,
    ):
        # NOTE: You may need to adjust the dataset path
        super().__init__()
        dataset = load_dataset("limingcv/Captioned_COCOPose")
        self.base_dataset = dataset["train"]
        self.condition_size = condition_size
        self.target_size = target_size
        self.image_size = image_size
        self.drop_text_prob = drop_text_prob
        self.available_conds = available_conds

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        target_image = item["image"]
        primary_pose_image = item["control_pose"]
        description = item["caption"]

        # --- 2. Correctly prepare all conditions and their types ---
        condition_latents = []
        final_condition_types = []

        # Randomly sample one extra condition type from the available pool
        if random.random() < 0.25:
            extra_cond_type = random.choice(self.available_conds)
            extra_cond_img = get_condition(target_image, extra_cond_type)
            condition_latents.append(extra_cond_img)
            final_condition_types.append(extra_cond_type)
        else:
            condition_latents.append(image_transform(primary_pose_image))
            final_condition_types.append("pose")

        if random.random() > 0.2:
            description = generate_control_prompt(final_condition_types[0], description)
        condition_latents_tensor = torch.stack(condition_latents)

        # --- 4. Return the updated dictionary structure ---
        return {
            "pixel_values": image_transform(target_image),
            "condition_latents": condition_latents_tensor,
            "condition_types": ",".join(final_condition_types),  # This is now a list
            "descriptions": description,                   # New key with generated prompt
            "items": description,                      # New key with generated item
        }

def prepare_pose_dataloader(args, accelerator):
    train_dataset = OpenposeDataset()
    
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
    )
    return train_dataloader
