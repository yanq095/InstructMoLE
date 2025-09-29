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

import cv2
import json
import torch
import random
import base64
import numpy as np
from PIL import Image, ImageDraw
from glob import glob
from torchvision import transforms as T
import os
import gc
from webdataset.filters import default_collation_fn, pipelinefilter
import yaml

def get_rank_and_worldsize():
    try:
        local_rank = int(os.environ.get("LOCAL_RANK"))
        global_rank = int(os.environ.get("RANK"))
        world_size = int(os.getenv('WORLD_SIZE', 1))
    except:
        local_rank = 0
        global_rank = 0
        world_size = 1
    return local_rank, global_rank, world_size

def get_train_config(config_path=None):
    if config_path is None:
        config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def calculate_aspect_ratios(resolution):
    ASPECT_RATIO = {
        '0.25': [128.0, 512.0], '0.26': [128.0, 496.0], '0.27': [128.0, 480.0], '0.28': [128.0, 464.0],
        '0.32': [144.0, 448.0], '0.33': [144.0, 432.0], '0.35': [144.0, 416.0], '0.4': [160.0, 400.0],
        '0.42': [160.0, 384.0], '0.48': [176.0, 368.0], '0.5': [176.0, 352.0], '0.52': [176.0, 336.0],
        '0.57': [192.0, 336.0], '0.6': [192.0, 320.0], '0.68': [208.0, 304.0], '0.72': [208.0, 288.0],
        '0.78': [224.0, 288.0], '0.82': [224.0, 272.0], '0.88': [240.0, 272.0], '0.94': [240.0, 256.0],
        '1.0': [256.0, 256.0], '1.07': [256.0, 240.0], '1.13': [272.0, 240.0], '1.21': [272.0, 224.0],
        '1.29': [288.0, 224.0], '1.38': [288.0, 208.0], '1.46': [304.0, 208.0], '1.67': [320.0, 192.0],
        '1.75': [336.0, 192.0], '2.0': [352.0, 176.0], '2.09': [368.0, 176.0], '2.4': [384.0, 160.0],
        '2.5': [400.0, 160.0], '2.89': [416.0, 144.0], '3.0': [432.0, 144.0], '3.11': [448.0, 144.0],
        '3.62': [464.0, 128.0], '3.75': [480.0, 128.0], '3.88': [496.0, 128.0], '4.0': [512.0, 128.0]
    }
    NEW_ASPECT_RATIO = {}
    for ratio in ASPECT_RATIO:
        height, width = ASPECT_RATIO[ratio]
        width = round(width / 256 * resolution)
        height = round(height / 256 * resolution)
        if width % 8 != 0:
            print(f"skip train resolution {width}, {height}")
            continue
        if height % 8 != 0:
            print(f"skip train resolution {width}, {height}")
            continue
        NEW_ASPECT_RATIO[ratio] = [height, width]
    return NEW_ASPECT_RATIO

ASPECT_RATIO_256 = calculate_aspect_ratios(256)
ASPECT_RATIO_384 = calculate_aspect_ratios(384)
ASPECT_RATIO_512 = calculate_aspect_ratios(512)
ASPECT_RATIO_768 = calculate_aspect_ratios(768)
ASPECT_RATIO_1024 = calculate_aspect_ratios(1024)

def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], closest_ratio


def _aspect_ratio_batched(
    data,
    batchsize=20,
    aspect_ratios=ASPECT_RATIO_512,
    batch_cross=False,
    collation_fn=default_collation_fn,
    partial=True,
):
    """Create batches of the given size.

    :param data: iterator
    :param batchsize: target batch size
    :param tensors: automatically batch lists of ndarrays into ndarrays
    :param partial: return partial batches
    :returns: iterator

    """
    assert collation_fn is not None
    buckets = {
        ratio: {"cross": [], "no_cross": []} for ratio in aspect_ratios.keys()
    }

    def check(buckets):
        for ratio in buckets:
            for bucket_name in buckets[ratio]:
                bucket = buckets[ratio][bucket_name]
                assert len(bucket) < batchsize

    for sample in data:
        check(buckets)
        height, width = sample['original_sizes']
        (new_height, new_width), closest_ratio = get_closest_ratio(height, width, aspect_ratios)

        bucket_name = "cross" if sample["has_cross"] and batch_cross else "no_cross"
        bucket = buckets[closest_ratio][bucket_name]
        bucket.append(sample)

        if len(bucket) >= batchsize:
            try:
                batch = collation_fn(bucket)
                yield batch
                del batch
            except Exception as e:
                print(f"[aspect_ratio_batched] collation_fn batch failed due to error {e}")
                for sample in bucket:
                    if "__key__" in sample:
                        print("error sample key in batch:", sample["__key__"])
                    if "__url__" in sample:
                        print("error sample url in batch:", sample["__url__"])
            buckets[closest_ratio][bucket_name] = []
            del bucket
            gc.collect()

    # yield the rest data and reset the buckets
    for ratio in buckets.keys():
        for bucket_name in ["cross", "no_cross"]:
            bucket = buckets[ratio][bucket_name]
            if len(bucket) > 0:
                if len(bucket) == batchsize or partial:
                    batch = collation_fn(bucket)
                    yield batch
                    del batch
                buckets[ratio][bucket_name] = []
                del bucket

aspect_ratio_batched = pipelinefilter(_aspect_ratio_batched)

def apply_aspect_ratio_batched(dataset, batchsize, aspect_ratios, batch_cross, collation_fn, partial=True):
    return dataset.compose(
        aspect_ratio_batched(
            batchsize, 
            aspect_ratios=aspect_ratios, 
            batch_cross=batch_cross,
            collation_fn=collation_fn, 
            partial=partial
        )
    )

def get_aspect_ratios(enable_aspect_ratio, resolution):
    if enable_aspect_ratio:
        # print("[Dataset] Multi Aspect Ratio Training Enabled")
        if resolution == 256:
            aspect_ratios = ASPECT_RATIO_256
        elif resolution == 384:
            aspect_ratios = ASPECT_RATIO_384
        elif resolution == 512:
            aspect_ratios = ASPECT_RATIO_512
        elif resolution == 768:
            aspect_ratios = ASPECT_RATIO_768
        elif resolution == 1024:
            aspect_ratios = ASPECT_RATIO_1024
        else:
            aspect_ratios = calculate_aspect_ratios(resolution)
    else:
        # print("[Dataset] Multi Aspect Ratio Training Disabled")
        aspect_ratios = {
            '1.0': [resolution, resolution]
        }
    return aspect_ratios

def bbox_to_grid(bbox, image_size, output_size=(224, 224)):
    """
    Convert bounding box to a grid of points.
    Args:
        bbox (list of float): [xmin, ymin, xmax, ymax]
        output_size (tuple of int): (height, width) of the output grid
        
    Returns:
        torch.Tensor: Grid of points with shape (output_height, output_width, 2)
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Create a meshgrid for the output grid
    h, w = output_size
    yy, xx = torch.meshgrid(
        torch.linspace(ymin, ymax, h),
        torch.linspace(xmin, xmax, w)
    )
    grid = torch.stack((xx, yy), -1)
    
    # Normalize grid to range [-1, 1]
    H, W = image_size
    grid[..., 0] = grid[..., 0] / (W - 1) * 2 - 1  # Normalize x to [-1, 1]
    grid[..., 1] = grid[..., 1] / (H - 1) * 2 - 1  # Normalize y to [-1, 1]
    
    return grid

def random_crop_instance(instance, min_crop_ratio):
    assert 0 < min_crop_ratio <= 1
    crop_width_ratio = random.uniform(min_crop_ratio, 1)
    crop_height_ratio = random.uniform(min_crop_ratio, 1)
    
    orig_width, orig_height = instance.size
    
    crop_width = int(orig_width * crop_width_ratio)
    crop_height = int(orig_height * crop_height_ratio)
    
    crop_left = random.randint(0, orig_width - crop_width)
    crop_top = random.randint(0, orig_height - crop_height)
    
    crop_box = (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height) # (left, upper, right, lower)
    return instance.crop(crop_box), crop_box

pil2tensor = T.ToTensor()
tensor2pil = T.ToPILImage()

cv2pil = lambda x: Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
pil2cv2 = lambda x: cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR)

def compute_psnr(x, y):
    y = y.resize(x.size)
    x = pil2tensor(x) * 255.
    y = pil2tensor(y) * 255.
    mse = torch.mean((x - y) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse)).item()

def replace_first_occurrence(sentence, word_or_phrase, replace_with):
    # Escape special characters in word_or_phrase for exact matching
    escaped_word_or_phrase = re.escape(word_or_phrase)
    pattern = r'\b' + escaped_word_or_phrase + r'\b'
    
    # Finding the first match
    match = next(re.finditer(pattern, sentence), None)
    if match:
        # Perform replacement
        result = re.sub(pattern, replace_with, sentence, count=1)
        replaced = True
        index = match.start()
    else:
        # No match found
        result = sentence
        replaced = False
        index = -1
    
    return result, replaced, index


def decode_base64_to_image(base64_str):
    # Decode the base64 string to bytes
    img_bytes = base64.b64decode(base64_str)
    # Create a BytesIO buffer from the bytes
    img_buffer = io.BytesIO(img_bytes)
    # Open the image using Pillow
    image = Image.open(img_buffer)
    return image

def jpeg_compression(pil_image, quality):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buffer.getvalue()))

def pad_to_square(pil_image):
    new_size = max(pil_image.width, pil_image.height)
    square_image = Image.new("RGB", (new_size, new_size), "white")
    left = (new_size - pil_image.width) // 2
    top = (new_size - pil_image.height) // 2
    square_image.paste(pil_image, (left, top))
    return square_image

def pad_to_target(pil_image, target_size):
    original_width, original_height = pil_image.size
    target_width, target_height = target_size
    
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height
    
    # Pad the image to the target aspect ratio
    if original_aspect_ratio > target_aspect_ratio:
        new_width = original_width
        new_height = int(new_width / target_aspect_ratio)
    else:
        new_height = original_height
        new_width = int(new_height * target_aspect_ratio)
    
    pad_image = Image.new("RGB", (new_width, new_height), "white")
    left = (new_width - original_width) // 2
    top = (new_height - original_height) // 2
    pad_image.paste(pil_image, (left, top))
    
    # Resize the image to the target size
    resized_image = pad_image.resize(target_size)
    return resized_image

def image_grid(imgs, rows, cols):
    # assert len(imgs) == rows * cols

    w, h = imgs[0].size
    if imgs[0].mode == 'L':
        grid = Image.new('L', size=(cols * w, rows * h))
    else:
        grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def split_grid(image):
    width = image.width // 2
    height = image.height // 2

    crop_tuples_list = [
        (0, 0, width, height),
        (width, 0, width*2, height),
        (0, height, width, height*2),
        (width, height, width*2, height*2),
    ]
    def crop_image(input_image, crop_tuple=None):
        if crop_tuple is None:
            return input_image
        return input_image.crop((crop_tuple[0], crop_tuple[1], crop_tuple[2], crop_tuple[3]))
    
    return [crop_image(image, crop_tuple) for crop_tuple in crop_tuples_list]

def add_border(img, border_color, border_thickness):
    """
    Add a colored border to an image without changing its size.
    
    Parameters:
        border_color (tuple): Border color in RGB (e.g., (255, 0, 0) for red).
        border_thickness (int): Thickness of the border in pixels.
    """
    width, height = img.size
    img = img.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, width, border_thickness), fill=border_color)
    draw.rectangle((0, height - border_thickness, width, height), fill=border_color)
    draw.rectangle((0, 0, border_thickness, height), fill=border_color)
    draw.rectangle((width - border_thickness, 0, width, height), fill=border_color)
    return img

def merge_bboxes(bboxes):
    if not bboxes:
        return None  # Handle empty input
    
    # Extract all coordinates
    x_mins = [b[0] for b in bboxes]
    y_mins = [b[1] for b in bboxes]
    x_maxs = [b[2] for b in bboxes]
    y_maxs = [b[3] for b in bboxes]
    
    # Compute the merged box
    merged_box = (
        min(x_mins),  # x_min
        min(y_mins),  # y_min
        max(x_maxs),  # x_max
        max(y_maxs)   # y_max
    )
    return merged_box


def flip_bbox_left_right(bbox, image_width):
    """
    Flips the bounding box horizontally on an image.
    
    Parameters:
    bbox (list of float): [x_min, y_min, x_max, y_max]
    image_width (int): The width of the image
    
    Returns:
    list of float: New bounding box after horizontal flip [x_min', y_min', x_max', y_max']
    """
    x_min, y_min, x_max, y_max = bbox
    new_x_min = image_width - x_max
    new_x_max = image_width - x_min
    new_bbox = [new_x_min, y_min, new_x_max, y_max]
    return new_bbox

def json_load(path, encoding='ascii'):
    with open(path, 'r', encoding=encoding) as file:
        return json.load(file)

def json_dump(obj, path, encoding='ascii', indent=4, create_dir=True, verbose=True, **kwargs):
    if create_dir and os.path.dirname(path) != '':
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding=encoding) as file:
        json.dump(obj, file, indent=4, ensure_ascii=False, **kwargs)
    if verbose:
        print(type(obj), 'saved to', path)
