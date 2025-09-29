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
import argparse

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import facer
import facer.transform
from copy import deepcopy
import PIL


def resize_image(image, max_size=1024):
    height,width,_ = image.shape
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int((height / width) * max_size)
        else:
            new_height = max_size
            new_width = int((width / height) * max_size)
        image = cv2.resize(image, (new_width, new_height))
    return image 

def open_and_resize_image(image_file, max_size=1024, return_type='numpy'):
    if isinstance(image_file, str) or isinstance(image_file, PIL.Image.Image):
        if isinstance(image_file, str):
            img = Image.open(image_file)
        else:
            img = image_file
        width, height = img.size
        if width > height:
            new_width = max_size
            new_height = int((height / width) * max_size)
        else:
            new_height = max_size
            new_width = int((width / height) * max_size)
        img = img.resize((new_width, new_height))
        if return_type == 'numpy':
            return np.array(img.convert('RGB'))
        else:
            return img
    elif isinstance(image_file, np.ndarray):
        height,width,_ = image_file.shape
        if width > height:
            new_width = max_size
            new_height = int((height / width) * max_size)
        else:
            new_height = max_size
            new_width = int((width / height) * max_size)
        img = cv2.resize(image_file, (new_width, new_height))
        assert return_type == 'numpy'
        return img
    else:
        raise TypeError("Do not support this img type")


@torch.no_grad()
def loose_warp_face(input_image, face_detector, face_target_shape=(512, 512), scale=1.3, face_parser=None, device=None, croped_face_scale=3, bg_value = 0, croped_face_y_offset=0.0):
    """ Get the tight/loose warp of the face in the image, in which only one face is of concern.

    Args:
        input_image: Image path, or PIL.Image.Image, or np.ndarray (dtype=np.uint8).
        face_detector: a facer.face_detector, for face detection.
        face_target_shape: Output resolution.
        scale: Scale of the output image w.r.t. the face it contains.

    Returns:
        PIL.Image.Image, single warped face.
    """
    _normalized_face_target_pts = torch.tensor([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.729904, 92.2041]]) / 112.0
    
    target_pts = ((_normalized_face_target_pts -
                   torch.tensor([0.5, 0.5])) / scale
                  + torch.tensor([0.5, 0.5]))
    if face_detector is not None:
        device = next(face_detector.parameters()).device

    if isinstance(input_image, str):
        # image_tensor_hwc = facer.read_hwc(input_image)
        np_img = open_and_resize_image(input_image)[:,:,:3]      # Downsample high-res images to avoid OOM.
        img_height, img_width = np_img.shape[:2]
        image_tensor_hwc = torch.from_numpy(np_img)
    elif isinstance(input_image, Image.Image):
        image_tensor_hwc = torch.from_numpy(np.array(input_image)[:,:,:3])
        img_height, img_width = image_tensor_hwc.shape[:2]
        assert image_tensor_hwc.dtype == torch.uint8
    else:
        assert isinstance(input_image, np.ndarray), 'Type %s of input_image is unsupported!' % type(input_image)
        assert input_image.dtype == np.uint8, 'dtype %s of input np.ndarray is unsupported!' % input_image.dtype
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)[:,:,:3]
        input_image = resize_image(input_image)
        image_tensor_hwc = torch.from_numpy(input_image)
        img_height, img_width = image_tensor_hwc.shape[:2]
    
    image_pt_bchw_255 = facer.hwc2bchw(image_tensor_hwc).to(device)

    res = {'cropped_face_masked': None, 'cropped_face': None, 'cropped_img': None, 'cropped_face_mask': None, 'align_face': None}

    if face_detector is not None:
        try:
            face_data = face_detector(image_pt_bchw_255)
        except:
            import pdb;pdb.set_trace()
        
        if len(face_data) == 0:
            return res
        
        if face_parser is not None:
            with torch.inference_mode():
                faces = face_parser(image_pt_bchw_255, face_data)
            seg_logits = faces['seg']['logits']
            seg_probs = seg_logits.softmax(dim=1)
            seg_probs = seg_probs.argmax(dim=1).unsqueeze(1)[:1]

        face_rects = face_data['rects'][:1]
        face_rects = face_data['rects'][:1]
        x1,y1,x2,y2 = face_rects[0][:4]
        x1 = (int(x1.item()))
        y1 = (int(y1.item()))
        x2 = (int(x2.item()))
        y2 = (int(y2.item()))
        face_width = x2-x1
        face_height = y2-y1
        center_x = int(0.5*(x1+x2))
        center_y = int(0.5*(y1+y2)) + croped_face_y_offset * face_height
        croped_face_width = face_width*croped_face_scale
        croped_face_height = face_height*croped_face_scale
        
        x1 = max(int(center_x-0.5*croped_face_width),0)
        x2 = min(int(center_x+0.5*croped_face_width), img_width-1)
        y1 = max(int(center_y-0.5*croped_face_height),0)
        y2 = min(int(center_y+0.5*croped_face_height), img_height-1)
        croped_face_height = y2-y1
        croped_face_width = x2-x1
        center_x = int(0.5*(x1+x2))
        center_y = int(0.5*(y1+y2))
        croped_face_len = min(croped_face_height, croped_face_width)
        x1 = int(center_x - 0.5*croped_face_len)
        y1 = int(center_y - 0.5*croped_face_len)
        x2 = x1+croped_face_len
        y2 = y1+croped_face_len
        croped_image_pt_bchw_255 = image_pt_bchw_255[:, :, y1:y2, x1:x2]
        face_points = face_data['points'][:1]
        batch_inds = face_data['image_ids'][:1]
        
        matrix_align = facer.transform.get_face_align_matrix(
            face_points, face_target_shape, 
            target_pts=(target_pts * torch.tensor(face_target_shape)))
        
        grid = facer.transform.make_tanh_warp_grid(
            matrix_align, 0.0, face_target_shape, image_pt_bchw_255.shape[2:],)
        image = F.grid_sample(
            image_pt_bchw_255.float()[batch_inds], 
            grid, 'bilinear', align_corners=False)
        image_align_raw = deepcopy(image)
        image_align_raw = facer.bchw2hwc(image_align_raw).to(torch.uint8).cpu().numpy()
        image_align_raw = Image.fromarray(image_align_raw)
        image_croped = facer.bchw2hwc(croped_image_pt_bchw_255).to(torch.uint8).cpu().numpy()
        image_croped = Image.fromarray(image_croped)
        if face_parser is not None:
            image_no_mask = deepcopy(image)
            new_size = list(seg_probs.shape)
            new_size[1] = image.shape[1]
            seg_probs = seg_probs.expand(new_size)
            assert seg_probs.shape[0] == 1 and image.shape[0] == 1, 'mask shape {}, != image shape {}'.format(seg_probs.shape, image.shape)
            mask_img = F.grid_sample(seg_probs.float(), grid, 'bilinear', align_corners=False)
            image[mask_img == 0] = bg_value
            mask_img[mask_img!=0] = 1
            assert mask_img.shape[0] == 1
        else:
            image_no_mask = image
            mask_img = None
    else:
        image = image_pt_bchw_255
        image_no_mask = image_pt_bchw_255
        image_align_raw = None
        image_croped = None

    image = facer.bchw2hwc(image).to(torch.uint8).cpu().numpy()
    image_no_mask = facer.bchw2hwc(image_no_mask).to(torch.uint8).cpu().numpy()
    
    res.update({'cropped_face_masked': Image.fromarray(image), 'cropped_face': Image.fromarray(image_no_mask), 'cropped_img':image_croped, 'cropped_face_mask': mask_img, 'align_face': image_align_raw})
    return res

def tight_warp_face(input_image, face_detector, face_parser=None, device=None):
    return loose_warp_face(input_image, face_detector, 
        face_target_shape=(112, 112), scale=1, face_parser=face_parser, device=device)
