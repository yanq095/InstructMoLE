from PIL import Image
import io
from src.data.dataset_square import KVDataset
import json
import bson
from src.adaptive_resize import AdaptiveResizeMultipleOf
import torch
import torchvision.transforms as T
from dataloader import KVReader
import random
import torch.nn.functional as F
import numpy as np
import cv2


class SampleDecoder:
    def __call__(self, item):
        try:
            # a = torch.tensor(item['id_embed'])
            # b = torch.tensor(item['swapped_face_id_embed'])
            # similarity = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            if item['similarity'] < 0.7:
                return None
            image = Image.open(io.BytesIO(item['image'])).convert('RGB')
            swapped_face = Image.open(io.BytesIO(item['swapped_face'])).convert('RGB')
            source_face = Image.open(io.BytesIO(item['source_face'])).convert('RGB')
            mask = Image.open(io.BytesIO(item['mask']))
            # image2_before = Image.open(io.BytesIO(item['image2_before']))
            prompts = ["Swap the face from the second image onto the person in the first image.",
                       "Face swap the person in the first image with the face from the second image.",
                       "Apply the face from the second image to the person in the first image.",
                       "Replace the face in the first image with the face from the second image.",
                       "Take the facial features from the second image and seamlessly blend them onto the person in the first image, matching the original lighting.",
                       "Perform a face swap: use the first image as the target and the second image as the face source.",
                       "Map the facial identity from the second image onto the person in the first image, preserving the background and style of the first image.",
                       "Substitute the face in the first image with the one from the second.",
                       "Transfer the facial likeness from the second image onto the first.",]
            return {
                "text": random.choice(prompts),
                "image": swapped_face,
                "target_face": image,
                "source_face": source_face,
                "kps": item['kps'],
                "mask": mask,
            }
        except Exception as e:
            print(f"SampleDecoder error: {e}")
            return None


def create_mask_from_points_template_alignment(points, image_shape):
    standard_points = np.array([
        [85, 100], [170, 100], [128, 145], [95, 180], [160, 180]
    ], dtype=np.float32)
    standard_mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(standard_mask, center=(128, 140), axes=(80, 100), angle=0, startAngle=0, endAngle=360, color=(255), thickness=-1)
    
    input_points = np.array(points, dtype=np.float32)
    transform_matrix, _ = cv2.estimateAffinePartial2D(standard_points, input_points, method=cv2.LMEDS)

    if transform_matrix is None:
        return np.zeros(image_shape[:2], dtype=np.uint8)
        
    target_size = (image_shape[0], image_shape[1])
    final_mask = cv2.warpAffine(standard_mask, transform_matrix, target_size)
    return Image.fromarray(final_mask, mode='L')


class SwapFaceDataset(KVDataset):
    def __init__(
        self,
        paths=["hdfs://harunafr/home/byte_voyager_fr/user/jinqi.xiao/data/self_swap_alamy_l3single_merge"],
        rank=0,
        world_size=1,
        shuffle=False,
        sample_decoder=SampleDecoder(),
        image_transform=T.Compose([
            AdaptiveResizeMultipleOf(max_size=512, multiple_of=16),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]),
    ):
        super().__init__(paths, rank, world_size, shuffle)
        self.sample_decoder = sample_decoder
        self.image_transform = image_transform
        self._length = 657016
        # for filepath in self.filepaths:
        #     try:
        #         reader = KVReader(filepath)
        #         self._length += len(reader.list_keys())
        #     except Exception as ex:
        #         print(f"Error counting keys in {filepath}: {ex}")
        #         continue
        # print(f"Dataset length: {self._length}")


    def __len__(self):
        return self._length

    def __iter__(self):
        for item in super().__iter__():
            try:
                try:
                    item = bson.loads(item)
                except:
                    item = json.loads(item)
                sample = self.sample_decoder(item)
                if sample is None:
                    continue
                
                # mask = create_mask_from_points_template_alignment(sample['kps'], sample['image'].size)
                if self.image_transform:
                    sample["image"] = self.image_transform(sample["image"])
                    sample["target_face"] = self.image_transform(sample["target_face"])
                    sample["source_face"] = self.image_transform(sample["source_face"])
                    mask = self.image_transform(sample["mask"])
                    mask = (mask > 0.5).float()
                    sample["mask"] = mask
                # target_height = sample["target_face"].shape[-2]
                # target_width = sample["target_face"].shape[-1]
                # sample["mask"] = F.interpolate(
                #                     mask, 
                #                     size=(target_height, target_width), 
                #                     mode='nearest'
                #                 )[0][0]
                yield sample
            except Exception as ex:
                print(f"Error: {ex}")
                continue


def pad_collate_fn(batch):
    keys = ['image', 'target_face', 'source_face', 'mask']
    batch_out = {}
    # 先处理需要pad和stack的字段
    for key in keys:
        images = [item[key] for item in batch]
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        padded_images = []
        for img in images:
            c, h, w = img.shape
            pad_h = max_h - h
            pad_w = max_w - w
            img = F.pad(img, (0, pad_w, 0, pad_h), value=0.0)
            padded_images.append(img)
        batch_out[key] = torch.stack(padded_images)
    # 处理其他字段（比如text、id_embed等），保持为list
    for k in batch[0]:
        if k not in keys:
            batch_out[k] = [d[k] for d in batch]
    return batch_out

    
def prepare_swap_face_dataloader(args, accelerator):
    train_dataset= SwapFaceDataset(
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size//2,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=pad_collate_fn,
    )
    return train_dataloader


# import torch.nn.functional as F
# from PIL import Image
# from diffusers.image_processor import VaeImageProcessor
# image_processor = VaeImageProcessor()
# from PIL import Image, ImageDraw, ImageFont
# def concat_images(img1, img2):
#     # 横向拼接
#     img1 = image_processor.postprocess(img1.unsqueeze(0), output_type="pil")[0]
#     img2 = image_processor.postprocess(img2.unsqueeze(0), output_type="pil")[0]
#     w, h = img1.size
#     new_img = Image.new('RGB', (w * 2, h))
#     new_img.paste(img1, (0, 0))
#     new_img.paste(img2, (w, 0))
#     return new_img

# def draw_similarity_on_image(image, similarity):
#     draw = ImageDraw.Draw(image)
#     text = f"Cosine similarity: {similarity:.4f}"
#     # 你可以自定义字体和位置
#     draw.text((10, 10), text, fill=(255, 0, 0))
#     return image
# from diffusers.image_processor import VaeImageProcessor
# image_processor = VaeImageProcessor(do_resize=True)
# def visualize_mask_application(transformed_image, 
#                                original_mask, 
#                                output_path: str):
#     binarized_mask = (original_mask > 0.5).float()
#     masked_image_tensor = transformed_image * binarized_mask
#     # 1. 获取变换后图像的目标尺寸
#     transformed_image_pil = image_processor.postprocess(
#                                 masked_image_tensor.unsqueeze(0),
#                                 output_type="pil",
#                             )[0]
    
#     transformed_image_pil.save(output_path)
#     print(f"已保存掩码可视化结果至: {output_path}")

# t = SwapFaceDataset()
# for i, sample in enumerate(t):
#     # print(sample)
#     visualize_mask_application(
#         transformed_image=sample["target_face"],
#         original_mask=sample['mask'],
#         output_path=f"target_face_masked.png" # 使用唯一ID命名
#     )
    # break
#     a = torch.tensor(sample['id_embed'])
#     b = torch.tensor(sample['swapped_face_id_embed'])
#     similarity = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
#     # if similarity < 0.2:
#     #     continue
#     print(similarity)
#     concat_img = concat_images(sample['swapped_face'], sample['source_face'])
#     concat_img = draw_similarity_on_image(concat_img, similarity)
#     concat_img.save("tmp/"+str(i)+".jpg")
