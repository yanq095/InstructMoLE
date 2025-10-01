from accelerate.logging import get_logger
import torch
import io
import random
logger = get_logger(__name__)
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler

def get_ordinal_en(n_idx, total_num):  # n_idx is the 0-based index
    if total_num == 2:
        return "left" if n_idx == 0 else "right"
    ordinals_list = [
        "first", "second", "third", "fourth", "fifth", "sixth",
        "seventh", "eighth", "ninth", "tenth",
    ]
    if n_idx < len(ordinals_list):
        return ordinals_list[n_idx]
    val = n_idx + 1
    if val % 10 == 1 and val % 100 != 11: return f"{val}st"
    elif val % 10 == 2 and val % 100 != 12: return f"{val}nd"
    elif val % 10 == 3 and val % 100 != 13: return f"{val}rd"
    else: return f"{val}th"

USING_PHRASES = [
    "using the", "utilize the", "apply the", "with the help of the", 
    "by means of the", "based on the", "leveraging the"
]

CANNY_PHRASES = [
    "canny edge of the {ref_no} image",
    "canny map from the {ref_no} image",
    "edge detection from the {ref_no} image",
    "structural outline from the {ref_no} image"
]

DEPTH_PHRASES = [
    "depth map of the {ref_no} image",
    "depth information from the {ref_no} image",
    "3D structure from the {ref_no} image"
]

SUBJECT_PHRASES = [
    "refer to the {item} in the {ref_no} image",
    "focus on the {item} from the {ref_no} image",
    "use the subject, which is a {item}, from the {ref_no} image",
    "take the {item} in the {ref_no} image as reference"
]

FILL_PHRASES = [
    "inpaint the black area based on the {ref_no} image",
    "fill in the masked region using the {ref_no} image",
    "complete the missing part with information from the {ref_no} image",
    "inpaint the void by utilizing the {ref_no} image"
]

RECREATE_PHRASES = [
    "to recreate the image.",
    "to generate the final image.",
    "to reconstruct the scene.",
    "in order to produce the final visual.",
    "and create the resulting image."
]

def prepare_dataset(dataset, accelerator, cond_num):
    image_processor = VaeImageProcessor(
        do_resize=True, do_convert_rgb=True
    )
    resolution = 512

    def preprocess(examples):
        # images = [image_transforms(image) for image in images]
        pixel_values = []
        condition_latents = []
        condition_types = []
        bboxes = []
        prompts = []
        for image, bbox, canny, depth, description in zip(
            examples['image'],
            examples['bbox'],
            examples['canny'],
            examples['depth'],
            examples["description"],
        ):
            desc = description["description_0"]
            item = description["item"]
            image = (
                image.convert("RGB")
                if not isinstance(image, str)
                else Image.open(image).convert("RGB")
            )
            width, height = image.size
            if width % 2 != 0:
                raise ValueError(
                    "Image width must be even to split into two equal parts."
                )
            left_image = image.crop((0, 0, width // 2, height)) 
            right_image = image.crop((width // 2, 0, width, height))
            # load mask image
            image_width, image_height = image.size
            bbox_pixel = [
                bbox[0] * image_width,
                bbox[1] * image_height,
                bbox[2] * image_width,
                bbox[3] * image_height,
            ]
            left = bbox_pixel[0] - bbox_pixel[2] / 2
            top = bbox_pixel[1] - bbox_pixel[3] / 2
            right = bbox_pixel[0] + bbox_pixel[2] / 2
            bottom = bbox_pixel[1] + bbox_pixel[3] / 2
            masked_left_image = left_image.copy()
            masked_left_image.paste(
                (0, 0, 0), (int(left), int(top), int(right), int(bottom))
            )
            bboxes.append(
                [
                    int(left * resolution / (width // 2)),
                    int(top * resolution / height),
                    int(right * resolution / (width // 2)),
                    int(bottom * resolution / height),
                ]
            )
            pixel_values.append(
                image_processor.preprocess(
                    left_image, width=resolution, height=resolution
                ).squeeze(0)
            )
            cond_types = random.sample(["subject", "canny", "depth", "fill"], cond_num)
            spatial_cond_prompts = []
            cond_prompts = []
            conditions = []
            for i, cond_type in enumerate(cond_types):
                ref_no = get_ordinal_en(i, len(cond_types))
                if cond_type == "canny":
                    phrase = random.choice(CANNY_PHRASES).format(ref_no=ref_no)
                    spatial_cond_prompts.append(phrase)
                    conditions.append(image_processor.preprocess(Image.open(io.BytesIO(canny["bytes"])), height=resolution, width=resolution)[0])
                elif cond_type == "depth":
                    phrase = random.choice(DEPTH_PHRASES).format(ref_no=ref_no)
                    spatial_cond_prompts.append(phrase)
                    conditions.append(image_processor.preprocess(Image.open(io.BytesIO(depth["bytes"])), height=resolution, width=resolution)[0])
                elif cond_type == "subject":
                    phrase = random.choice(SUBJECT_PHRASES).format(item=item, ref_no=ref_no)
                    cond_prompts.append(phrase)
                    conditions.append(image_processor.preprocess(right_image, height=resolution, width=resolution)[0])
                elif cond_type == "fill":
                    phrase = random.choice(FILL_PHRASES).format(ref_no=ref_no)
                    cond_prompts.append(phrase)
                    conditions.append(image_processor.preprocess(masked_left_image, height=resolution, width=resolution)[0])
                else:
                    raise ValueError(f"Unknown condition type: {cond_type}")
            prompt = ""
            all_conditions = cond_prompts + spatial_cond_prompts
            random.shuffle(all_conditions)
            
            style = random.choice(["imperative", "descriptive", "conjunctive"])

            if style == "imperative":
                prompt = ". ".join([p.capitalize() for p in all_conditions])
            elif style == "descriptive":
                prompt = "Generate an image by " + " and ".join(all_conditions)
            elif style == "conjunctive":
                prompt = ", and also ".join(all_conditions).capitalize()

            if random.random() < 0.3:
                prompt += " " + random.choice(RECREATE_PHRASES)
            else:
                if style == "descriptive":
                    prompt += f", making sure the final result is {desc}."
                else:
                    prompt += f", the goal is to create {desc}."
                    
            prompt = prompt.replace("..", ".").replace(" ,", ",").strip()
            cond_tensors = torch.stack(conditions, dim=0)
            condition_latents.append(cond_tensors)
            condition_types.append(cond_types)
            prompts.append(prompt)
        examples["pixel_values"] = pixel_values
        examples["condition_latents"] = condition_latents
        examples["condition_types"] = condition_types
        examples["bbox"] = bboxes
        examples["prompts"] = prompts
        return examples

    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess)

    return dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    condition_latents = torch.stack(
        [example["condition_latents"] for example in examples]
    )
    condition_latents = condition_latents.to(
        memory_format=torch.contiguous_format
    ).float()
    bboxes = [example["bbox"] for example in examples]
    condition_types = [example["condition_types"] for example in examples]
    # descriptions = [example["description"]["description_0"] for example in examples]
    descriptions = [example["prompts"] for example in examples]
    items = [example["description"]["item"] for example in examples]
    return {
        "pixel_values": pixel_values,
        "condition_latents": condition_latents,
        "condition_types": condition_types,
        "descriptions": descriptions,
        "bboxes": bboxes,
        "items": items,
    }

def prepare_sub_dataloader(accelerator, cond_num, args):
    train_dataset = prepare_dataset(
        load_dataset("Xuan-World/SubjectSpatial200K", split='train'), accelerator, cond_num
    )
    bsz = max(1, args.train_batch_size//cond_num)
    train_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        drop_last=True,
    )
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        shuffle=False,
        sampler=train_sampler,
        batch_size=bsz,
        num_workers=args.dataloader_num_workers,
    )
    return train_dataloader

# from accelerate import Accelerator
# accelerator = Accelerator()
# train_dataset = prepare_dataset(
#         load_dataset("Xuan-World/SubjectSpatial200K", split='train'), accelerator, 1
#     )
# print(train_dataset[0])