from accelerate.logging import get_logger
import torch
import io
import random
from PIL import (
    Image,
    ImageDraw,
)

logger = get_logger(__name__)
# from .condition import Condition
from diffusers.image_processor import VaeImageProcessor
from datasets import load_dataset, concatenate_datasets
# from .condition import Condition


def generate_combined_prompt(original_items, original_descriptions, condition_types):
    """
    Generates a clear, natural, and varied prompt that intelligently distinguishes
    between 'subject' (reference) conditions and spatial (constraint) conditions.
    """
    num_combined = len(original_descriptions)
    if num_combined == 0:
        return "", ""

    # --- Template Banks: Separated by Instruction Type ---

    # 1. For 'subject' conditions, emphasizing "reference" and "inspiration".
    SUBJECT_PROMPT_TEMPLATES = [
        "use the reference image as a visual guide to create {description}",
        "take inspiration from the reference picture to generate {description}",
        "based on the provided image, create a new version that shows {description}",
    ]

    # 2. For spatial conditions, emphasizing "constraints" and "guidelines".
    SPATIAL_PROMPT_TEMPLATES = {
        "canny": [
            "use the Canny map to create {description}",
            "generate {description}, with its structure strictly defined by the Canny outlines",
        ],
        "depth": [
            "use the depth map to create {description}",
            "generate {description}, following the precise spatial layout from the depth map",
        ],
        "fill": [
            "inpaint the missing areas of the image to show {description}",
            "complete the partial image, using the visible parts as context to create {description}",
        ],
    }

    # Helper to create a full description string
    def get_full_description(item, desc):
        item_lower = item.lower()
        desc_lower = desc.lower()
        clean_desc = desc.strip().rstrip(".")
        if item_lower in desc_lower:
            return clean_desc
        if random.random() < 0.3:
            return item
        return clean_desc

    # --- Main Logic ---
    
    instruction_parts = []
    item_parts = []

    for i in range(num_combined):
        condition = condition_types[i]
        full_desc = get_full_description(original_items[i], original_descriptions[i])
        ordinal = _get_ordinal_en(i, num_combined)
        
        instruction_body = ""
        # *** CORE LOGIC: Choose the correct template bank based on condition type ***
        if condition == "subject":
            template = random.choice(SUBJECT_PROMPT_TEMPLATES)
            instruction_body = template.format(description=full_desc)
        else:  # It's a spatial condition (canny, depth, fill, or pose)
            template = random.choice(SPATIAL_PROMPT_TEMPLATES[condition])
            instruction_body = template.format(description=full_desc)

        # Combine the ordinal reference with the generated instruction body
        linking_phrase = random.choice([
            "For the {ordinal} image, {body}",
            "Regarding the {ordinal} part, {body}",
        ])
        if random.random() < 0.2:
            final_instruction = linking_phrase.format(ordinal=ordinal, body=full_desc)  
        else:
            final_instruction = linking_phrase.format(ordinal=ordinal, body=instruction_body)
        instruction_parts.append(final_instruction)
        # item_parts.append(f"the {ordinal} image shows {full_desc}")
    
    # Join the individual instructions into one clear, composite prompt.
    prompt = "; ".join(instruction_parts) + "."
    # item_prompt = "; ".join(item_parts) + "."

    return prompt

# Helper functions for prompt generation
def _num_to_word(n):
    words = {
        1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
        6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
    }
    return words.get(n, str(n))


def _get_ordinal_en(n_idx, total_num):  # n_idx is the 0-based index
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

def get_dataset(args):
    dataset = []
    assert isinstance(args.dataset_name, list), "dataset dir should be a list"
    if args.dataset_name is not None:
        for name in args.dataset_name:
            dataset.append(load_dataset(name, cache_dir=args.cache_dir, split="train"))
    dataset = concatenate_datasets(dataset)
    if getattr(args, "shuffle_before_combining", True):
        dataset = dataset.shuffle(seed=getattr(args, "seed", None))
    return dataset


def prepare_dataset(dataset, accelerator, args, num_to_combine):
    image_processor = VaeImageProcessor(do_resize=True, do_convert_rgb=True)
    AVAILABLE_CONDITION_TYPES = ["subject", "canny", "depth", "fill"]

    def preprocess_single_image_for_pixel_values(image, resolution):
        return image_processor.preprocess(
            image, height=resolution, width=resolution
        ).squeeze(0)

    def preprocess_single_condition_image(cond_image, resolution):
        return image_processor.preprocess(
            cond_image, height=resolution, width=resolution
        ).squeeze(0)

    def preprocess(examples_batch):
        new_batch_combined_pixel_values = []
        new_batch_combined_condition_latents = []
        new_batch_combined_condition_types = []
        new_batch_combined_bboxes = []
        new_batch_combined_descriptions = []
        new_batch_combined_items = []

        num_original_examples_in_input_batch = len(examples_batch[args.image_column])
        current_idx = 0
        while current_idx <= num_original_examples_in_input_batch - num_to_combine:
            processed_left_images_for_concat = []
            chosen_condition_latents_for_stack = []
            chosen_condition_types_for_list = []
            adjusted_bboxes_for_list = []
            current_descriptions_list = []
            current_items_list = []

            for i in range(num_to_combine):
                original_example_global_idx = current_idx + i
                image_data = examples_batch[args.image_column][original_example_global_idx]
                bbox_data_orig_norm = examples_batch[args.bbox_column][original_example_global_idx]
                canny_data_bytes = examples_batch[args.canny_column][original_example_global_idx]["bytes"]
                depth_data_bytes = examples_batch[args.depth_column][original_example_global_idx]["bytes"]
                description_dict_orig = examples_batch["description"][original_example_global_idx]

                image = (
                    image_data.convert("RGB")
                    if not isinstance(image_data, str)
                    else Image.open(
                        io.BytesIO(image_data["bytes"])
                        if isinstance(image_data, dict) and "bytes" in image_data
                        else image_data
                    ).convert("RGB")
                )
                img_width, img_height = image.size
                if img_width % 2 != 0:
                    image = image.crop((0, 0, img_width - 1, img_height))
                    img_width = image.width
                    if img_width % 2 != 0:
                        raise ValueError(f"Image width {img_width} could not be made even.")

                left_image_pil = image.crop((0, 0, img_width // 2, img_height))
                right_image_pil = image.crop((img_width // 2, 0, img_width, img_height))

                w_half, h_orig = left_image_pil.size
                bbox_cx_px = bbox_data_orig_norm[0] * w_half
                bbox_cy_px = bbox_data_orig_norm[1] * h_orig
                bbox_w_px = bbox_data_orig_norm[2] * w_half
                bbox_h_px = bbox_data_orig_norm[3] * h_orig

                l_bbox = int(bbox_cx_px - bbox_w_px / 2)
                t_bbox = int(bbox_cy_px - bbox_h_px / 2)
                r_bbox = int(bbox_cx_px + bbox_w_px / 2)
                b_bbox = int(bbox_cy_px + bbox_h_px / 2)

                masked_left_image_pil = left_image_pil.copy()
                draw = ImageDraw.Draw(masked_left_image_pil)
                draw.rectangle([l_bbox, t_bbox, r_bbox, b_bbox], fill=(0, 0, 0))

                chosen_type_str = random.choice(AVAILABLE_CONDITION_TYPES)
                
                if chosen_type_str == "fill":
                    chosen_condition_pil = masked_left_image_pil
                elif chosen_type_str == "subject":
                    chosen_condition_pil = right_image_pil
                elif chosen_type_str == "canny":
                    chosen_condition_pil = Image.open(io.BytesIO(canny_data_bytes)).convert("RGB")
                elif chosen_type_str == "depth":
                    chosen_condition_pil = Image.open(io.BytesIO(depth_data_bytes)).convert("RGB")
                
                processed_left_img_tensor = preprocess_single_image_for_pixel_values(left_image_pil, args.resolution)
                processed_condition_tensor = preprocess_single_condition_image(chosen_condition_pil, args.resolution)

                processed_left_images_for_concat.append(processed_left_img_tensor)
                chosen_condition_latents_for_stack.append(processed_condition_tensor)
                chosen_condition_types_for_list.append(chosen_type_str)
                current_descriptions_list.append(description_dict_orig["description_0"])
                current_items_list.append(description_dict_orig["item"])

                l_res_scaled = int(l_bbox * args.resolution / w_half)
                t_res_scaled = int(t_bbox * args.resolution / h_orig)
                r_res_scaled = int(r_bbox * args.resolution / w_half)
                b_res_scaled = int(b_bbox * args.resolution / h_orig)
                horizontal_offset = i * args.resolution
                final_bbox_coords = [
                    l_res_scaled + horizontal_offset, t_res_scaled,
                    r_res_scaled + horizontal_offset, b_res_scaled,
                ]
                adjusted_bboxes_for_list.append(final_bbox_coords)

            combined_pixel_tensor = torch.cat(processed_left_images_for_concat, dim=2)
            combined_conditions_tensor = torch.stack(chosen_condition_latents_for_stack, dim=0)
            
            # final_combined_prompt_string, final_combined_item_prompt_string = generate_combined_prompt(
            #     current_items_list, current_descriptions_list, chosen_condition_types_for_list
            # )
            final_combined_prompt_string = generate_combined_prompt(current_items_list, current_descriptions_list, chosen_condition_types_for_list)
            new_batch_combined_pixel_values.append(combined_pixel_tensor)
            new_batch_combined_condition_latents.append(combined_conditions_tensor)
            new_batch_combined_condition_types.append(chosen_condition_types_for_list)
            new_batch_combined_bboxes.append(adjusted_bboxes_for_list)
            new_batch_combined_descriptions.append(final_combined_prompt_string)
            # new_batch_combined_items.append(final_combined_item_prompt_string)

            current_idx += num_to_combine

        output_batch = {
            "pixel_values": new_batch_combined_pixel_values,
            "condition_latents": new_batch_combined_condition_latents,
            "condition_types": new_batch_combined_condition_types,
            "bbox": new_batch_combined_bboxes,
            "prompt": new_batch_combined_descriptions,
            "items": current_items_list,
        }
        return output_batch

    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess)
    return dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    condition_latents = torch.stack([example["condition_latents"] for example in examples])
    condition_latents = condition_latents.to(memory_format=torch.contiguous_format).float()
    bboxes = [example["bbox"] for example in examples]
    condition_types = [example["condition_types"] for example in examples]
    descriptions = [example["prompt"] for example in examples]
    items = [example["items"] for example in examples]
    return {
        "pixel_values": pixel_values,
        "condition_latents": condition_latents,
        "condition_types": condition_types,
        "descriptions": descriptions,
        "bboxes": bboxes,
        "items": items,
    }