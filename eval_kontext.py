import argparse
import copy
import logging
import math
import os
import re
from safetensors.torch import save_file
from contextlib import contextmanager
from PIL import Image
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    set_seed,
)
from peft import LoraConfig, get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from src.pipeline_flux_kontext import FluxKontextPipeline as MoePipeline
from src.lora_moe import (
    convert_to_lora_moe,
    save_mole,
    set_expert_gate_status,
    load_experts_weights,
)
import diffusers
from src.text_encoder import encode_prompt
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline
import random
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import (
    check_min_version,
)
import json
from src.transformer_flux_kontext import FluxTransformer2DModel
from diffusers.image_processor import VaeImageProcessor
from src.spatial_eval_dataset import prepare_spatial_eval_dataloader
from datasets import load_from_disk

image_processor = VaeImageProcessor(do_resize=True)
# if is_wandb_available():
#     import wandb
from accelerate import DistributedDataParallelKwargs
import torch.nn.functional as F
from train_kontext import parse_target_modules
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.35.0.dev0")
from src.condition import Condition

logger = get_logger(__name__, log_level="INFO")


def resize_tensor_to_max_side(tensor, max_side=512):
    """
    tensor: [B, 3, H, W] or [3, H, W], float32, RGB, [-1,1]
    返回等比例缩小到最大边不超过max_side，且高宽都是16的倍数的tensor
    """
    is_batched = tensor.ndim == 4
    if not is_batched:
        tensor = tensor.unsqueeze(0)  # [1, 3, H, W]
    _, _, H, W = tensor.shape
    scale = min(max_side / H, max_side / W, 1.0)  # 只缩小，不放大
    new_H = int(H * scale)
    new_W = int(W * scale)
    # 保证高宽为16的倍数
    new_H = max(16, (new_H // 16) * 16)
    new_W = max(16, (new_W // 16) * 16)
    tensor_resized = F.interpolate(
        tensor, size=(new_H, new_W), mode="bilinear", align_corners=False
    )
    if not is_batched:
        tensor_resized = tensor_resized.squeeze(0)
    return tensor_resized


@contextmanager
def preserve_requires_grad(model):
    # 备份 requires_grad 状态
    requires_grad_backup = {
        name: param.requires_grad for name, param in model.named_parameters()
    }
    yield
    # 恢复 requires_grad 状态
    for name, param in model.named_parameters():
        param.requires_grad = requires_grad_backup[name]


def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
    )
    return text_encoder_one, text_encoder_two


def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.device).to(vae.dtype)).latent_dist.sample()
    pixel_latents = (
        pixel_latents - vae.config.shift_factor
    ) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def get_lora_pipe(ckpt="output/moe_V2.5_alltype-20k/"):
    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path="models/black-forest-labs/FLUX.1-Kontext-dev",
        subfolder="transformer",
    ).to(dtype=torch.bfloat16)
    transformer.load_lora_adapter(
        ckpt + "pytorch_lora_weights.safetensors",
        use_safetensors=True,
    )
    pipe = MoePipeline.from_pretrained(
        "models/black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
        transformer=transformer,
    ).to(dtype=torch.bfloat16)
    return pipe


def get_moe_pipe(ckpt="output/moe_V2.5_alltype-20k/"):
    with open(os.path.join(ckpt, 'train_config.json'), 'r') as f:
        config = json.load(f)
    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path="models/black-forest-labs/FLUX.1-Kontext-dev",
        subfolder="transformer",
    ).to(dtype=torch.bfloat16)

    transformer.load_lora_adapter(
        os.path.join(ckpt, "pytorch_lora_weights.safetensors"),
        use_safetensors=True,
    )
    moe_config = config["moe_config"]
    target_modules = parse_target_modules(moe_config["target_modules"], transformer)
    convert_to_lora_moe(transformer, moe_config, target_modules)

    if os.path.exists(os.path.join(ckpt, "condition_type_embedder.pt")):
        transformer.add_type_embedding(
            os.path.join(ckpt, "condition_type_embedder.pt"),
        )
    load_experts_weights(
        transformer,
        os.path.join(ckpt, "mole_experts.pt"),
    )
    # print(transformer)

    pipeline = MoePipeline.from_pretrained(
        "models/black-forest-labs/FLUX.1-Kontext-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to(dtype=torch.bfloat16)
    return pipeline


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="models/black-forest-labs/FLUX.1-Kontext-dev",
    )
    parser.add_argument(
        "--transformer",
        type=str,
        default="models/black-forest-labs/FLUX.1-Kontext-dev",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/black-forest-labs/FLUX.1-Kontext-dev",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
    )
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument(
        "--task",
        type=str,
        default="complex_edit",
    )
    parser.add_argument("--gen_gt_img", action="store_true")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def save_images(images, captions, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i, img in enumerate(images):
        if captions is not None:
            # 去掉特殊字符，防止文件名出错
            caption_str = "".join(
                c for c in captions[i] if c.isalnum() or c in [" ", "_", "-"]
            )[:50]
            filename = f"{caption_str}.png"
        else:
            exit(-1)
        save_path = os.path.join(save_dir, filename)
        img.save(save_path)

from PIL import Image

def stitch_images(images: list, orientation: str = 'horizontal', bg_color: tuple = (255, 255, 255)) -> Image.Image:
    if not images:
        raise ValueError("Image list cannot be empty.")
    if len(images) == 1:
        return images[0]

    if orientation == 'horizontal':
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        new_image = Image.new('RGB', (total_width, max_height), bg_color)
        
        current_x = 0
        for img in images:
            new_image.paste(img, (current_x, 0))
            current_x += img.width
            
    elif orientation == 'vertical':
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)
        new_image = Image.new('RGB', (max_width, total_height), bg_color)
        
        current_y = 0
        for img in images:
            new_image.paste(img, (0, current_y))
            current_y += img.height
    else:
        raise ValueError("Invalid orientation. Choose 'horizontal' or 'vertical'.")
        
    return new_image

def get_lora_pipe(ckpt):
    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path="models/black-forest-labs/FLUX.1-Kontext-dev",
        subfolder="transformer",
    ).to(dtype=torch.bfloat16)
    transformer.load_lora_adapter(
        os.path.join(
            ckpt,
            "pytorch_lora_weights.safetensors",
        ),
        use_safetensors=True,
    )
    pipe = MoePipeline.from_pretrained(
        "models/black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
        transformer=transformer,
    ).to(dtype=torch.bfloat16)
    return pipe


def main(args):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        # kwargs_handlers=[ddp_kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    set_seed(0)
    args.work_dir = os.path.join("output/moe", args.task, args.work_dir)
    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.work_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    
    use_lora = not os.path.exists(os.path.join(args.ckpt, 'train_config.json'))
    if use_lora:
        pipeline = get_lora_pipe(args.ckpt).to(accelerator.device, dtype=torch.bfloat16)
    else:
        pipeline = get_moe_pipe(args.ckpt).to(accelerator.device, dtype=torch.bfloat16)
    # from diffusers import FluxKontextPipeline
    # pipeline = FluxKontextPipeline.from_pretrained("models/black-forest-labs/FLUX.1-Kontext-dev").to(accelerator.device).to(dtype=torch.bfloat16)

    accelerator.wait_for_everyone()
    logger.info("All models loaded successfully")
    from datasets import load_dataset
    if args.task == "complex_edit":
        dataset = load_dataset("dataset/UCSC-VLAA/Complex-Edit")
        test_data = dataset["test_real"]
        # 获取所有的数据
        all_images = test_data["image"]
        all_prompts = [
            edit["compound"][7]["compound_instruction"]
            for edit in test_data["edit"]
        ]
    elif args.task == "omnicontext":
        dataset = load_from_disk("dataset/OmniContext")
        all_images = dataset["input_images"]
        all_prompts = dataset["instruction"]
        all_types = dataset["task_type"]
        all_keys = dataset["key"]
    elif args.task == "gedit":
        dataset = load_from_disk("dataset/stepfun-ai/GEdit-Bench-en")
        all_images = dataset["input_image"]
        all_prompts = dataset["instruction"]
        all_types = dataset["task_type"]
        all_keys = dataset["key"]
    elif args.task == "imagedit":
        with open("dataset/imgedit_asset/Benchmark/singleturn/singleturn.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        all_images = []
        all_prompts = []
        all_types = []
        all_keys = []
        for k, v in data.items():
            all_images.append(v["id"])
            all_prompts.append(v["prompt"])
            all_types.append(v["edit_type"])
            all_keys.append(k)
    images_to_process = all_images[accelerator.process_index::accelerator.num_processes]
    prompts_to_process = all_prompts[accelerator.process_index::accelerator.num_processes]
    if args.task == "complex_edit":
        indices_to_process = list(range(len(all_images)))[accelerator.process_index::accelerator.num_processes]
    else:
        task_type_to_process = all_types[accelerator.process_index::accelerator.num_processes]
        key_to_process = all_keys[accelerator.process_index::accelerator.num_processes]
    logger.info(f"Process {accelerator.process_index}: processing {len(images_to_process)} images.")
    pipeline.transformer.eval()
    save_path = args.work_dir
    # 3. Process each sample
    if args.task == "complex_edit": 
        for idx, image, prompt in zip(indices_to_process, images_to_process, prompts_to_process):
            image = image.convert("RGB")
            output_filename = f"{idx:05d}.png"
            output_filepath = os.path.join(save_path, output_filename)
            if os.path.exists(output_filepath):
                continue
            width, height = image.size
            edited_image = pipeline(
                    prompt=prompt,
                    images=[image],
                    # height=height,
                    # width=width,
                    # max_sequence_length=512,
                    guidance_scale=2.5,
                ).images[0]
            # edited_image = pipeline(
            #         prompt=prompt,
            #         image=image,
            #         # height=height,
            #         # width=width,
            #         # max_sequence_length=512,
            #         # guidance_scale=2.5,
            #     ).images[0]
            edited_image.save(output_filepath)
    elif args.task == "omnicontext":
        for key, task_type, image, prompt in zip(key_to_process, task_type_to_process, images_to_process, prompts_to_process):
            # image = image.convert("RGB")
            imgs = []
            for img in image:
                img = img.convert("RGB")
                imgs.append(img)
            output_filename = f"{key}.png"
            dir = os.path.join(save_path, "fullset", task_type)
            os.makedirs(dir, exist_ok=True)
            output_filepath = os.path.join(dir, output_filename)
            if os.path.exists(output_filepath):
                continue
            edited_image = pipeline(
                    prompt=prompt,
                    images=imgs[:1],
                    conds=imgs[1:],
                    guidance_scale=2.5,
                ).images[0]
            # image = stitch_images(imgs)
            # edited_image = pipeline(
            #         prompt=prompt,
            #         image=image,
            #         # height=height,
            #         # width=width,
            #         # max_sequence_length=512,
            #         # guidance_scale=2.5,
            #     ).images[0]
            edited_image.save(output_filepath)
    elif args.task == "gedit":
        for key, task_type, image, prompt in zip(key_to_process, task_type_to_process, images_to_process, prompts_to_process):
            image = image.convert("RGB")
            output_filename = f"{key}.png"
            dir = os.path.join(save_path, "fullset", task_type, 'en')
            os.makedirs(dir, exist_ok=True)
            output_filepath = os.path.join(dir, output_filename)
            if os.path.exists(output_filepath):
                continue
            edited_image = pipeline(
                    prompt=prompt,
                    images=[image],
                    guidance_scale=2.5,
                ).images[0]
            # image = stitch_images(imgs)
            # edited_image = pipeline(
            #         prompt=prompt,
            #         image=image,
            #         # height=height,
            #         # width=width,
            #         # max_sequence_length=512,
            #         # guidance_scale=2.5,
            #     ).images[0]
            edited_image.save(output_filepath)
    elif args.task == "imagedit":
        for key, task_type, image, prompt in zip(key_to_process, task_type_to_process, images_to_process, prompts_to_process):
            path = os.path.join('dataset/imgedit_asset/Benchmark/singleturn', image)
            image = Image.open(path).convert("RGB")
            output_filename = f"{key}.png"
            dir = os.path.join(save_path, "fullset", task_type, 'en')
            os.makedirs(dir, exist_ok=True)
            output_filepath = os.path.join(dir, output_filename)
            if os.path.exists(output_filepath):
                continue
            edited_image = pipeline(
                    prompt=prompt,
                    images=[image],
                    guidance_scale=2.5,
                ).images[0]
            # image = stitch_images(imgs)
            # edited_image = pipeline(
            #         prompt=prompt,
            #         image=image,
            #         # height=height,
            #         # width=width,
            #         # max_sequence_length=512,
            #         # guidance_scale=2.5,
            #     ).images[0]
            edited_image.save(output_filepath)

if __name__ == "__main__":
    args = parse_args()
    main(args)
