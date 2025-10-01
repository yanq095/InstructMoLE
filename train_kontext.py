import argparse
import copy
import logging
import math
import os
from PIL import Image
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import json
import wandb
import torch
from peft import LoraConfig, get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from src.pipeline_flux_kontext import FluxKontextPipeline
from src.lora_moe import (
    convert_to_lora_moe,
    save_mole,
    set_expert_gate_status,
    LoRAMoE,
    TopKGate
)
import diffusers
from src.text_encoder import encode_prompt
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.combined_dataloader import CombinedDataLoader
from src.transformer_flux_kontext import FluxTransformer2DModel
from diffusers.image_processor import VaeImageProcessor
from src.dataloader_pose import prepare_pose_dataloader
from src.dataloader_subject import prepare_sub_dataloader
from src.dataloader_omniedit import prepare_omniedit_dataloader

image_processor = VaeImageProcessor(do_resize=True)
import torch.nn.functional as F
check_min_version("0.35.0.dev0")
logger = get_logger(__name__, log_level="INFO")

def prepare_dataloaders(args, accelerator: Accelerator):
    """
    Prepares dataloaders, distributing them based on the machine (node).
    All GPUs on the same machine will handle the same set of dataloaders.
    """
    world_size = accelerator.num_processes
    global_rank = accelerator.process_index
    local_rank = accelerator.local_process_index
    
    # This assumes all machines have the same number of GPUs.
    gpus_per_machine = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    
    # Check for sane environment
    if world_size % gpus_per_machine != 0:
        raise RuntimeError("WORLD_SIZE must be divisible by the number of GPUs per machine.")
        
    num_machines = world_size // gpus_per_machine
    machine_id = global_rank // gpus_per_machine

    if global_rank == 0:
        print("--- Dataloader Distribution Setup ---")
        print(f"Total GPUs (WORLD_SIZE): {world_size}")
        print(f"GPUs per Machine (LOCAL_WORLD_SIZE): {gpus_per_machine}")
        print(f"Detected Machines: {num_machines}")
        print("Distributing tasks across machines...")
        print("-------------------------------------")
    print(f"Machine ID: {machine_id}, Local Rank: {local_rank}")
    datasets = []
    first_eval_batches = []
    train_dataloader_pose = prepare_pose_dataloader(args, accelerator)
    first_eval_batches += [next(iter(train_dataloader_pose))]
    datasets.append(train_dataloader_pose)
    
    train_dataloader_sub = prepare_sub_dataloader(accelerator, 1, args)
    datasets.append(train_dataloader_sub)
    train_dataloader_sub = prepare_sub_dataloader(accelerator, 2, args)
    datasets.append(train_dataloader_sub)
    first_eval_batches += [next(iter(train_dataloader_sub))]
    train_dataloader_sub = prepare_sub_dataloader(accelerator, 3, args)
    datasets.append(train_dataloader_sub)
    
    train_dataloader_edit = prepare_omniedit_dataloader(args, accelerator)
    first_eval_batches += [next(iter(train_dataloader_edit))]
    datasets.append(train_dataloader_edit)

    from src.dataloader_multi_subjects import prepare_multi_sub_dataloader
    train_dataloader_multi_sub = prepare_multi_sub_dataloader(args, accelerator)
    first_eval_batches += [next(iter(train_dataloader_multi_sub))]
    datasets.append(train_dataloader_multi_sub)
    
    train_dataloader = CombinedDataLoader(dataloaders_list=datasets, seed=args.seed+global_rank)
    print(f"rank {global_rank} has {len(datasets)} dataloaders")
    accelerator.wait_for_everyone()
    return train_dataloader, first_eval_batches

def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
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


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.1-Kontext-dev",
    )
    parser.add_argument(
        "--transformer",
        type=str,
        default="black-forest-labs/FLUX.1-Kontext-dev",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--gate_learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=(
            'We default to the "none" weighting scheme for uniform sampling and uniform loss'
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-04,
        help="Weight decay to use for unet params",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
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
    parser.add_argument(
        "--seed", type=int, default=3407, help="A seed for reproducible training."
    )

    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=100000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=100000,
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--config", type=str, default="train_config.json")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def parse_target_modules(module_templates, transformer):
    final_modules = []
    num_single_layers = len(transformer.single_transformer_blocks) if hasattr(transformer, 'single_transformer_blocks') else 0
    num_double_layers = len(transformer.transformer_blocks) if hasattr(transformer, 'transformer_blocks') else 0
    for template in module_templates:
        if "*" not in template:
            final_modules.append(template)
            continue
        if "single_transformer_blocks" in template:
            f_string = template.replace("*", "{i}")
            final_modules.extend([f_string.format(i=i) for i in range(num_single_layers)])
        elif "transformer_blocks" in template:
            f_string = template.replace("*", "{i}")
            final_modules.extend([f_string.format(i=i) for i in range(num_double_layers)])
        else:
            pass
    return final_modules


@torch.no_grad()
def get_cosine_decay_schedule(current_step: int, total_steps: int, min_value: float = 1e-5) -> float:
    if current_step >= total_steps:
        return min_value
    initial_value = 1.0
    decay_value = min_value + 0.5 * (initial_value - min_value) * (
        1 + math.cos(math.pi * current_step / total_steps)
    )
    return decay_value

def save_images(images, captions, save_dir, prefixs="img", resize_height=512):
    os.makedirs(save_dir, exist_ok=True)
    for i, img in enumerate(images):
        # resize
        original_width, original_height = img.size
        aspect_ratio = original_width / original_height
        new_width = int(resize_height * aspect_ratio)
        resized_img = img.resize((new_width, resize_height), Image.LANCZOS)
        if captions is not None:
            caption_str = "".join(
                c for c in captions[i] if c.isalnum() or c in [" ", "_", "-"]
            )[:50]
            filename = f"{'_'.join(prefixs[i])}_{i}_{caption_str}.png"
        else:
            filename = f"{'_'.join(prefixs[i])}_{i}.png"
        save_path = os.path.join(save_dir, filename)
        resized_img.save(save_path)


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    if args.work_dir is None:
        from datetime import datetime

        args.work_dir = os.path.join(
            "output/train_result", f"{datetime.now().strftime('%y_%m_%d-%H:%M')}"
        )
    else:
        from datetime import datetime
        args.work_dir = os.path.join(
            args.work_dir, "train_result", f"{datetime.now().strftime('%y_%m_%d-%H:%M')}"
        )    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.work_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    # args.output_denoising_lora = "_".join(args.condition_types)
    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder"
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    text_encoder_one, text_encoder_two = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two
    )
    text_encoder_one = text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two = text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    ).to(accelerator.device, dtype=weight_dtype)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
    ).to(accelerator.device, dtype=weight_dtype)

    # freeze parameters of models to save more memory
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
   
    with open(args.config, 'r') as f:
        config = json.load(f)
    use_lora_only = "moe_config" not in config
    lora_config = config["lora_config"]
    lora_config["target_modules"] = parse_target_modules(lora_config["target_modules"], transformer)
    transformer.add_adapter(
        LoraConfig(**lora_config),
    )
    if not use_lora_only:
        moe_config = config["moe_config"]
        target_modules = parse_target_modules(moe_config["target_modules"], transformer)
        convert_to_lora_moe(transformer, moe_config, target_modules)
        if moe_config["use_type_embedding"]:
            transformer.add_type_embedding()

    # transformer.add_siglip_embedder()
    accelerator.wait_for_everyone()
    # print(transformer)
    transformer.enable_gradient_checkpointing()
    # transformer.gradient_checkpointing = True
    # transformer._gradient_checkpointing_func = checkpoint
    logger.info("All models loaded successfully")
    transformer = transformer.to(dtype=weight_dtype)
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(transformer, dtype=torch.float32)

    p_gate = []
    p_lora = []
    for n, p in transformer.named_parameters():
        if p.requires_grad:
            if "gate" in n:
                p_gate.append(p)
            else:
                p_lora.append(p)

    total_params = sum(p.numel() for p in p_gate + p_lora)
    total_params_in_millions = total_params / 1e6
    print(f"Total trainable parameters: {total_params_in_millions:.2f}M")
    # p_lora_with_lr = {"params": p_lora, "lr": args.learning_rate}
    param_groups = [
        {"params": p_lora, "lr": args.learning_rate},
        {"params": p_gate, "lr": args.gate_learning_rate},
    ]
    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoders, tokenizers, prompt, args.max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            text_ids = text_ids.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    # num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(
            len(train_dataloader) / accelerator.num_processes
        )
        num_update_steps_per_epoch = math.ceil(
            len_train_dataloader_after_sharding / args.gradient_accumulation_steps
        )

    # Preprocessing the datasets.
    train_dataloader, first_eval_batches = prepare_dataloaders(args, accelerator)
    logger.info("Training dataset and Dataloader initialized successfully.")

    (
        transformer,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        transformer,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # accelerator.init_trackers(
        #     "wandb",
        #     config=vars(args),
        #     init_kwargs={"project": "MultiKontext"}
        # )
        wandb.init(project="MultiKontext")

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total trainable parameters: = {total_params_in_millions:.2f}M")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.work_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.work_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    freeze_gate = False
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            use_text = random.random() > 0.1
            use_cond = random.random() > 0.05
            semantic_conds = []
            hidden_idx = []
            # with_face_img = False
            with torch.no_grad():
                if "face_image_tensor" in batch:
                    prompts = batch["text"]
                    # resized_image = resize_tensor_to_max_side(
                    #     batch["image"], max_side=args.resolution
                    # )
                    latent_image = encode_images(
                        pixels=batch["image"], vae=vae, weight_dtype=weight_dtype
                    )
                    condition_types = [["face"]] * args.train_batch_size
                    # with_face_img = random.random() > 0.75
                    # with_face_img = not args.without_face_img
                    # face_id_embed = batch["id_embed"]
                    cond_num = 1
                    condition_latents = [batch["face_image_tensor"]]
                    if random.random() < 0.8:
                        hidden_idx = [0]
                elif "edited_img" in batch:
                    prompts = batch["prompt"]
                    latent_image = encode_images(
                        pixels=batch["edited_img"], vae=vae, weight_dtype=weight_dtype
                    )
                    condition_types = batch["task"]
                    condition_latents = [batch["src_img"]]
                    hidden_idx = [0]
                else:
                    prompts = batch["descriptions"]
                    latent_image = encode_images(
                        pixels=batch["pixel_values"], vae=vae, weight_dtype=weight_dtype
                    )
                latent_mask = None
                if "target_face" in batch:
                    if "mask" in batch:
                        latent_height = latent_image.shape[2]
                        latent_width = latent_image.shape[3]
                        latent_mask = F.interpolate(
                            batch["mask"].float(),  # 确保掩码是 float 类型
                            size=(latent_height, latent_width),
                            mode="area",
                        )
                        latent_mask = (latent_mask > 0.5).float().to(accelerator.device)

                if not use_text:
                    prompts = [""] * len(prompts)
                prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                    prompts, text_encoders, tokenizers
                )
                # 1.2 Get positional id.
                img_ids = [
                    FluxKontextPipeline._prepare_latent_image_ids(
                        latent_image.shape[0],
                        latent_image.shape[2] // 2,
                        latent_image.shape[3] // 2,
                        accelerator.device,
                        weight_dtype,
                    )
                ]
                # 2.1 Convert Conditions to latent space list.
                # 2.2 Get Conditions positional id list.
                # 2.3 Get Conditions types string list.
                # (bs, cond_num, c, h, w) -> [cond_num, (bs, c, h ,w)]
                bsz = latent_image.shape[0]
                # print("#prompts#", accelerator.process_index, prompts)
                if "condition_latents" in batch:
                    # if isinstance(batch["condition_types"][0], list):
                    #     all_cond_types = batch["condition_types"]
                    # else:
                    #     all_cond_types = [types.split(",") for types in batch["condition_types"]]
                    # num_selected = random.randint(1, cond_num)
                    # selected_indices = random.sample(range(cond_num), num_selected)
                    # selected_conditions = batch["condition_latents"][
                    #     :, selected_indices, :, :, :
                    # ]
                    condition_latents = list(
                        torch.unbind(batch["condition_latents"], dim=1)
                    )
                    # condition_types = [
                    #     batch["condition_types"][0][i] for i in selected_indices
                    # ]
                    # condition_types = all_cond_types
                    # print("#condition_types#", accelerator.process_index, condition_types)

                    # [cond_num, (len ,3) ]
                    # [cond_num]
                # cond_type_ids = [Condition.get_type_ids(types) for types in condition_types ]

                conds_ids = []
                ref_offset = 1
                condition_encode_latents = []
                offset_w = 0
                for i, images_per_condition in enumerate(condition_latents):
                    # i means condition No.i.
                    # images_per_condition = (bs, c, h ,w)
                    # if condition_types[i] == "face" or condition_types[i] == "subject":
                    #     semantic_conds.append(
                    #         siglip_image_encoder(images_per_condition)
                    #     )
                    if isinstance(images_per_condition, list):
                        print(
                            i,
                            len(images_per_condition),
                            images_per_condition[0].shape,
                            len(condition_latents),
                        )
                    images_per_condition = encode_images(
                        pixels=images_per_condition, vae=vae, weight_dtype=weight_dtype
                    )
                    cond_ids = FluxKontextPipeline._prepare_latent_image_ids(
                        images_per_condition.shape[0],
                        images_per_condition.shape[2] // 2,
                        images_per_condition.shape[3] // 2,
                        accelerator.device,
                        weight_dtype,
                    )

                    cond_ids[..., 0] = ref_offset
                    cond_ids[..., 2] += offset_w
                    offset_w += images_per_condition.shape[2] // 2
                    ref_offset += 1
                    if images_per_condition is not None:
                        conds_ids.append(cond_ids)
                        condition_encode_latents.append(images_per_condition)

                condition_latents = condition_encode_latents
                # 3 Sample noise that we'll add to the latents
                noise = torch.randn_like(latent_image)

                # 4 Sample a random timestep for each image
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(
                    device=accelerator.device
                )

                # 5 Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(
                    timesteps, n_dim=latent_image.ndim, dtype=latent_image.dtype
                )
                noisy_model_input = (1.0 - sigmas) * latent_image + sigmas * noise

                # 6.1 pack noisy_model_input
                packed_noisy_model_input = FluxKontextPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=latent_image.shape[0],
                    num_channels_latents=latent_image.shape[1],
                    height=latent_image.shape[2],
                    width=latent_image.shape[3],
                )
                # 6.2 pack Conditions latents
                vae_conds_len = []
                for i, images_per_condition in enumerate(condition_latents):
                    condition_latents[i] = FluxKontextPipeline._pack_latents(
                        images_per_condition.to(accelerator.device),
                        batch_size=images_per_condition.shape[0],
                        num_channels_latents=images_per_condition.shape[1],
                        height=images_per_condition.shape[2],
                        width=images_per_condition.shape[3],
                    )
                    vae_conds_len.append(condition_latents[i].shape[1])

                # 7 handle guidance
                if accelerator.unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.tensor(
                        [args.guidance_scale], device=accelerator.device
                    )
                    guidance = guidance.expand(latent_image.shape[0])
                else:
                    guidance = None

                hidden_states = [packed_noisy_model_input]
                for idx in hidden_idx:
                    hidden_states.append(condition_latents[idx])
                    img_ids.append(conds_ids[idx])
                    condition_latents.pop(idx)
                    conds_ids.pop(idx)
                    vae_conds_len.pop(idx)
                hidden_states = torch.concat(hidden_states, dim=-2)
            if len(condition_latents) > 0:
                condition_latents = torch.concat(condition_latents, dim=-2)
                cond_ids = torch.concat(conds_ids, dim=-2)
            else:
                condition_latents = None
                cond_ids = None
            img_ids = torch.concat(img_ids, dim=-2)

            with accelerator.accumulate(transformer):
                # 8 Predict the noise residual
                # print(condition_types)
                if not use_cond:
                    cond_hidden_states = None
                    cond_ids = None
                model_pred = transformer(
                    hidden_states=hidden_states,
                    cond_hidden_states=condition_latents,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    cond_ids=cond_ids,
                    vae_conds_len=vae_conds_len,
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, : packed_noisy_model_input.size(1)]
                model_pred = FluxKontextPipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[2] * vae_scale_factor,
                    width=noisy_model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )
                # flow matching loss
                target = noise - latent_image

                if latent_mask is not None:
                    per_latent_pixel_loss = (model_pred.float() - target.float()) ** 2
                    weighted_masked_loss = (
                        weighting.float() * per_latent_pixel_loss * latent_mask
                    )
                    loss_sum_per_sample = torch.sum(
                        weighted_masked_loss.reshape(target.shape[0], -1), dim=1
                    )
                    mask_weight_sum_per_sample = (
                        torch.sum(latent_mask.reshape(target.shape[0], -1), dim=1)
                        * target.shape[1]
                    )
                    epsilon = 1e-8
                    true_mean_loss_per_sample = loss_sum_per_sample / (
                        mask_weight_sum_per_sample + epsilon
                    )
                    main_loss = true_mean_loss_per_sample.mean()
                else:
                    main_loss = torch.mean(
                        (
                            weighting.float()
                            * (model_pred.float() - target.float()) ** 2
                        ).reshape(target.shape[0], -1),
                        1,
                    )
                    main_loss = main_loss.mean()
                loss = main_loss
                # if not use_lora_only and args.entropy_loss_weight > 0:
                #     entropy_losses = []
                #     for module in accelerator.unwrap_model(transformer).modules():
                #         # 检查模块是否是 LoRAMoE 并且暂存了权重 (意味着它刚执行了软路由)
                #         if isinstance(module, LoRAMoE) and hasattr(module, "latest_routing_weights") and module.latest_routing_weights is not None:
                #             routing_weights = module.latest_routing_weights
                            
                #             # H(p) = - sum(p * log(p))
                #             epsilon = 1e-8
                #             entropy_per_sample = -torch.sum(
                #                 routing_weights * torch.log(routing_weights + epsilon),
                #                 dim=1
                #             )
                #             entropy_losses.append(entropy_per_sample.mean())

                #     # 如果找到了任何软路由层，就将它们的平均熵损失加入总损失
                #     if entropy_losses:
                #         low_entropy_loss = torch.mean(torch.stack(entropy_losses))
                #         loss = main_loss + args.entropy_loss_weight * low_entropy_loss

                accelerator.backward(loss)

            if accelerator.sync_gradients:
                if accelerator.is_main_process:
                    logs = {}
                    logs["loss/total"] = loss.detach().item()
                    wandb.log(logs, step=global_step)

                params_to_clip = transformer.parameters()
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                global_step += 1

                LoRAMoE.set_aux_loss_weight(get_cosine_decay_schedule(global_step, args.max_train_steps))


            if global_step % args.checkpointing_steps == 0:
                # if accelerator.is_main_process:
                # if True:
                save_path = os.path.join(
                        args.work_dir, f"checkpoint-{global_step}"
                    )
                # if accelerator.is_main_process:
                if True:
                    os.makedirs(save_path, exist_ok=True)
                    if (
                        accelerator.unwrap_model(
                            transformer
                        ).condition_type_embedder
                        is not None
                    ):
                        state_dict = {}
                        state_dict["condition_type_embedder"] = (
                            accelerator.unwrap_model(
                                transformer
                            ).condition_type_embedder.state_dict()
                        )
                        # state_dict["time_cond_embed.cond_embedder"] = (
                        #     accelerator.unwrap_model(
                        #         transformer
                        #     ).time_cond_embed.cond_embedder.state_dict()
                        # )
                        torch.save(
                            state_dict,
                            save_path + "/condition_type_embedder.pt",
                        )
                    if not use_lora_only:
                        save_mole(
                            accelerator.unwrap_model(transformer),
                            save_path,
                            moe_config,
                        )
                        config_save_path = os.path.join(save_path, "train_config.json")
                        try:
                            with open(config_save_path, 'w', encoding='utf-8') as f:
                                json.dump(config, f, ensure_ascii=False, indent=4)
                            print(f"Save training config: {config_save_path}")
                        except Exception as e:
                            print(f"ERROR: {e}")
                        if freeze_gate:
                            set_expert_gate_status(
                                accelerator.unwrap_model(transformer),
                                moe_config,
                                requires_grad=False,
                            )
                    else:
                        FluxKontextPipeline.save_lora_weights(
                            save_directory=save_path,
                            transformer_lora_layers=get_peft_model_state_dict(
                                model=accelerator.unwrap_model(transformer),
                            ),
                            safe_serialization=True,
                        )
                    logger.info(f"Saved state to {save_path}")
                    affinity_data_collector = {}
                    token_affinity_data_collector = {}
                    unwrapped_model = accelerator.unwrap_model(transformer)
                    unwrapped_model.eval()
                    for first_eval_batch in first_eval_batches:
                        with torch.no_grad():
                            imgs = []
                            conds = []
                            pipeline = FluxKontextPipeline.from_pretrained(
                                args.pretrained_model_name_or_path,
                                vae=vae,
                                text_encoder=accelerator.unwrap_model(text_encoder_one),
                                text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                                transformer=accelerator.unwrap_model(transformer),
                                torch_dtype=weight_dtype,
                            ).to(accelerator.device, dtype=weight_dtype)
                            vae_conds_len = []
                            if "edited_img" in first_eval_batch:
                                prompt = first_eval_batch["prompt"]
                                height = first_eval_batch["edited_img"].shape[2]
                                width = first_eval_batch["edited_img"].shape[3]
                                condition_types = [[x] for x in first_eval_batch["task"]]
                                imgs = [first_eval_batch["src_img"]]
                            else:
                                prompt = first_eval_batch["descriptions"]
                                item_only = False
                                if item_only:
                                    height = 512
                                    width = 512
                                    prompt = first_eval_batch["items"]
                                else:
                                    height = first_eval_batch["pixel_values"].shape[2]
                                    width = first_eval_batch["pixel_values"].shape[3]

                                conds = list(
                                    torch.unbind(
                                        first_eval_batch["condition_latents"].to(
                                            accelerator.device, dtype=weight_dtype
                                        ),
                                        dim=1,
                                    )
                                )
                                if isinstance(first_eval_batch["condition_types"][0], list):
                                    condition_types = first_eval_batch["condition_types"]
                                else:
                                    condition_types = [
                                        types.split(",")
                                        for types in first_eval_batch["condition_types"]
                                    ]
                            # cond_type_ids = [Condition.get_type_ids(types) for types in condition_types ]
                            # print(accelerator.process_index, condition_types, len(imgs), len(conds))
                            if accelerator.is_main_process:
                                LoRAMoE.debug = True                               
                            generated_images = pipeline(
                                prompt=prompt,
                                images=imgs,
                                conds=conds,
                                height=height,
                                width=width,
                                max_sequence_length=512,
                                # num_inference_steps=1,
                            ).images
                            LoRAMoE.debug = False
                            if accelerator.is_main_process:
                                for name, module in unwrapped_model.named_modules():
                                    if isinstance(module, LoRAMoE) and isinstance(module.gate, TopKGate):
                                        if module.route_by_type:
                                            selected_indices = module.latest_token_indices
                                            selected_weights = module.latest_token_weights
                                            
                                            if selected_indices is None: 
                                                print("No selected indices")
                                                continue
                                            
                                            if name not in affinity_data_collector:
                                                affinity_data_collector[name] = {}
                                            
                                            task_name = condition_types[0][0] if condition_types[0][0] else "unknown"
                                            if task_name not in affinity_data_collector[name]:
                                                affinity_data_collector[name][task_name] = torch.zeros(
                                                    moe_config["num_experts"], 
                                                    device=selected_weights.device, 
                                                    dtype=selected_weights.dtype
                                                )
                                            
                                            affinity_data_collector[name][task_name].scatter_add_(
                                                dim=0,
                                                index=selected_indices.view(-1),
                                                src=selected_weights.view(-1)
                                            )
                                        else:
                                            if hasattr(module, "latest_token_indices") and module.latest_token_indices is not None:
                                                if name not in token_affinity_data_collector:
                                                    token_affinity_data_collector[name] = torch.zeros(
                                                        moe_config["num_experts"], device=accelerator.device, dtype=torch.float32
                                                    )
                                                
                                                indices = module.latest_token_indices.view(-1)
                                                weights = module.latest_token_weights.view(-1)
                                                
                                                token_affinity_data_collector[name].scatter_add_(0, indices, weights.float())
                
                            res = []
                            if "edited_img" in first_eval_batch:
                                res = []
                                gt = image_processor.postprocess(
                                    first_eval_batch["edited_img"],
                                    output_type="pil",
                                )
                                cond = image_processor.postprocess(
                                    first_eval_batch["src_img"],
                                    output_type="pil",
                                )
                                for i, gen_img in enumerate(generated_images):
                                    width, height = gt[i].size
                                    concat_image = Image.new("RGB", (width * 3, height))
                                    concat_image.paste(gt[i], (0, 0))
                                    concat_image.paste(cond[i], (width, 0))
                                    concat_image.paste(
                                        gen_img.resize((width, height)), (width * 2, 0)
                                    )
                                    res.append(concat_image)
                            else:
                                for i, result_img in enumerate(generated_images):
                                    cond_images = image_processor.postprocess(
                                        first_eval_batch["condition_latents"][i],
                                        output_type="pil",
                                    )
                                    concat_image = Image.new(
                                        "RGB", (width + len(cond_images) * 512, 512)
                                    )
                                    for j, cond_image in enumerate(cond_images):
                                        concat_image.paste(cond_image, (j * 512, 0))
                                    concat_image.paste(
                                        result_img.resize((width, height)),
                                        (j * 512 + 512, 0),
                                    )
                                    res.append(concat_image)
                            generated_images = res
                            save_images(
                                generated_images,
                                prompt,
                                save_dir=save_path,
                                prefixs=condition_types,
                                resize_height=512,
                            )
                    if token_affinity_data_collector and accelerator.is_main_process:
                        for layer_name, expert_weights in token_affinity_data_collector.items():
                            expert_weights_np = expert_weights.cpu().numpy()
                            num_experts = len(expert_weights_np)
                            expert_ids = [f"Expert {i}" for i in range(num_experts)]
                            
                            total_weight = expert_weights_np.sum()
                            if total_weight > 1e-8:
                                distribution = (expert_weights_np / total_weight) * 100
                            else:
                                distribution = expert_weights_np # Avoid division by zero
                                
                            try:
                                fig, ax = plt.subplots(figsize=(12, 7))
                                ax.bar(expert_ids, distribution, color=sns.color_palette("viridis", num_experts))
                                ax.set_ylabel("Routing Percentage (%)")
                                ax.set_xlabel("Expert ID")
                                ax.set_title(f"Token Routing Distribution @ Step {global_step}\nLayer: {layer_name}")
                                plt.xticks(rotation=45, ha="right")
                                plt.tight_layout()
                                
                                wandb.log(
                                    {f"token_routing_distribution/{layer_name}": wandb.Image(fig)},
                                    step=global_step
                                )
                                plt.close(fig)
                            except (ImportError, NameError):
                                logger.warning("matplotlib or seaborn not found. Skipping token routing distribution plot.")

                    if affinity_data_collector and accelerator.is_main_process:
                        for layer_name, collected_data_by_task in affinity_data_collector.items():
                            
                            task_total_weights = {}
                            num_experts = moe_config["num_experts"]
                            
                            for task_name, total_weights_tensor in collected_data_by_task.items():
                              
                                sum_of_weights = total_weights_tensor.sum()
                                if sum_of_weights > 1e-6: 
                                    normalized_weights = total_weights_tensor / sum_of_weights
                                else:
                                    normalized_weights = total_weights_tensor
                                
                                task_total_weights[task_name] = normalized_weights.cpu().float().numpy()
                            tasks = sorted(list(task_total_weights.keys()))
                            expert_ids = [f"E{i}" for i in range(num_experts)]
                            
                            heatmap_data = [task_total_weights[task] for task in tasks]
                            
                            try:
                                fig, ax = plt.subplots(figsize=(10, max(6, len(tasks))))
                                sns.heatmap(
                                    np.array(heatmap_data),
                                    xticklabels=expert_ids,
                                    yticklabels=tasks,
                                    annot=True, fmt=".3f", cmap="viridis", ax=ax
                                )

                                ax.set_title(f"Expert Routing Weight Distribution @ step {global_step}\nLayer: {layer_name}")
                                plt.tight_layout()
                                
                                wandb.log(
                                    {f"type_routing_distribution/{layer_name}": wandb.Image(fig)},
                                    step=global_step
                                )
                                plt.close(fig)
                            except ImportError:
                                logger.warning("matplotlib/seaborn not found. Skipping heatmap.")
            accelerator.wait_for_everyone()
            
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
