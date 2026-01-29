from __future__ import annotations

from typing import Tuple
import os

import torch
from diffusers import DDPMScheduler, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict


def build_pipeline(config, device: torch.device, inference_dtype: torch.dtype) -> Tuple[StableDiffusionPipeline, torch.nn.Module]:
    # Load VAE separately for maximum control over precision, just like the official script
    vae = AutoencoderKL.from_pretrained(
        config.model.name_or_path,
        subfolder="vae",
        revision=config.model.revision,
        torch_dtype=torch.float32, # Force float32 for stability
    )
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.model.name_or_path,
        vae=vae,
        revision=config.model.revision,
        torch_dtype=torch.float32, # Load everything in fp32 initially
    )
    
    # SD1.4 training scripts typically use DDPMScheduler
    pipeline.scheduler = DDPMScheduler.from_pretrained(
        config.model.name_or_path, 
        subfolder="scheduler"
    )
    
    pipeline.safety_checker = None

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.model.use_lora)

    # Move to device with specific dtypes
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=inference_dtype)
    
    # Memory optimizations
    try:
         pipeline.enable_xformers_memory_efficient_attention()
    except Exception:
         pipeline.enable_attention_slicing()

    if config.model.use_lora:
        pipeline.unet.to(device, dtype=inference_dtype)
        
        # Exact same target modules as train_text_to_image.py
        lora_config = LoraConfig(
            r=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha or config.model.lora_rank,
            init_lora_weights="gaussian",
            target_modules=config.model.lora_target_modules or ["to_k", "to_q", "to_v", "to_out.0"],
        )
        pipeline.unet.add_adapter(lora_config)
        # pipeline.unet.enable_gradient_checkpointing()
        
        unet = pipeline.unet
    else:
        pipeline.unet.to(device, dtype=inference_dtype)
        unet = pipeline.unet

    return pipeline, unet


def encode_prompts(pipeline: StableDiffusionPipeline, prompts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = pipeline.tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(device)
    prompt_embeds = pipeline.text_encoder(tokens)[0]
    return prompt_embeds, tokens


def negative_prompt_embeds(pipeline: StableDiffusionPipeline, batch_size: int, device: torch.device) -> torch.Tensor:
    tokens = pipeline.tokenizer(
        [""],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(device)
    embeds = pipeline.text_encoder(tokens)[0]
    return embeds.repeat(batch_size, 1, 1)


def save_unet(pipeline: StableDiffusionPipeline, unet: torch.nn.Module, output_dir: str, use_lora: bool) -> None:
    if use_lora:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        # Use standard PEFT saving
        lora_state_dict = get_peft_model_state_dict(unet)
        torch.save(lora_state_dict, os.path.join(output_dir, "custom_lora_weights.pt"))
    elif isinstance(unet, UNet2DConditionModel):
        unet.save_pretrained(output_dir)
    else:
        raise ValueError("Unexpected unet type for saving")


def load_unet(pipeline: StableDiffusionPipeline, unet: torch.nn.Module, input_dir: str, use_lora: bool) -> None:
    if use_lora:
        weight_path = os.path.join(input_dir, "custom_lora_weights.pt")
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location="cpu")
            set_peft_model_state_dict(unet, state_dict)
        else:
            print(f"Warning: LoRA weights not found at {weight_path}")

    elif isinstance(unet, UNet2DConditionModel):
        loaded = UNet2DConditionModel.from_pretrained(input_dir)
        unet.register_to_config(**loaded.config)
        unet.load_state_dict(loaded.state_dict())
    else:
        raise ValueError("Unexpected unet type for loading")

