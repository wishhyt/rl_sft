from __future__ import annotations

import contextlib
import hashlib
import json
import time
from collections import defaultdict
from concurrent import futures
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from .bootstrap import bootstrap
from .data import build_dataloaders
from .pipeline import (
    build_pipeline,
    encode_prompts,
    load_unet,
    negative_prompt_embeds,
    save_unet,
)
from .patches.ddim_with_logprob import ddim_step_with_logprob
from .patches.pipeline_with_logprob import pipeline_with_logprob
from .rewards import build_reward_fn
from .dpo_losses import LOSS_FUNCTIONS, get_adaptive_lose_l_scale

logger = get_logger(__name__)


# ==============================================================================
# Logging Utilities
# ==============================================================================

class JsonlLogger:
    """Simple JSONL file logger for metrics."""
    
    def __init__(self, output_dir: str) -> None:
        self.path = Path(output_dir) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


class WandbLogger:
    """WandB logger wrapper. Only logs on main process."""
    
    def __init__(self, config, accelerator: Accelerator) -> None:
        self.enabled = config.logging.use_wandb and accelerator.is_main_process
        self._wandb = None
        
        if self.enabled:
            try:
                import wandb
                self._wandb = wandb
                
                run_name = config.logging.wandb_run_name or config.run.name
                wandb.init(
                    project=config.logging.wandb_project,
                    entity=config.logging.wandb_entity,
                    name=run_name,
                    config=config.to_dict(),
                    dir=config.run.output_dir,
                    resume="allow",
                )
                logger.info(f"WandB initialized: {wandb.run.url}")
            except ImportError:
                logger.warning("wandb not installed. Disabling WandB logging.")
                self.enabled = False
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
                self.enabled = False
    
    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if self.enabled and self._wandb:
            self._wandb.log(payload, step=step)
    
    def finish(self) -> None:
        if self.enabled and self._wandb:
            self._wandb.finish()


class CombinedLogger:
    """Combines JSONL and WandB logging."""
    
    def __init__(self, config, accelerator: Accelerator) -> None:
        self.jsonl = JsonlLogger(config.run.output_dir)
        self.wandb = WandbLogger(config, accelerator)
        self.is_main_process = accelerator.is_main_process
    
    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        """Log payload to JSONL and WandB."""
        if self.is_main_process:
            self.jsonl.log(payload)
            self.wandb.log(payload, step=step)
    
    def finish(self) -> None:
        self.wandb.finish()


# ==============================================================================
# LR Scheduler
# ==============================================================================

def _build_lr_scheduler(config, optimizer, num_training_steps: int):
    """Build learning rate scheduler based on config."""
    from transformers import get_scheduler
    
    # Calculate warmup steps
    if config.scheduler.warmup_ratio > 0:
        warmup_steps = int(num_training_steps * config.scheduler.warmup_ratio)
    else:
        warmup_steps = config.scheduler.warmup_steps
    
    scheduler_name = config.scheduler.name
    if scheduler_name == "cosine":
        scheduler_type = "cosine"
    elif scheduler_name == "linear":
        scheduler_type = "linear"
    elif scheduler_name == "constant":
        scheduler_type = "constant_with_warmup"
    else:
        logger.warning(f"Unknown scheduler '{scheduler_name}', using cosine")
        scheduler_type = "cosine"
    
    return get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )


# ==============================================================================
# Helper Functions  
# ==============================================================================

# Bootstrap once at module level or provide a function
def get_stat_tracker(config):
    bootstrap()
    from flow_grpo.stat_tracking import PerPromptStatTracker
    return PerPromptStatTracker(config.training.global_std) if config.training.per_prompt_stat_tracking else None


def _inference_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _build_accelerator(config, num_train_timesteps: int) -> Accelerator:
    project_config = ProjectConfiguration(
        project_dir=config.run.output_dir,
        automatic_checkpoint_naming=True,
        total_limit=config.run.checkpoint_limit,
    )
    return Accelerator(
        mixed_precision=config.precision.mixed_precision,
        project_config=project_config,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps * num_train_timesteps,
    )


def _prepare_optimizer(config, params):
    # Filter only trainable parameters (crucial for LoRA)
    trainable_params = list(filter(lambda p: p.requires_grad, params))
    
    if config.training.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError("Install bitsandbytes for 8-bit Adam") from exc
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    return optimizer_cls(
        trainable_params,
        lr=config.training.learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        weight_decay=config.training.adam_weight_decay,
        eps=config.training.adam_epsilon,
    )


def _prompt_generators(prompts: list[str], base_seed: int, device: torch.device) -> list[torch.Generator]:
    generators = []
    for prompt in prompts:
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        prompt_hash = int.from_bytes(digest[:4], "big")
        seed = (base_seed + prompt_hash) % (2**31)
        gen = torch.Generator(device=device).manual_seed(seed)
        generators.append(gen)
    return generators


def _save_state(
    config, 
    accelerator, 
    pipeline, 
    unet, 
    optimizer, 
    epoch, 
    global_step,
    lr_scheduler=None,
    is_best: bool = False,
    is_final: bool = False,
) -> None:
    """Save checkpoint with optional best/final markers."""
    if not accelerator.is_main_process:
        return
    
    if is_best:
        ckpt_root = Path(config.run.output_dir) / "checkpoints" / "best"
    elif is_final:
        ckpt_root = Path(config.run.output_dir) / "checkpoints" / "final"
    else:
        ckpt_root = Path(config.run.output_dir) / "checkpoints" / f"epoch_{epoch}"
    
    ckpt_root.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(unet)
    save_unet(pipeline, unwrapped, str(ckpt_root / "unet"), config.model.use_lora)
    torch.save(optimizer.state_dict(), ckpt_root / "optimizer.pt")
    
    if lr_scheduler is not None:
        torch.save(lr_scheduler.state_dict(), ckpt_root / "scheduler.pt")
    
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "config": config.to_dict(),
    }
    (ckpt_root / "trainer_state.json").write_text(
        json.dumps(state, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Saved checkpoint to {ckpt_root}", main_process_only=True)


def _save_config(config, output_dir: str) -> None:
    """Save resolved config to output directory."""
    config_path = Path(output_dir) / "config.json"
    config_path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _load_state(config, pipeline, unet, optimizer, lr_scheduler=None) -> tuple[int, int]:
    if not config.run.resume_from:
        return 0, 0
    ckpt_root = Path(config.run.resume_from)
    if not ckpt_root.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_root}")

    load_unet(pipeline, unet, str(ckpt_root / "unet"), config.model.use_lora)
    optimizer_state = torch.load(ckpt_root / "optimizer.pt", map_location="cpu")
    optimizer.load_state_dict(optimizer_state)
    
    if lr_scheduler is not None:
        scheduler_path = ckpt_root / "scheduler.pt"
        if scheduler_path.exists():
            scheduler_state = torch.load(scheduler_path, map_location="cpu")
            lr_scheduler.load_state_dict(scheduler_state)

    state_path = ckpt_root / "trainer_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return int(state.get("epoch", 0)), int(state.get("global_step", 0))
    return 0, 0


def _compute_advantages_grpo(config, accelerator, rewards, num_image_per_prompt: int):
    """
    GRPO-style advantage computation: group by prompt and compute advantages within each group.
    This matches the official DanceGRPO implementation where each prompt's multiple generations
    are compared against each other, not globally.
    
    Args:
        config: Configuration object
        accelerator: Accelerator instance
        rewards: Reward tensor of shape [batch_size * num_image_per_prompt]
        num_image_per_prompt: Number of images generated per prompt
    
    Returns:
        advantages: Advantage tensor with same shape as rewards
    """
    # Gather rewards from all processes
    gathered_rewards = accelerator.gather(rewards).cpu()
    
    # Reshape to [num_prompts, num_image_per_prompt]
    n_prompts = len(gathered_rewards) // num_image_per_prompt
    grouped_rewards = gathered_rewards.reshape(n_prompts, num_image_per_prompt)
    
    # Compute mean and std within each group (per-prompt)
    group_mean = grouped_rewards.mean(dim=1, keepdim=True)
    group_std = grouped_rewards.std(dim=1, keepdim=True) + 1e-8
    
    # Normalize within each group
    advantages = (grouped_rewards - group_mean) / group_std
    advantages = advantages.reshape(-1)
    
    # Distribute back to processes
    advantages = advantages.reshape(accelerator.num_processes, -1)[
        accelerator.process_index
    ].to(accelerator.device)
    
    return advantages


def _compute_advantages(config, accelerator, pipeline, prompt_ids, rewards, stat_tracker, mode: str):
    """Legacy advantage computation for backward compatibility with stat_tracker."""
    gathered_rewards = {key: accelerator.gather(value) for key, value in rewards.items()}
    gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}

    prompt_ids_gather = accelerator.gather(prompt_ids).cpu().numpy()
    prompts = pipeline.tokenizer.batch_decode(prompt_ids_gather, skip_special_tokens=True)

    if stat_tracker is not None:
        advantages = stat_tracker.update(prompts, gathered_rewards["avg"], type=mode)
    else:
        advantages = (gathered_rewards["avg"] - gathered_rewards["avg"].mean()) / (
            gathered_rewards["avg"].std() + 1e-4
        )

    advantages = torch.as_tensor(advantages)
    advantages = advantages.reshape(accelerator.num_processes, -1)[
        accelerator.process_index
    ].to(accelerator.device)
    return advantages



def _get_grad_norm(model) -> float:
    """Compute gradient norm for logging."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def _evaluate(
    config,
    accelerator,
    pipeline,
    reward_fn,
    test_dataloader,
    autocast,
    epoch: int = 0,
) -> dict[str, float]:
    """Run evaluation and return metrics."""
    pipeline.unet.eval()
    all_rewards = defaultdict(list)
    eval_results = []

    # Use DDIM scheduler for evaluation sampling/logprob to satisfy patch helpers
    from diffusers import DDIMScheduler
    original_scheduler = pipeline.scheduler
    pipeline.scheduler = DDIMScheduler.from_config(original_scheduler.config)

    for batch_idx, batch in enumerate(test_dataloader):
        prompt_embeds, _ = encode_prompts(pipeline, batch.prompts, accelerator.device)
        neg_prompt_embeds = None
        if config.sampling.eval_guidance_scale > 1.0:
            neg_prompt_embeds = negative_prompt_embeds(
                pipeline, len(batch.prompts), accelerator.device
            )
        with autocast():
            images, _, _, _ = pipeline_with_logprob(
                pipeline,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_prompt_embeds,
                num_inference_steps=config.sampling.eval_num_steps,
                guidance_scale=config.sampling.eval_guidance_scale,
                output_type="pt",
                height=config.sampling.resolution,
                width=config.sampling.resolution,
            )
            # Keep images in float32 after VAE decoding to avoid color artifacts
            # VAE is forced to float32 (line 593/849), converting to fp16/bf16 causes gray haze
            if isinstance(images, torch.Tensor):
                images = images.to(dtype=torch.float32)
        scores = reward_fn(images, batch.prompts, batch.metadata, only_strict=False)
        for key, value in scores.items():
            all_rewards[key].append(torch.as_tensor(value, device=accelerator.device))
        
        # Collect individual results for saving
        # Postprocess images to PIL
        pil_images = pipeline.image_processor.postprocess(images, output_type="pil")
        
        ckpt_root = Path(config.run.output_dir) / "checkpoints" / f"epoch_{epoch}"
        eval_img_dir = ckpt_root / "eval_images"
        if accelerator.is_main_process:
            eval_img_dir.mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()

        for i in range(len(batch.prompts)):
            res = {
                "prompt": batch.prompts[i],
                "metadata": batch.metadata[i],
            }
            for key, value in scores.items():
                res[key] = value[i]
            eval_results.append(res)

            # Save generated image - include rank to avoid collisions
            prompt_snippet = batch.prompts[i][:30].replace(" ", "_").replace("/", "_")
            img_path = eval_img_dir / f"rank{accelerator.process_index}_b{batch_idx}_i{i}_{prompt_snippet}.png"
            pil_images[i].save(img_path)

    metrics = {}
    for key, value in all_rewards.items():
        local_tensor = torch.cat(value).float()
        # gather_for_metrics handles potential padding and varied sizes
        all_results = accelerator.gather_for_metrics(local_tensor)
        if accelerator.is_main_process:
            metrics[f"eval/{key}"] = all_results.mean().item()
            
    # Gather all individual evaluation results from all processes
    from accelerate.utils import gather_object
    gathered_results = gather_object(eval_results)

    if accelerator.is_main_process:
        # Flatten the list of lists from gather_object
        all_eval_results = []
        for partial_list in gathered_results:
            all_eval_results.extend(partial_list)

        logger.info(f"Eval metrics: {metrics}", main_process_only=True)
        # Save evaluation table
        eval_base_dir = config.run.eval_output_dir or (Path(config.run.output_dir) / "eval")
        eval_dir = Path(eval_base_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_path = eval_dir / f"eval_epoch_{epoch:05d}.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(all_eval_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(all_eval_results)} evaluation results to {eval_path}", main_process_only=True)

    # Restore original scheduler (DDPM) after eval
    pipeline.scheduler = original_scheduler

    return metrics


def _prepare_samples(
    config,
    accelerator,
    pipeline,
    train_iter,
    train_dataloader,
    train_sampler,
    reward_fn,
    executor,
    autocast,
    return_prev_sample_mean: bool,
    epoch: int,
):
    samples = []
    for i in range(config.sampling.num_batches_per_epoch):
        logger.info(f"    [Sampling] Batch {i+1}/{config.sampling.num_batches_per_epoch}...", main_process_only=True)
        
        train_sampler.set_epoch(epoch * config.sampling.num_batches_per_epoch + i)
        try:
            batch = next(train_iter)
        except StopIteration:
            logger.info("    [Sampling] Restarting dataloader iterator...", main_process_only=True)
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        prompt_embeds, prompt_ids = encode_prompts(
            pipeline, batch.prompts, accelerator.device
        )
        neg_prompt_embeds = None
        if config.sampling.guidance_scale > 1.0:
            neg_prompt_embeds = negative_prompt_embeds(
                pipeline, len(batch.prompts), accelerator.device
            )
        generator = None
        if config.sampling.same_latent:
            generator = _prompt_generators(
                batch.prompts,
                base_seed=epoch * 10000 + i,
                device=accelerator.device,
            )

        num_images = config.sampling.num_image_per_prompt
        with autocast():
            with torch.no_grad():
                if return_prev_sample_mean:
                    images, _, latents, log_probs, prev_sample_mean = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=neg_prompt_embeds,
                        num_inference_steps=config.sampling.num_steps,
                        guidance_scale=config.sampling.guidance_scale,
                        eta=config.sampling.eta,
                        generator=generator,
                        num_images_per_prompt=num_images,
                        output_type="pt",
                        height=config.sampling.resolution,
                        width=config.sampling.resolution,
                        return_prev_sample_mean=True,
                    )
                else:
                    images, _, latents, log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=neg_prompt_embeds,
                        num_inference_steps=config.sampling.num_steps,
                        guidance_scale=config.sampling.guidance_scale,
                        eta=config.sampling.eta,
                        generator=generator,
                        num_images_per_prompt=num_images,
                        output_type="pt",
                        height=config.sampling.resolution,
                        width=config.sampling.resolution,
                    )
                    prev_sample_mean = None

        logger.info(f"    [Sampling] Batch {i+1} image generated. Submitting to reward...", main_process_only=True)

        latents = torch.stack(latents, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        if prev_sample_mean is not None:
            prev_sample_mean = torch.stack(prev_sample_mean, dim=1)

        timesteps = pipeline.scheduler.timesteps.repeat(len(batch.prompts) * num_images, 1)

        # Expand prompts and metadata for reward calculation
        expanded_prompts = [p for p in batch.prompts for _ in range(num_images)]
        expanded_metadata = [m for m in batch.metadata for _ in range(num_images)]
        expanded_prompt_ids = prompt_ids.repeat_interleave(num_images, dim=0)

        reward_future = executor.submit(
            reward_fn,
            images,
            expanded_prompts,
            expanded_metadata,
            only_strict=True,
        )
        # Yield control for threads
        time.sleep(0.01)

        samples.append(
            {
                "prompt_ids": expanded_prompt_ids,
                "prompt_embeds": prompt_embeds.repeat_interleave(num_images, dim=0),
                "timesteps": timesteps,
                "latents": latents[:, :-1],
                "next_latents": latents[:, 1:],
                "log_probs": log_probs,
                "rewards": reward_future,
                "prev_sample_mean": prev_sample_mean,
            }
        )

    logger.info(f"    [Sampling] All batches submitted. Waiting for rewards...", main_process_only=True)

    for idx, sample in enumerate(samples):
        try:
            rewards = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }
        except Exception as e:
            logger.error(f"      [Reward] Batch {idx+1} reward calculation failed: {e}", main_process_only=True)
            raise e
    
    logger.info(f"    [Sampling] All rewards collected.", main_process_only=True)

    merged = {}
    for key in samples[0].keys():
        if key == "prev_sample_mean" and samples[0][key] is None:
            merged[key] = None
            continue
        if key == "rewards":
            merged[key] = {
                sub_key: torch.cat([s[key][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][key]
            }
        else:
            merged[key] = torch.cat([s[key] for s in samples], dim=0)
    return merged


# ==============================================================================
# GRPO Training
# ==============================================================================

def train_grpo(config) -> None:
    """GRPO training loop."""
    repo_paths = bootstrap()
    config.resolve(repo_paths)
    config.validate()
    Path(config.run.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save resolved config
    _save_config(config, config.run.output_dir)

    num_train_timesteps = max(1, int(config.sampling.num_steps * config.training.timestep_fraction))
    accelerator = _build_accelerator(config, num_train_timesteps)

    # BLOCK STDOUT/STDERR from dependencies to reduce clutter
    import os
    import warnings
    warnings.filterwarnings("ignore")
    os.environ["TQDM_DISABLE"] = "1" # Disable common tqdm from libs

    if config.precision.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    logger.info(config.to_dict())
    set_seed(config.run.seed, device_specific=True)

    inference_dtype = _inference_dtype(config.precision.mixed_precision)
    pipeline, unet = build_pipeline(config, accelerator.device, inference_dtype)
    pipeline.set_progress_bar_config(disable=True)

    optimizer = _prepare_optimizer(config, unet.parameters())

    reward_fn = build_reward_fn(config, accelerator.device)
    train_dataloader, test_dataloader, train_sampler = build_dataloaders(config, accelerator)

    # Calculate total training steps for LR scheduler
    total_steps = config.run.num_epochs * config.sampling.num_batches_per_epoch * config.training.num_inner_epochs
    lr_scheduler = _build_lr_scheduler(config, optimizer, total_steps)

    start_epoch, global_step = _load_state(config, pipeline, unet, optimizer, lr_scheduler)
    
    # Force VAE to float32 for training stability
    pipeline.vae.to(accelerator.device, dtype=torch.float32)

    # Prepare for distributed training
    if config.training.mode == "sft" and config.dataset.name == "spright":
        unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)
    else:
        unet, optimizer, lr_scheduler, train_dataloader, test_dataloader = accelerator.prepare(
            unet, optimizer, lr_scheduler, train_dataloader, test_dataloader
        )

    autocast = contextlib.nullcontext if config.model.use_lora else accelerator.autocast
    executor = futures.ThreadPoolExecutor(max_workers=config.reward.max_workers)
    
    # Initialize loggers
    metrics_logger = CombinedLogger(config, accelerator)
    stat_tracker = get_stat_tracker(config)
    
    # Best model tracking
    best_reward = float("-inf")
    
    # Track total processed samples (prompts in GRPO)
    processed_sample = 0

    train_iter = iter(train_dataloader)
    if accelerator.is_main_process:
        logger.info(f"Training started. Epochs: {start_epoch}-{config.run.num_epochs}. Processes: {accelerator.num_processes}", main_process_only=True)
    
    progress_bar = tqdm(
        range(start_epoch, config.run.num_epochs),
        desc="Epochs",
        disable=not accelerator.is_main_process,
    )
    
    for epoch in progress_bar:
        # Evaluation
        if epoch % config.run.eval_freq == 0 and epoch > 0:
            logger.info(f"Evaluating at epoch {epoch}...")
            eval_metrics = _evaluate(config, accelerator, pipeline, reward_fn, test_dataloader, autocast, epoch)
            if eval_metrics:
                eval_metrics["epoch"] = epoch
                eval_metrics["global_step"] = global_step
                metrics_logger.log(eval_metrics, step=global_step)
                
                # Check for best model
                avg_reward = eval_metrics.get("eval/avg", float("-inf"))
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    logger.info(f"New best reward: {best_reward:.4f}")
                    _save_state(config, accelerator, pipeline, unet, optimizer, epoch, global_step, lr_scheduler, is_best=True)

        # Save checkpoint
        if epoch % config.run.save_freq == 0 and epoch > 0:
            logger.info(f"Saving state at epoch {epoch}...")
            _save_state(config, accelerator, pipeline, unet, optimizer, epoch, global_step, lr_scheduler)

        # Sampling phase
        pipeline.unet.eval()
        logger.info(f"\n[Epoch {epoch}] Phase: Sampling samples...", main_process_only=True)
        
        samples = _prepare_samples(
            config, accelerator, pipeline, train_iter, train_dataloader, train_sampler,
            reward_fn, executor, autocast,
            return_prev_sample_mean=config.grpo.guard,
            epoch=epoch,
        )
        
        # Compute advantages using GRPO-style grouping (per-prompt comparison)
        logger.info(f"[Epoch {epoch}] Phase: Computing advantages...", main_process_only=True)
        
        advantages = _compute_advantages_grpo(
            config,
            accelerator,
            samples["rewards"]["avg"],
            config.sampling.num_image_per_prompt,
        )

        
        # Training phase
        pipeline.unet.train()
        logger.info(f"[Epoch {epoch}] Phase: Training model...", main_process_only=True)
        
        info = defaultdict(list)
        num_timesteps = samples["timesteps"].shape[1]
        
        inner_pbar = tqdm(
            range(config.training.num_inner_epochs),
            desc=f"  Epoch {epoch} Training",
            leave=False,
            disable=not accelerator.is_main_process,
        )

        for inner_epoch in inner_pbar:
            num_samples = samples["latents"].shape[0]
            indices = torch.arange(num_samples, device=accelerator.device)
            log_info = {}  # Initialize to avoid UnboundLocalError
            
            for t in range(num_train_timesteps):
                # Process in minibatches to avoid OOM
                for start_idx in range(0, num_samples, config.training.batch_size):
                    end_idx = min(start_idx + config.training.batch_size, num_samples)
                    mb_idx = indices[start_idx:end_idx]

                    if accelerator.is_main_process:
                        inner_pbar.set_postfix({
                            "step": f"{t}/{num_train_timesteps}", 
                            "mb": f"{start_idx//config.training.batch_size}/{num_samples//config.training.batch_size}",
                            "loss": info["loss"][-1].item() if info["loss"] else "N/A"
                        })
                    
                    with accelerator.accumulate(unet):
                        with autocast():
                            latents_t = samples["latents"][mb_idx, t]
                            next_latents_t = samples["next_latents"][mb_idx, t]
                            timesteps_t = samples["timesteps"][mb_idx, t]
                            prompt_embeds = samples["prompt_embeds"][mb_idx]
                            
                            # CFG support during training (matching official DanceGRPO)
                            if config.training.cfg and config.sampling.guidance_scale > 1.0:
                                # Prepare negative prompt embeds for this minibatch
                                neg_embeds = negative_prompt_embeds(
                                    pipeline, len(mb_idx), accelerator.device
                                )
                                # Concat negative and positive prompts
                                embeds = torch.cat([neg_embeds, prompt_embeds])
                                
                                # Forward pass with CFG
                                noise_pred = unet(
                                    torch.cat([latents_t] * 2),
                                    torch.cat([timesteps_t] * 2),
                                    encoder_hidden_states=embeds,
                                ).sample
                                
                                # Split and apply CFG
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = (
                                    noise_pred_uncond
                                    + config.sampling.guidance_scale * (noise_pred_text - noise_pred_uncond)
                                )
                            else:
                                # No CFG: conditional forward pass only
                                noise_pred = unet(
                                    latents_t,
                                    timesteps_t,
                                    encoder_hidden_states=prompt_embeds,
                                ).sample
                            
                            # Compute current log prob
                            _, current_log_prob = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred,
                                timesteps_t,
                                latents_t,
                                eta=config.sampling.eta,
                                prev_sample=next_latents_t,

                            )
                            
                            # GRPO loss
                            old_log_prob = samples["log_probs"][mb_idx, t]
                            ratio = torch.exp(current_log_prob - old_log_prob)
                            
                            # Clipping
                            mb_advantages = advantages[mb_idx]
                            clipped_advantages = torch.clamp(
                                mb_advantages, 
                                -config.training.adv_clip_max, 
                                config.training.adv_clip_max
                            )
                            
                            unclipped_loss = -clipped_advantages * ratio
                            clipped_ratio = torch.clamp(
                                ratio, 
                                1 - config.training.clip_range, 
                                1 + config.training.clip_range
                            )
                            clipped_loss = -clipped_advantages * clipped_ratio
                            loss = torch.max(unclipped_loss, clipped_loss).mean()
                            
                            # Diagnostics
                            with torch.no_grad():
                                is_clipped = (ratio < 1 - config.training.clip_range) | (ratio > 1 + config.training.clip_range)
                                info["clip_fraction"].append(is_clipped.float().mean())
                                approx_kl = ratio - 1 - torch.log(ratio)
                                info["approx_kl"].append(approx_kl.mean())

                            # GRPO-Guard adjustment
                            if config.grpo.guard and samples["prev_sample_mean"] is not None:
                                prev_mean = samples["prev_sample_mean"][mb_idx, t]
                                guard_loss = F.mse_loss(noise_pred, prev_mean) * config.grpo.guard_scale
                                loss = loss + guard_loss
                                info["guard_loss"].append(guard_loss.detach())
                        
                        info["loss"].append(loss.detach())
                        info["ratio"].append(ratio.mean().detach())
                        info["ratio_max"].append(ratio.max().detach())
                        info["ratio_min"].append(ratio.min().detach())
                        
                        accelerator.backward(loss)
                        
                        if accelerator.sync_gradients:
                            grad_norm = accelerator.clip_grad_norm_(unet.parameters(), config.training.max_grad_norm)
                            if config.logging.log_grad_norm:
                                info["grad_norm"].append(torch.as_tensor(grad_norm, device=accelerator.device))
                        
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                if accelerator.sync_gradients:
                    # Log metrics
                    log_info = {k: torch.mean(torch.stack(v)).item() for k, v in info.items() if v}
                    log_info.update({
                        "epoch": epoch,
                        "inner_epoch": inner_epoch,
                        "timestep": t,
                        "global_step": global_step,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_reward": samples["rewards"]["avg"].mean().item(),
                    })

                    # Advantage stats
                    log_info.update({
                        "advantage/mean": advantages.mean().item(),
                        "advantage/std": advantages.std().item(),
                        "advantage/max": advantages.max().item(),
                        "advantage/min": advantages.min().item(),
                    })

                    # Individual Reward components
                    for r_key, r_val in samples["rewards"].items():
                        if r_key != "avg":
                            log_info[f"reward/{r_key}"] = r_val.mean().item()
                    
                    # Update processed_sample counter (number of prompts processed)
                    batch_size = len(samples["prompt_ids"]) // config.sampling.num_image_per_prompt
                    processed_sample += batch_size
                    log_info["processed_sample"] = processed_sample
                    
                    if global_step % config.logging.log_every_n_steps == 0:
                        metrics_logger.log(log_info, step=global_step)
                    
                    info = defaultdict(list)
                    global_step += 1

            # Only update progress bar if we have logged info
            if log_info:
                progress_bar.set_postfix({"loss": log_info.get("loss", 0), "reward": log_info.get("avg_reward", 0)})


    # Save final model
    _save_state(config, accelerator, pipeline, unet, optimizer, config.run.num_epochs, global_step, lr_scheduler, is_final=True)
    metrics_logger.finish()
    logger.info("Training complete!")


# ==============================================================================
# SFT Training
# ==============================================================================

def train_sft(config) -> None:
    """SFT (Supervised Fine-Tuning) training loop."""
    repo_paths = bootstrap()
    config.resolve(repo_paths)
    config.validate()
    Path(config.run.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save resolved config
    _save_config(config, config.run.output_dir)

    accelerator = _build_accelerator(config, 1)

    if config.precision.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    logger.info(config.to_dict())
    set_seed(config.run.seed, device_specific=True)

    inference_dtype = _inference_dtype(config.precision.mixed_precision)
    pipeline, unet = build_pipeline(config, accelerator.device, inference_dtype)
    pipeline.set_progress_bar_config(disable=True)

    optimizer = _prepare_optimizer(config, unet.parameters())

    reward_fn = build_reward_fn(config, accelerator.device)
    train_dataloader, test_dataloader, train_sampler = build_dataloaders(config, accelerator)

    # Calculate total training steps for LR scheduler
    total_steps = config.run.num_epochs * config.sampling.num_batches_per_epoch * config.training.num_inner_epochs
    lr_scheduler = _build_lr_scheduler(config, optimizer, total_steps)

    start_epoch, global_step = _load_state(config, pipeline, unet, optimizer, lr_scheduler)

    # Force VAE to float32 for training stability
    pipeline.vae.to(accelerator.device, dtype=torch.float32)

    if config.training.mode == "sft" and config.dataset.name == "spright":
        unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)
    else:
        unet, optimizer, lr_scheduler, train_dataloader, test_dataloader = accelerator.prepare(
            unet, optimizer, lr_scheduler, train_dataloader, test_dataloader
        )

    autocast = contextlib.nullcontext if config.model.use_lora else accelerator.autocast
    
    # Initialize loggers
    metrics_logger = CombinedLogger(config, accelerator)
    
    # Best model tracking
    best_loss = float("inf")
    
    # Track total processed samples (image-text pairs in SFT)
    processed_sample = 0

    # Compute timestep range for training
    scheduler_timesteps = getattr(
        pipeline.scheduler.config, "num_train_timesteps", pipeline.scheduler.num_train_timesteps
    )
    max_train_timesteps = max(1, int(scheduler_timesteps * config.training.timestep_fraction))
    min_train_timestep = max(0, scheduler_timesteps - max_train_timesteps)

    train_iter = iter(train_dataloader)
    logger.info(f"Training started. Epochs: {start_epoch}-{config.run.num_epochs}. Processes: {accelerator.num_processes}", main_process_only=True)
    progress_bar = tqdm(
        range(start_epoch, config.run.num_epochs),
        desc="Epochs",
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in progress_bar:
        # Evaluation
        if epoch % config.run.eval_freq == 0 and epoch > 0:
            logger.info(f"Evaluating at epoch {epoch}...")
            eval_metrics = _evaluate(config, accelerator, pipeline, reward_fn, test_dataloader, autocast, epoch)
            if eval_metrics:
                eval_metrics["epoch"] = epoch
                eval_metrics["global_step"] = global_step
                metrics_logger.log(eval_metrics, step=global_step)

        # Save checkpoint
        if epoch % config.run.save_freq == 0 and epoch > 0:
            logger.info(f"Saving state at epoch {epoch}...")
            _save_state(config, accelerator, pipeline, unet, optimizer, epoch, global_step, lr_scheduler)

        pipeline.unet.train()
        info = defaultdict(list)
        epoch_losses = []
        
        batch_progress = tqdm(
            range(config.sampling.num_batches_per_epoch),
            desc="Batches",
            leave=False,
            disable=not accelerator.is_local_main_process,
        )
        for i in batch_progress:
            train_sampler.set_epoch(epoch * config.sampling.num_batches_per_epoch + i)
            batch = next(train_iter)
            if batch.images is None:
                raise ValueError("SFT requires paired image-text data in *_pairs.jsonl")

            prompt_embeds, _ = encode_prompts(
                pipeline, batch.prompts, accelerator.device
            )

            with torch.no_grad():
                # Check if images are already tensors (from bucket_collate_fn)
                if isinstance(batch.images, torch.Tensor):
                    # bucket_collate_fn already applied Normalize([0.5], [0.5]) -> [-1, 1]
                    pixel_values = batch.images.to(accelerator.device, dtype=torch.float32)
                else:
                    # image_processor.preprocess outputs [-1, 1] by default for VaeImageProcessor
                    pixel_values = pipeline.image_processor.preprocess(
                        batch.images,
                        height=config.sampling.resolution,
                        width=config.sampling.resolution,
                    ).to(accelerator.device, dtype=torch.float32)
                
                # Encode in float32
                latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * pipeline.vae.config.scaling_factor
                latents = latents.to(dtype=inference_dtype)

            for inner_epoch in range(config.training.num_inner_epochs):
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    min_train_timestep,
                    scheduler_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                    dtype=torch.long,
                )
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

                with accelerator.accumulate(unet):
                    with autocast():
                        # Point 2: SFT training should only use conditional forward pass (NO CFG)
                        # Matches official train_text_to_image.py logic
                        noise_pred = unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=prompt_embeds,
                        ).sample

                        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                    
                    info["loss"].append(loss.detach())
                    epoch_losses.append(loss.detach())
                    
                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(unet.parameters(), config.training.max_grad_norm)
                        if config.logging.log_grad_norm:
                            info["grad_norm"].append(torch.tensor(grad_norm, device=accelerator.device))
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    log_info = {k: torch.mean(torch.stack(v)).item() for k, v in info.items() if v}
                    log_info.update({
                        "epoch": epoch,
                        "inner_epoch": inner_epoch,
                        "global_step": global_step,
                        "lr": lr_scheduler.get_last_lr()[0],
                    })
                    
                    # Update processed_sample counter (number of image-text pairs processed)
                    batch_size = len(batch.prompts)
                    processed_sample += batch_size
                    log_info["processed_sample"] = processed_sample
                    
                    if global_step % config.logging.log_every_n_steps == 0:
                        metrics_logger.log(log_info, step=global_step)
                    
                    info = defaultdict(list)
                    global_step += 1

        # Check for best model based on epoch loss
        if epoch_losses:
            avg_epoch_loss = torch.mean(torch.stack(epoch_losses)).item()
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                logger.info(f"New best loss: {best_loss:.6f}")
                _save_state(config, accelerator, pipeline, unet, optimizer, epoch, global_step, lr_scheduler, is_best=True)
            
            progress_bar.set_postfix({"loss": avg_epoch_loss})

    # Save final model
    _save_state(config, accelerator, pipeline, unet, optimizer, config.run.num_epochs, global_step, lr_scheduler, is_final=True)
    metrics_logger.finish()
    logger.info("Training complete!")


# ==============================================================================
# DPO Training
# ==============================================================================

def train_dpo(config) -> None:
    """DPO (Direct Preference Optimization) training loop.
    
    Uses preference data (winner/loser image pairs) to directly optimize
    the diffusion model policy without explicit reward modeling.
    
    Reference: Diffusion-SDPO (https://github.com/AID-AI/Diffusion-SDPO)
    """
    from diffusers import UNet2DConditionModel
    
    repo_paths = bootstrap()
    config.resolve(repo_paths)
    config.validate()
    Path(config.run.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save resolved config
    _save_config(config, config.run.output_dir)

    accelerator = _build_accelerator(config, 1)
    logger.info("Building accelerator...", main_process_only=True)

    if config.precision.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    logger.info(config.to_dict(), main_process_only=True)
    set_seed(config.run.seed, device_specific=True)

    inference_dtype = _inference_dtype(config.precision.mixed_precision)
    logger.info("Building pipeline...", main_process_only=True)
    pipeline, unet = build_pipeline(config, accelerator.device, inference_dtype)
    pipeline.set_progress_bar_config(disable=True)

    # Load reference UNet (frozen copy for DPO)
    logger.info("Loading reference UNet...", main_process_only=True)
    ref_unet = UNet2DConditionModel.from_pretrained(
        config.model.name_or_path,
        revision=config.model.revision,
        subfolder="unet",
    )
    ref_unet.to(accelerator.device, dtype=inference_dtype)
    ref_unet.requires_grad_(False)
    ref_unet.eval()

    logger.info("Preparing optimizer and dataloaders...", main_process_only=True)
    optimizer = _prepare_optimizer(config, unet.parameters())
    train_dataloader, test_dataloader, train_sampler = build_dataloaders(config, accelerator)

    # Calculate total training steps for LR scheduler
    total_steps = config.run.num_epochs * config.sampling.num_batches_per_epoch * config.training.num_inner_epochs
    lr_scheduler = _build_lr_scheduler(config, optimizer, total_steps)

    start_epoch, global_step = _load_state(config, pipeline, unet, optimizer, lr_scheduler)

    # Force VAE to float32 for training stability
    pipeline.vae.to(accelerator.device, dtype=torch.float32)

    # Prepare for distributed training (WebDataset handles its own sharding)
    unet, optimizer, lr_scheduler = accelerator.prepare(
        unet, optimizer, lr_scheduler
    )

    autocast = contextlib.nullcontext if config.model.use_lora else accelerator.autocast
    
    # Initialize loggers
    metrics_logger = CombinedLogger(config, accelerator)
    
    # Track total processed samples (preference pairs in DPO)
    processed_sample = 0

    train_iter = iter(train_dataloader)
    logger.info(f"DPO Training started. Epochs: {start_epoch}-{config.run.num_epochs}. Processes: {accelerator.num_processes}", main_process_only=True)
    progress_bar = tqdm(
        range(start_epoch, config.run.num_epochs),
        desc="Epochs",
        disable=not accelerator.is_main_process,
    )
    
    # Get scheduler config
    noise_scheduler = pipeline.scheduler
    scheduler_timesteps = getattr(
        noise_scheduler.config, "num_train_timesteps", noise_scheduler.num_train_timesteps
    )
    
    for epoch in progress_bar:
        # Save checkpoint periodically
        if epoch % config.run.save_freq == 0 and epoch > 0:
            logger.info(f"Saving state at epoch {epoch}...")
            _save_state(config, accelerator, pipeline, unet, optimizer, epoch, global_step, lr_scheduler)

        pipeline.unet.train()
        info = defaultdict(list)
        epoch_losses = []
        implicit_acc_list = []
        
        batch_progress = tqdm(
            range(config.sampling.num_batches_per_epoch),
            desc="Batches",
            leave=False,
            disable=not accelerator.is_main_process,
        )
        
        for batch_idx in batch_progress:
            train_sampler.set_epoch(epoch * config.sampling.num_batches_per_epoch + batch_idx)
            batch = next(train_iter)
            
            # DPO batch has pixel_values_w (winner) and pixel_values_l (loser)
            # Images are already preprocessed by dpo_collate_fn (normalized to [-1, 1])
            pixel_values_w = batch.pixel_values_w.to(accelerator.device, dtype=inference_dtype)
            pixel_values_l = batch.pixel_values_l.to(accelerator.device, dtype=inference_dtype)
            prompts = batch.prompts

            # Encode prompts
            prompt_embeds, _ = encode_prompts(pipeline, prompts, accelerator.device)
            
            # Encode images to latent space
            with torch.no_grad():
                # Force VAE to float32 for DPO encoding to avoid gray haze
                latents_w = pipeline.vae.encode(pixel_values_w.to(dtype=torch.float32)).latent_dist.sample()
                latents_w = latents_w * pipeline.vae.config.scaling_factor
                latents_w = latents_w.to(dtype=inference_dtype)

                latents_l = pipeline.vae.encode(pixel_values_l.to(dtype=torch.float32)).latent_dist.sample()
                latents_l = latents_l * pipeline.vae.config.scaling_factor
                latents_l = latents_l.to(dtype=inference_dtype)

            # Concatenate winner and loser latents for batch processing
            # If KTO, we treat them as independent samples (2*B batch) with labels
            latents = torch.cat([latents_w, latents_l], dim=0)
            
            # Create labels for KTO (1 for winner, 0 for loser)
            # labels: [1, 1, ..., 0, 0, ...]
            kto_labels = torch.cat([
                torch.ones(latents_w.shape[0], device=accelerator.device),
                torch.zeros(latents_l.shape[0], device=accelerator.device)
            ])
            
            encoder_hidden_states = prompt_embeds.repeat(2, 1, 1)
            
            bsz = latents_w.shape[0]

            # CRITICAL: Sample noise and timesteps ONCE per batch (outside inner_epoch loop)
            # This matches ALL reference DPO implementations:
            # - DiffusionDPO: samples noise outside training loop
            # - Curriculum-DPO: samples noise outside training loop  
            # - DSPO: samples noise outside training loop
            # - Diffusion-SDPO: samples noise outside training loop
            # - diffusion-kto: samples noise outside training loop
            # Unlike SFT which resamples noise in each inner_epoch for data augmentation,
            # DPO requires the SAME noise across inner epochs to properly compute
            # the preference-based loss with consistent comparisons.
            noise = torch.randn_like(latents_w)
            noise = torch.cat([noise, noise], dim=0)  # Same noise for winner and loser
            
            timesteps = torch.randint(
                0, scheduler_timesteps,
                (bsz,), device=latents.device, dtype=torch.long
            )
            timesteps = torch.cat([timesteps, timesteps])  # Same timesteps for pairs

            for inner_epoch in range(config.training.num_inner_epochs):
                # Use the SAME noise and timesteps for all inner epochs
                # Add noise to latents (using shared noise and timesteps)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                
                target = noise  # epsilon prediction
                with accelerator.accumulate(unet):
                    with autocast():
                        # Model prediction
                        model_pred = unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                        ).sample
                        
                        # Compute MSE losses for each sample
                        model_losses = (model_pred - target).pow(2).mean(dim=[1, 2, 3])
                        model_losses_w, model_losses_l = model_losses.chunk(2)
                        raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
                        model_diff = model_losses_w - model_losses_l
                        
                        # Reference model prediction
                        with torch.no_grad():
                            ref_pred = ref_unet(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states=encoder_hidden_states,
                            ).sample
                            ref_losses = (ref_pred - target).pow(2).mean(dim=[1, 2, 3])
                            ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                            ref_diff = ref_losses_w - ref_losses_l
                            raw_ref_loss = ref_losses.mean()
                        
                        # DPO loss computation
                        scale_term = -0.5 * config.dpo.beta_dpo
                        inside_term = scale_term * (model_diff - ref_diff)
                        implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                        
                        if config.dpo.train_method == "kto":
                            # KTO uses strict label-based loss (no pairs in loss calc, but pairs in batch)
                            loss = LOSS_FUNCTIONS["kto"](model_losses, ref_losses, kto_labels, config)
                        elif config.dpo.train_method in LOSS_FUNCTIONS:
                            loss = LOSS_FUNCTIONS[config.dpo.train_method](
                                model_diff, ref_diff, model_pred, ref_pred, target, inside_term, config
                            )
                        else:
                            raise ValueError(f"Unknown DPO train_method: {config.dpo.train_method}")
                    
                    info["loss"].append(loss.detach())
                    info["model_mse"].append(raw_model_loss.detach())
                    info["ref_mse"].append(raw_ref_loss.detach())
                    info["implicit_acc"].append(implicit_acc.detach())
                    
                    # Enhanced DPO metrics
                    with torch.no_grad():
                        info["margin/model"].append(model_diff.mean().detach())
                        info["margin/ref"].append(ref_diff.mean().detach())
                        info["margin/relative"].append((model_diff - ref_diff).mean().detach())
                        info["loss_component/model_w"].append(model_losses_w.mean().detach())
                        info["loss_component/model_l"].append(model_losses_l.mean().detach())
                        info["loss_component/ref_w"].append(ref_losses_w.mean().detach())
                        info["loss_component/ref_l"].append(ref_losses_l.mean().detach())

                    epoch_losses.append(loss.detach())
                    implicit_acc_list.append(implicit_acc.detach())
                    
                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(unet.parameters(), config.training.max_grad_norm)
                        if config.logging.log_grad_norm:
                            info["grad_norm"].append(torch.as_tensor(grad_norm, device=accelerator.device))
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    log_info = {k: torch.mean(torch.stack(v)).item() for k, v in info.items() if v}
                    log_info.update({
                        "epoch": epoch,
                        "inner_epoch": inner_epoch,
                        "global_step": global_step,
                        "lr": lr_scheduler.get_last_lr()[0],
                    })
                    
                    # Update processed_sample counter (number of preference pairs processed)
                    batch_size = len(batch.prompts)
                    processed_sample += batch_size
                    log_info["processed_sample"] = processed_sample
                    
                    if global_step % config.logging.log_every_n_steps == 0:
                        metrics_logger.log(log_info, step=global_step)
                    
                    info = defaultdict(list)
                    global_step += 1
            
            # Curriculum Update (Per Epoch or Per Step depending on preference)
            # Here we do it per batch loop (effectively per epoch end if we wanted, but config says update_cl usually)
            pass

        # Curriculum update at end of epoch
        if config.dpo.curriculum_groups > 0 and hasattr(train_dataloader.dataset, "update_cl"):
            train_dataloader.dataset.update_cl()

        # Log epoch summary
        if epoch_losses:
            avg_epoch_loss = torch.mean(torch.stack(epoch_losses)).item()
            avg_implicit_acc = torch.mean(torch.stack(implicit_acc_list)).item()
            progress_bar.set_postfix({"loss": avg_epoch_loss, "acc": avg_implicit_acc})

    # Save final model
    _save_state(config, accelerator, pipeline, unet, optimizer, config.run.num_epochs, global_step, lr_scheduler, is_final=True)
    metrics_logger.finish()
    logger.info("DPO Training complete!")
