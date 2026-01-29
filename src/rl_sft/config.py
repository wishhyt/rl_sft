from __future__ import annotations

import dataclasses
import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import default_dataset_root


ALLOWED_DATASETS = {"geneval", "geneval2", "ocr", "spright", "pickapic"}
ALLOWED_REWARDS = {"geneval", "geneval2", "ocr"}
ALLOWED_MODES = {"grpo", "sft", "dpo"}


@dataclass
class RunConfig:
    name: str = ""
    seed: int = 42
    log_dir: str = "logs"
    save_freq: int = 20
    eval_freq: int = 20
    num_epochs: int = 100000
    resume_from: str | None = None
    output_dir: str = ""
    eval_output_dir: str = ""
    checkpoint_limit: int = 5


@dataclass
class ModelConfig:
    name_or_path: str = "CompVis/stable-diffusion-v1-4"
    revision: str = "main"
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] | None = None


@dataclass
class DatasetConfig:
    name: str = "geneval"
    root: str = ""
    train_split: str = "train"
    test_split: str = "test"
    # DPO-specific: path to local pickapic webdataset or HuggingFace dataset name
    dpo_dataset_path: str = ""
    dpo_dataset_name: str = "pickapic_v2_webdataset"  # or "fifa_pickapic_v2"
    dpo_score_dir: str = ""  # Path to pre-computed scores for curriculum learning


@dataclass
class RewardConfig:
    weights: dict[str, float] = field(default_factory=lambda: {"geneval": 1.0})
    only_strict: bool = True
    max_workers: int = 4


@dataclass
class SamplingConfig:
    num_steps: int = 30
    eval_num_steps: int = 30
    guidance_scale: float = 4.5
    eval_guidance_scale: float = 4.5
    resolution: int = 512
    train_batch_size: int = 1
    num_image_per_prompt: int = 4
    num_batches_per_epoch: int = 2
    test_batch_size: int = 2
    same_latent: bool = False
    eta: float = 0.0
    d_fusion: bool = False  # Enable D-Fusion sampling for validation


@dataclass
class TrainingConfig:
    mode: str = "grpo"
    use_8bit_adam: bool = False
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_inner_epochs: int = 1
    max_grad_norm: float = 1.0
    timestep_fraction: float = 1.0
    clip_range: float = 1e-4
    adv_clip_max: float = 5.0
    cfg: bool = False
    per_prompt_stat_tracking: bool = True
    global_std: bool = True
    learning_rate: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8


@dataclass
class GrpoConfig:
    no_cfg: bool = False
    guard: bool = False
    guard_scale: float = 1.0


@dataclass
class DpoConfig:
    """DPO (Direct Preference Optimization) specific configuration."""
    beta_dpo: float = 5000.0  # KL penalty strength
    train_method: str = "diffusion-dpo"  # Options: diffusion-dpo, dspo, dmpo, sdpo, kto
    
    # KTO specific
    kto_lambda_d: float = 1.0
    kto_lambda_u: float = 1.0
    
    # SDPO specific
    sdpo_mu: float = 0.1
    sdpo_alpha: float = 1.0  # max_lambda
    
    # Curriculum specific
    curriculum_groups: int = 0  # 0 means disabled



@dataclass
class PrecisionConfig:
    mixed_precision: str = "bf16"
    allow_tf32: bool = True


@dataclass
class LoggingConfig:
    """Logging and experiment tracking configuration."""
    use_wandb: bool = False
    wandb_project: str = "sd14-grpo-sft"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None  # Auto-generated from run.name if None
    log_grad_norm: bool = True
    log_every_n_steps: int = 1


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    name: str = "cosine"  # "cosine", "linear", "constant"
    warmup_steps: int = 100
    warmup_ratio: float = 0.0  # If > 0, overrides warmup_steps


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    grpo: GrpoConfig = field(default_factory=GrpoConfig)
    dpo: DpoConfig = field(default_factory=DpoConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return cls(
            run=RunConfig(**data.get("run", {})),
            model=ModelConfig(**data.get("model", {})),
            dataset=DatasetConfig(**data.get("dataset", {})),
            reward=RewardConfig(**data.get("reward", {})),
            sampling=SamplingConfig(**data.get("sampling", {})),
            training=TrainingConfig(**data.get("training", {})),
            grpo=GrpoConfig(**data.get("grpo", {})),
            dpo=DpoConfig(**data.get("dpo", {})),
            precision=PrecisionConfig(**data.get("precision", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            scheduler=SchedulerConfig(**data.get("scheduler", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def resolve(self, repo_paths: dict[str, Path]) -> None:
        if not self.run.name:
            self.run.name = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        
        if not self.run.output_dir:
            self.run.output_dir = str(Path(self.run.log_dir) / self.run.name)

        if not self.dataset.root:
            self.dataset.root = str(default_dataset_root(repo_paths, self.dataset.name))

        if self.grpo.no_cfg:
            self.training.cfg = False
            self.sampling.guidance_scale = 1.0
            self.sampling.eval_guidance_scale = 1.0

    def validate(self) -> None:
        if self.dataset.name not in ALLOWED_DATASETS:
            raise ValueError(f"dataset.name must be one of {sorted(ALLOWED_DATASETS)}")
        # Only validate reward weights for GRPO mode
        if self.training.mode == "grpo":
            for reward_name in self.reward.weights:
                if reward_name not in ALLOWED_REWARDS:
                    raise ValueError(
                        f"reward.weights key {reward_name} not in {sorted(ALLOWED_REWARDS)}"
                    )
        if self.training.mode not in ALLOWED_MODES:
            raise ValueError(f"training.mode must be one of {sorted(ALLOWED_MODES)}")
        # DPO-specific validation
        if self.training.mode == "dpo":
            if self.dataset.name != "pickapic":
                raise ValueError("DPO mode requires dataset.name='pickapic'")
            if self.dpo.train_method not in {"diffusion-dpo", "dspo", "dmpo", "sdpo", "kto"}:
                raise ValueError("dpo.train_method must be 'diffusion-dpo', 'dspo', 'dmpo', 'sdpo', or 'kto'")


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if key == "weights" and isinstance(value, dict):
            base[key] = value
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _parse_override(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _set_by_path(base: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cursor = base
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def load_config(path: str | None, overrides: list[str]) -> Config:
    base = Config().to_dict()
    if path:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        _deep_update(base, raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be in key=value form")
        key, value = item.split("=", 1)
        _set_by_path(base, key, _parse_override(value))
    return Config.from_dict(base)

