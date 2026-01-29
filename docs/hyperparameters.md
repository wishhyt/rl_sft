# SD14 GRPO/SFT 超参数配置指南

本文档详细说明所有可配置的超参数及其对显存 (VRAM) 的影响。

## 目录
- [快速配置示例](#快速配置示例)
- [显存影响速查表](#显存影响速查表)
- [详细参数说明](#详细参数说明)

---

## 快速配置示例

### 24GB 显卡 (RTX 3090/4090)
```bash
python run.py --config configs/grpo_geneval.json \
  --set sampling.train_batch_size=1 \
  --set sampling.num_image_per_prompt=4 \
  --set sampling.num_steps=20 \
  --set model.use_lora=true
```

### 48GB 显卡 (A6000/L40)
```bash
python run.py --config configs/grpo_geneval.json \
  --set sampling.train_batch_size=2 \
  --set sampling.num_image_per_prompt=8 \
  --set sampling.num_steps=30
```

### 80GB 显卡 (A100/H100)
```bash
python run.py --config configs/grpo_geneval.json \
  --set sampling.train_batch_size=4 \
  --set sampling.num_image_per_prompt=16 \
  --set sampling.num_steps=50
```

---

## 显存影响速查表

| 参数 | 默认值 | 显存影响 | 说明 |
|------|--------|----------|------|
| `sampling.train_batch_size` | 1 | 🔴 **极高** | 每增加1，约增加 2-4GB |
| `sampling.num_image_per_prompt` | 4 | 🔴 **极高** | GRPO 分组大小，影响采样显存 |
| `sampling.resolution` | 512 | 🔴 **极高** | 768 比 512 增加约 2.25x 显存 |
| `sampling.num_steps` | 30 | 🟠 **中等** | 影响采样时间，对峰值显存影响较小 |
| `model.use_lora` | true | 🟢 **降低** | LoRA 比全量微调省约 60% 显存 |
| `training.gradient_accumulation_steps` | 1 | 🟢 **降低** | 增大可用更小 batch_size |
| `precision.mixed_precision` | "bf16" | 🟢 **降低** | bf16/fp16 比 fp32 省约 50% |
| `training.cfg` | true | 🟠 **中等** | 禁用 CFG (no_cfg) 可省约 30% |

### 显存估算公式

```
基础显存 ≈ 8GB (SD1.4 + VAE + Text Encoder)

训练显存 ≈ 基础显存 
         + batch_size × num_image_per_prompt × 0.8GB  (采样)
         + batch_size × 1.5GB  (梯度+优化器)
         × (2 if cfg else 1)   (CFG 倍增)
         × (0.4 if use_lora else 1)  (LoRA 节省)
```

---

## 详细参数说明

### 运行配置 (`run`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | auto | 运行名称，自动生成时间戳 |
| `seed` | int | 42 | 随机种子 |
| `log_dir` | str | "logs" | 日志目录 |
| `save_freq` | int | 20 | 每 N epoch 保存 checkpoint |
| `eval_freq` | int | 20 | 每 N epoch 评估 |
| `num_epochs` | int | 100000 | 总训练 epoch 数 |
| `resume_from` | str | null | 恢复训练的 checkpoint 路径 |

### 模型配置 (`model`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name_or_path` | str | "CompVis/stable-diffusion-v1-4" | 模型路径 |
| `revision` | str | "main" | 模型版本 |
| `use_lora` | bool | true | 是否使用 LoRA (推荐开启) |
| `lora_rank` | int | 16 | LoRA 秩 (4-64，越大效果越好但更耗显存) |
| `lora_alpha` | int | 32 | LoRA alpha (通常设为 2×rank) |

### 采样配置 (`sampling`) ⚠️ 显存关键

| 参数 | 类型 | 默认值 | 显存影响 | 说明 |
|------|------|--------|----------|------|
| `train_batch_size` | int | 1 | 🔴 | 训练 batch 大小 |
| `num_image_per_prompt` | int | 4 | 🔴 | GRPO 每个 prompt 生成图像数 |
| `num_batches_per_epoch` | int | 2 | - | 每 epoch 的 batch 数 |
| `resolution` | int | 512 | 🔴 | 图像分辨率 |
| `num_steps` | int | 30 | 🟠 | 采样步数 |
| `guidance_scale` | float | 4.5 | - | CFG 引导强度 |
| `same_latent` | bool | false | - | 相同 prompt 使用相同噪声 |

### 训练配置 (`training`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | str | "grpo" | 训练模式: "grpo" 或 "sft" |
| `learning_rate` | float | 1e-5 | 学习率 |
| `gradient_accumulation_steps` | int | 1 | 梯度累积步数 (增大可减少显存) |
| `num_inner_epochs` | int | 1 | 每批数据的内循环次数 |
| `max_grad_norm` | float | 1.0 | 梯度裁剪阈值 |
| `cfg` | bool | true | 是否使用 CFG |
| `clip_range` | float | 1e-4 | PPO/GRPO 裁剪范围 |
| `adv_clip_max` | float | 5.0 | 优势值裁剪上限 |

### GRPO 配置 (`grpo`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `no_cfg` | bool | false | 禁用 CFG (可省约 30% 显存) |
| `guard` | bool | false | 启用 GRPO-Guard 稳定训练 |
| `guard_scale` | float | 1.0 | Guard loss 权重 |

### 学习率调度器 (`scheduler`) 🆕

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | "cosine" | 调度器类型: "cosine", "linear", "constant" |
| `warmup_steps` | int | 100 | warmup 步数 |
| `warmup_ratio` | float | 0.0 | warmup 比例 (>0 时覆盖 warmup_steps) |

### 日志配置 (`logging`) 🆕

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_wandb` | bool | false | 启用 WandB 日志 |
| `wandb_project` | str | "sd14-grpo-sft" | WandB 项目名 |
| `wandb_entity` | str | null | WandB 团队/用户名 |
| `log_grad_norm` | bool | true | 记录梯度范数 |
| `log_every_n_steps` | int | 1 | 每 N 步记录一次 |

### 精度配置 (`precision`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mixed_precision` | str | "bf16" | 混合精度: "bf16", "fp16", "no" |
| `allow_tf32` | bool | true | 允许 TF32 (Ampere+ GPU) |

---

## 配置文件示例

### 低显存配置 (16-24GB)
```json
{
  "model": {"use_lora": true, "lora_rank": 8},
  "sampling": {
    "train_batch_size": 1,
    "num_image_per_prompt": 2,
    "resolution": 512,
    "num_steps": 20
  },
  "training": {"gradient_accumulation_steps": 4},
  "grpo": {"no_cfg": true}
}
```

### 高显存配置 (80GB+)
```json
{
  "model": {"use_lora": false},
  "sampling": {
    "train_batch_size": 4,
    "num_image_per_prompt": 16,
    "resolution": 512,
    "num_steps": 50
  },
  "logging": {"use_wandb": true}
}
```

---

## 命令行覆盖语法

使用 `--set` 覆盖任意配置项：

```bash
# 单个参数
python run.py --config configs/grpo_geneval.json --set sampling.train_batch_size=2

# 多个参数
python run.py --config configs/grpo_geneval.json \
  --set sampling.train_batch_size=2 \
  --set training.learning_rate=5e-6 \
  --set logging.use_wandb=true

# 快捷参数
python run.py --config configs/grpo_geneval.json --no-cfg --grpo-guard

# 预览配置 (不运行训练)
python run.py --config configs/grpo_geneval.json --dry-run
```
