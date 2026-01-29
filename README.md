# rl_sft

SD1.4/SD1.5 训练框架，支持 **GRPO**、**DPO** 和 **SFT** 三种训练模式。

## 训练模式

| 模式 | 配置文件 | 数据集 | 说明 |
|------|---------|--------|------|
| **GRPO** | `grpo_geneval.json` | geneval | 在线采样 + geneval2 reward |
| **DPO** | `dpo_pickapic.json` | pickapic | 离线偏好对数据 |
| **SFT** | `sft_spright.json` | spright | GT 图像监督学习 |

## 快速开始

```bash
# GRPO + Geneval2 reward
python run.py --config configs/grpo_geneval.json --gpus 2,3

# DPO + PickaPic 偏好数据
python run.py --config configs/dpo_pickapic.json

# SFT + Spright GT 图像
python run.py --config configs/sft_spright.json

# 预览配置（不训练）
python run.py --config configs/grpo_geneval.json --dry-run
```

## 必须配置的路径

| 参数 | 说明 |
|------|------|
| `model.name_or_path` | SD1.4/SD1.5 预训练模型路径 |
| `dataset.root` | 数据集目录（geneval/spright） |
| `dataset.dpo_dataset_path` | pickapic WebDataset tar 文件目录（DPO 模式） |

## 模式参数

### DPO

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dpo.beta_dpo` | 5000.0 | KL 惩罚强度 |
| `dpo.train_method` | `diffusion-dpo` | 损失类型：`diffusion-dpo`/`dspo`/`dmpo`/`sdpo`/`kto` |

#### KTO/SDPO Specifics
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dpo.kto_lambda_d` | 1.0 | KTO desirable weight |
| `dpo.kto_lambda_u` | 1.0 | KTO undesirable weight |
| `dpo.sdpo_mu` | 0.1 | SDPO mu |
| `dpo.sdpo_alpha` | 1.0 | SDPO alpha (strength) |

#### Curriculum Learning
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dpo.curriculum_groups` | 0 | 课程分组数 (0禁用)。若 >0，需在 `data.py` 配置分数逻辑。 |


### GRPO

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `reward.weights.geneval` | 1.0 | geneval reward 权重 |
| `sampling.num_image_per_prompt` | 4 | 每个 prompt 采样数量 |

### SFT

使用 GT 图像直接监督训练，MSE loss。

**Aspect Ratio Bucketing**: SFT 模式自动使用长宽比分桶，保留原始图像比例：
- 支持 13 种标准分辨率 (512x512, 576x448, 640x384, ...)
- 每个 batch 内图像自动缩放到相同 bucket 尺寸
- 最大化保留图像内容，减少裁剪损失

## 数据集准备

### PickaPic (DPO)

WebDataset tar 结构：
```
├── 0.jpg / jpg_0.jpg    # 图像1
├── 1.jpg / jpg_1.jpg    # 图像2
└── sample.json          # {"caption": "...", "label_0": 0|1}
```
`label_0=1` 表示 jpg_0 获胜。

### Geneval (GRPO)

需要 geneval2 服务运行在 `http://127.0.0.1:18085`。

### Spright (SFT)

WebDataset tar，包含 `json`（含 `spatial_caption`）和 `jpg`。

## CLI 覆盖

```bash
python run.py --config configs/dpo_pickapic.json \
  --set dpo.beta_dpo=2500 \
  --set logging.use_wandb=true
```

## 输出

每个训练任务通过 `run.output_dir` 指定独立输出目录，避免权重混淆：

```json
{
  "run": {
    "output_dir": "logs/grpo_geneval_beta5000"
  }
}
```

输出结构：
```
logs/<output_dir>/
├── config.json
├── metrics.jsonl
└── checkpoints/{epoch_N, best, final}/
```

## 依赖

- `torch`, `diffusers`, `accelerate`, `transformers`, `webdataset`
- Optional: `wandb`, `bitsandbytes`


