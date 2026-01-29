# Multi-Method DPO Quick Start Guide

## 快速开始

### 1. 创建配置文件

编辑 `configs/my_dpo_comparison.json`:

```json
{
  "base_config": "configs/dpo_pickapic.json",
  "experiments": [
    {
      "name": "standard_dpo",
      "gpus": "0",
      "dpo_method": "diffusion-dpo",
      "beta_dpo": 5000.0,
      "output_dir": "logs/my_comparison/standard",
      "eval_output_dir": "logs/my_comparison/standard/eval"
    },
    {
      "name": "sdpo",
      "gpus": "1",
      "dpo_method": "sdpo",
      "beta_dpo": 5000.0,
      "sdpo_mu": 0.1,
      "output_dir": "logs/my_comparison/sdpo",
      "eval_output_dir": "logs/my_comparison/sdpo/eval"
    },
    {
      "name": "kto",
      "gpus": "2",
      "dpo_method": "kto",
      "beta_dpo": 5000.0,
      "kto_lambda_d": 1.0,
      "kto_lambda_u": 1.0,
      "output_dir": "logs/my_comparison/kto",
      "eval_output_dir": "logs/my_comparison/kto/eval"
    }
  ]
}
```

### 2. 运行实验

```bash
# 预览配置
python run_multi_dpo.py --config configs/my_dpo_comparison.json --dry-run

# 并行运行所有方法
python run_multi_dpo.py --config configs/my_dpo_comparison.json

# 只运行前两个方法
python run_multi_dpo.py --config configs/my_dpo_comparison.json --methods standard_dpo sdpo

# 顺序运行（单GPU或GPU资源有限）
python run_multi_dpo.py --config configs/my_dpo_comparison.json --sequential
```

### 3. 对比结果

```bash
# 生成对比报告
python compare_dpo_results.py \
  --experiments logs/my_comparison/standard logs/my_comparison/sdpo logs/my_comparison/kto \
  --output comparison_report.md \
  --show-plots
```

## 配置字段说明

### 实验配置字段

- `name`: 实验名称（用于标识和日志）
- `gpus`: GPU ID，用逗号分隔（如 "0" 或 "0,1"）
- `dpo_method`: DPO 方法类型
  - `"diffusion-dpo"`: 标准 DPO
  - `"sdpo"`: Winner-Preserving SDPO
  - `"dspo"`: DSPO
  - `"dmpo"`: DMPO
  - `"kto"`: KTO
- `beta_dpo`: KL 惩罚强度（默认 5000.0）
- `output_dir`: checkpoint 保存目录
- `eval_output_dir`: 评估结果保存目录

### 方法特定参数

**SDPO**:
- `sdpo_mu`: Winner-preserving μ 参数（默认 0.1）
- `sdpo_alpha`: 最大 λ 值（默认 1.0）

**KTO**:
- `kto_lambda_d`: Desirable 样本权重（默认 1.0）
- `kto_lambda_u`: Undesirable 样本权重（默认 1.0）

## 常见场景

### 场景1：对比两个方法

```json
{
  "base_config": "configs/dpo_pickapic.json",
  "experiments": [
    {"name": "method_a", "gpus": "0", "dpo_method": "diffusion-dpo", ...},
    {"name": "method_b", "gpus": "1", "dpo_method": "sdpo", ...}
  ]
}
```

### 场景2：同一方法不同超参数

```json
{
  "base_config": "configs/dpo_pickapic.json",
  "experiments": [
    {"name": "beta_1000", "gpus": "0", "dpo_method": "diffusion-dpo", "beta_dpo": 1000.0, ...},
    {"name": "beta_5000", "gpus": "1", "dpo_method": "diffusion-dpo", "beta_dpo": 5000.0, ...},
    {"name": "beta_10000", "gpus": "2", "dpo_method": "diffusion-dpo", "beta_dpo": 10000.0, ...}
  ]
}
```

### 场景3：单GPU顺序运行

```json
{
  "base_config": "configs/dpo_pickapic.json",
  "experiments": [
    {"name": "exp1", "gpus": "0", ...},
    {"name": "exp2", "gpus": "0", ...},
    {"name": "exp3", "gpus": "0", ...}
  ]
}
```

运行: `python run_multi_dpo.py --config ... --sequential`

## Tips

1. **使用 --dry-run 先预览**: 确保配置正确后再真正运行
2. **合理分配 GPU**: 避免多个实验使用同一个 GPU（除非顺序运行）
3. **独立输出目录**: 确保每个实验有不同的 `output_dir`
4. **全局参数覆盖**: 使用 `--set` 快速修改所有实验的共同参数
5. **监控日志**: 查看 `.multi_dpo_temp/launch_*.log` 了解每个实验的详细输出
