# 训练指标说明文档 (Training Metrics Documentation)

本文档旨在说明在 GRPO、DPO 和 SFT 训练过程中记录在 WandB 上的各个指标的含义，帮助观察和分析训练过程。

## 通用指标 (Common Metrics)

*   **epoch**: 当前训练的轮数。
*   **inner_epoch**: 在当前 batch 上的内部迭代次数。
*   **global_step**: 总训练步数。
*   **lr**: 当前的学习率。
*   **loss**: 总损失函数值。
*   **grad_norm**: 梯度的范数，用于观察训练是否稳定，防止梯度爆炸或消失。

---

## GRPO 相关指标 (GRPO Metrics)

GRPO (Group Relative Policy Optimization) 是一种强化学习算法，主要通过对比同一 prompt 下的一组样本的相对得分来更新模型。

*   **avg_reward**: 当前 batch 样本的平均奖励得分。
*   **reward/{name}**: 具体的奖励分项得分（例如 `reward/geneval`, `reward/ocr` 等）。
*   **advantage/mean**: 优势值（Advantage）的平均值。GRPO 中通常是奖励减去其均值并除以标准差。
*   **advantage/std**: 优势值的标准差。
*   **advantage/max / advantage/min**: 优势值的最大值和最小值。
*   **ratio**: 采样概率比值 $\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}$ 的平均值。
*   **ratio_max / ratio_min**: 采样概率比值的最大值和最小值。
*   **clip_fraction**: 损失函数中被 clip（裁剪）的比率。如果该值过高，说明学习步长可能过大。
*   **approx_kl**: 当前策略与采样策略之间的近似 KL 散度，计算公式为 $ratio - 1 - \log(ratio)$。用于观察模型偏离原始策略的程度。
*   **guard_loss**: (如果启用了 GRPO-Guard) 防止模型过快偏离原模型的辅助损失。

---

## DPO 相关指标 (DPO Metrics)

DPO (Direct Preference Optimization) 及其变体 (KTO, SDPO 等) 通过偏好对（胜者/败者）直接优化模型。

*   **model_mse**: 当前模型在所有样本上的平均 MSE 损失。
*   **ref_mse**: 参考模型（冻结的原始模型）在所有样本上的平均 MSE 损失。
*   **implicit_acc**: 隐式准确率。表示模型预测胜者的概率大于败者的概率的比例。该值越接近 1 说明模型越能符合偏好数据。
*   **margin/model**: 模型在胜者和败者之间的得分差值（隐式奖励差）。
*   **margin/ref**: 参考模型在胜者和败者之间的得分差值。
*   **margin/relative**: 模型边际与参考模型边际的差值，即相对于原模型，当前模型对偏好的优化程度。
*   **loss_component/model_w / model_l**: 模型在胜者（w）和败者（l）上的 MSE 损失，分别观察模型对两类数据的拟合情况。
*   **loss_component/ref_w / ref_l**: 参考模型在胜者和败者上的 MSE 损失。

---

## SFT 相关指标 (SFT Metrics)

SFT (Supervised Fine-Tuning) 是标准的监督训练过程。

*   **loss**: 预测噪声与真实噪声之间的 MSE 损失。
*   **eval/***: 验证集上的各项指标。
