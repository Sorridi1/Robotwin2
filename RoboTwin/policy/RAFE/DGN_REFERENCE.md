# DGN 参数核对与 RAFE 调参入口

核对来源为作者的[论文原文](https://arxiv.org/html/2506.07505v1)、
[项目主页](https://alessing.github.io/data-guided-noise/)及其链接的官方实现。
代码以下列固定版本为准：
`alessing/dgn@4b96da70a9126ec2691ee179bf4eeb52ea4988b5`。

## 论文与官方实现

论文 Algorithm 1 的顺序为 rollout → off-policy RL 更新 → 定期专家 NLL 更新。
NLL 使用当前策略均值，并只训练噪声分布。
[原文 §4.1–4.2](https://arxiv.org/html/2506.07505v1#S4)。

| 噪声训练参数 | Robomimic | Adroit |
|---|---:|---:|
| 刷新间隔（环境动作步） | 1000 | 2000 |
| 每次刷新完整专家数据 epoch 数 | 2 | 10 |
| minibatch 大小 | 128 | 128 |
| 优化器 | AdamW | AdamW |
| weight decay | 0.03 | 0.03 |
| 论文 Table 2 是否报告噪声学习率 | 未报告 | 未报告 |
| 官方代码噪声学习率 | 3e-4 | 3e-4 |

前五行来自[论文 Appendix A, Table 2](https://arxiv.org/html/2506.07505v1#A1)。
噪声 LR 由[Robomimic 配置](https://github.com/alessing/dgn/blob/4b96da70a9126ec2691ee179bf4eeb52ea4988b5/dgn_robomimic/rl/exploration/learned_cov.py#L12)
和 [Adroit 配置](https://github.com/alessing/dgn/blob/4b96da70a9126ec2691ee179bf4eeb52ea4988b5/dgn_adroit/rlpd/explore_utils.py#L45)
核实；不能把论文 Table 1 的 RL 学习率当成噪声学习率。

官方 Robomimic 实现先计算当前 Actor 在全部专家状态下的残差，再完整遍历
DataLoader。因此，若有 M 个专家样本，一次刷新执行
`2 * ceil(M / 128)` 次 optimizer step，Adroit 对应 `10 * ceil(M / 128)`。
噪声权重和 optimizer 状态在各次刷新之间保留。
[Robomimic 训练循环](https://github.com/alessing/dgn/blob/4b96da70a9126ec2691ee179bf4eeb52ea4988b5/dgn_robomimic/rl/exploration/learned_cov.py#L240)，
[Adroit 训练循环](https://github.com/alessing/dgn/blob/4b96da70a9126ec2691ee179bf4eeb52ea4988b5/dgn_adroit/rlpd/explore_utils.py#L241)。

实现中的时间边界与伪代码略有不同：在线循环开头先检查是否刷新噪声，然后
执行环境动作与 RL 更新，step 0 也会触发噪声刷新。
[Robomimic](https://github.com/alessing/dgn/blob/4b96da70a9126ec2691ee179bf4eeb52ea4988b5/dgn_robomimic/train_rl.py#L400)，
[Adroit](https://github.com/alessing/dgn/blob/4b96da70a9126ec2691ee179bf4eeb52ea4988b5/dgn_adroit/train_cov.py#L175)。
Adroit 官方命令还安排了 50 个噪声预训练 epoch，随后 step 0 再训练 10 个
epoch；这与环境交互 warmup 的含义不同。
[官方命令](https://github.com/alessing/dgn/blob/4b96da70a9126ec2691ee179bf4eeb52ea4988b5/dgn_adroit/README.md)，
[噪声 warmup](https://github.com/alessing/dgn/blob/4b96da70a9126ec2691ee179bf4eeb52ea4988b5/dgn_adroit/train_cov.py#L166)。

RL 更新次数也不是 Actor/Critic 同步重复 20 次：Adroit 官方代码每次更新先做
20 次 Critic step，再做 1 次 Actor step；Robomimic 的发布配置每 2 个环境步
更新一次，其中 Critic 做 5 次、Actor 做 1 次。
[Adroit learner](https://github.com/alessing/dgn/blob/4b96da70a9126ec2691ee179bf4eeb52ea4988b5/dgn_adroit/rlpd/agents/sac/sac_learner.py#L255)，
[Robomimic train](https://github.com/alessing/dgn/blob/4b96da70a9126ec2691ee179bf4eeb52ea4988b5/dgn_robomimic/train_rl.py#L528)。

## 对 RAFE 的参考意义

RAFE 使用 PPO 和 Flow 中间转移，DGN 使用 off-policy RL 和环境动作分布。
以下为迁移建议，不是论文验证的 PPO 配置。

- 保持完整 PPO batch 更新结束后，再冻结最新 Actor 做噪声 NLL；不要在 PPO
  minibatch 之间修改 sigma，否则同一批 PPO 数据的分布会发生变化。
- 当前默认 `update_interval=1, gradient_steps=4, batch_size=16, learning_rate=3e-6`。
  默认每个 rollout 的 PPO 是 2 epochs；
  在 256 条决策样本、batch 4、梯度累积 4 时约为 32 次 PPO optimizer step
  （target KL 可能提前终止），之后做 4 次噪声 optimizer step，即常规预算下
  Actor∶Noise = 32∶4。前 4 轮 Critic warmup 时 Actor 不更新，噪声仍更新。
- 优先比较噪声 LR `3e-5 / 1e-4 / 3e-4`，梯度步数 `1 / 4 / 16`。为保持环境
  预算公平，Actor LR、PPO epochs、rollout 长度、seed 与评估场景固定。
- RAFE 的 `update_interval` 单位是 PPO iteration，DGN 的间隔单位是环境
  动作步。RAFE 的 action chunk 和提前终止会影响实际 `global_step`，两者不能
  直接按数值对应。
- PPO rollout 中有许多 Flow likelihood 项；NLL 对 B、K、H、Da 求平均。
  学习率相同不代表与 DGN 具有相同的梯度强度。记录实际 optimizer 步数、
  梯度范数、sigma 变化以及上下界比例，再决定增加监督预算。
- DGN 的完整数据 epoch、dropout、weight decay 和噪声退火不自动加入 RAFE。
  尤其预训练 Actor 已经较好时，长时间噪声预训练可能迅速把 sigma 推向下界。

新增可选配置 `reinflow_rl_pointcloud_robotwin2_expert_interp_dgn_tuning.yaml`：
噪声 LR `1e-4`、每轮 4 个 minibatch 更新，batch 仍为 16；其余沿用默认。
这只是一个尚未经真实 RL 训练验证的初始调参点，训练命令替换配置名即可使用。

## 本次噪声参数化修改

```python
raw_delta = noise_head_output.float()  # default residual_scale=1.0
log_sigma = torch.log(base_sigma.float()) + raw_delta
log_sigma = log_sigma.clamp(math.log(sigma_min), math.log(sigma_max))
sigma = torch.exp(log_sigma)
```

实际代码同时保留原 sampling floor，rollout、PPO 和专家 NLL 共用计算。
`residual_scale` 现在仅为线性增益，默认 1.0；固定噪声对照设置为 0。
非零增益不限制可表达范围。默认 sigma 仍限制在 `[0.003, 0.06]`，没有自动
放宽这组上、下界。旧版后期 base `0.0072` 受 `0.5*tanh` 限制，最大约
`0.01187`；现在所有 step 都可达到 `0.06`。移除 tanh 不保证实际学到的噪声
增大，NLL 仍可能推动 sigma 收缩；硬 clamp 外的梯度也仍为零。

sigma 的 log/exp 在 float32 中计算，先 clamp 再 exp；诊断更新为真实全局
边界，不再报告 tanh 饱和或旧 `base*exp(±residual_scale)` 区间。
RL resume 格式升级为版本 2，含参数化标记，明确拒绝旧 bounded-noise 状态。
原始 ManiFlow 初始化 checkpoint 和只含 Flow Actor 的评估 checkpoint 仍可使用。
