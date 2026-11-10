# RAFE: Expert-Interpolation-Guided Noise Learning

RAFE 是独立的 RoboTwin policy：评估时使用 `policy_name=RAFE`。从当前 `RL测试`
分支的 ManiFlow 复制所需源码、配置、脚本和评估 cohort；原 `policy/ManiFlow`
目录保持不变。内部保留 `ManiFlow/maniflow` 包名，以复用 checkpoint 和 Hydra
target；启动脚本优先加载 RAFE 自己的源码，无需重新安装或替换原 ManiFlow 包。
第三方库继续使用现有 `maniflow` conda 环境，不复制第三方源码和资产。

RAFE 已移除观测预测动作参考 Aref 的 MLP、coarse-action target、动作参考损失、
DiT-X 参考投影和注入，以及噪声头的 plan/ReferenceFlow 分支。源码中不再保留
`FutureActionPlanPredictor`，配置中也不再提供 `policy.action_plan`。旧版无参考
RAFE checkpoint 中的 `enabled:false` 元数据只在加载时丢弃；带参考网络的
权重或启用参考的配置会明确报错。原 ManiFlow 仍保留其自身实现。

噪声现使用无 `tanh` 的 log residual：先对 `log(base_sigma)+raw_delta` 限幅，
再取指数。DGN 原文、官方实现参数及调参入口见 [DGN_REFERENCE.md](DGN_REFERENCE.md)。

## 算法与代码位置

所有下列路径均相对于 `RoboTwin/policy/RAFE`。

| 文件 | 改动与原因 |
|---|---|
| `ManiFlow/maniflow/policy/maniflow_pointcloud_policy.py` | 移除 Aref MLP 的构造、预测、监督损失、optimizer 参数组和参考推理接口 |
| `ManiFlow/maniflow/model/diffusion/ditx.py` | 移除动作参考投影、参数和注入，只保留原 Flow 条件输入 |
| `ManiFlow/maniflow/model/action_plan.py` | 删除该文件，不再保留参考预测网络定义 |
| `ManiFlow/maniflow/model/rl/ppo_maniflow_pointcloud.py` | `compute_expert_interpolation_state` 构造完整动作块的插值；`diagonal_gaussian_nll` 计算逐元素 NLL；`compute_expert_interpolation_noise_loss` 实现专家监督；`gaussian_transition` 统一三条路径的均值、sigma、时间参数和裁剪 |
| `ManiFlow/maniflow/common/expert_data.py` | 复用 `RoboTwinDataset`，独立可循环 DataLoader，保存数据位置、排列及增强 RNG |
| `ManiFlow/maniflow/common/ppo_optimizer.py` | 增加 actor-only 参数组选择，保留旧联合 optimizer 的 Adam 状态迁移 |
| `ManiFlow/maniflow/workspace/train_reinflow_rl_robotwin2_workspace.py` | 独立 noise optimizer、PPO→NLL 交替、严格初始化/恢复、诊断与本地 JSONL、沿用 Actor 评估和 best 选择 |
| `ManiFlow/maniflow/workspace/train_maniflow_robotwin2_workspace.py` | 本地包路径优先，防止已安装的原 ManiFlow 遮蔽 RAFE 评估源码 |
| `ManiFlow/maniflow/config/reinflow_rl_pointcloud_robotwin2_expert_interp.yaml` | 新方法独立 Hydra 配置；基础 RL/监督配置也已删除全部动作参考设置 |
| `ManiFlow/maniflow/config/reinflow_rl_pointcloud_robotwin2_{ppo_sigma,fixed_sigma}.yaml` | 无 Reference 的公平对照配置 |
| `ManiFlow/maniflow/config/reinflow_rl_pointcloud_robotwin2_expert_interp_dgn_tuning.yaml` | 可选噪声 LR `1e-4`、每轮 4 个 minibatch 的调参起点 |
| `__init__.py`, `deploy_policy.py`, `deploy_ppo_actor_policy.py`, `deploy_policy.yml` | 注册 RAFE，延迟 simulator 导入、优先本地源码、定位本方法输出；PPO 部署严格加载 |
| `scripts/train_rl_policy.sh`, `scripts/train_policy.sh`, `train_eval_robotwin2.sh` | RAFE 输出/评估路径，保留原训练评估调用方式 |
| `scripts/check_expert_interpolation.py` | 真实预训练权重 + 真实专家数据的 CPU/GPU 更新、采样及可选恢复检查 |
| `tests/` | 数学、梯度隔离、概率一致性、真实 Zarr loader 与小型 PPO/恢复闭环测试 |
| `ManiFlow/setup.py` | 分发名称 `rafe-maniflow`；运行不依赖重新安装 |

其余 `ManiFlow/` 源码、原配置、数据处理和场景验证脚本直接复制复用。
仓库根 `.gitignore` 仅增加 RAFE 测试/cohort 的追踪例外，并忽略训练输出。

1. `A = checkpoint_normalizer['action'].normalize(batch['action'])`，保留原始完整
   `[B,H,Da]` 动作块；同一次监督的所有 step 共用一份 `X0 ~ N(0,I)`。
2. `Xt=(1-t)*X0+t*A`，下一状态标签使用 `t+1/K` 的同一条插值路径。
3. `gaussian_transition` 调用当前在线 Actor 的 `_velocity`，`target_t` 沿用
   relative/absolute 约定；均值为 `clip(Xt+v/K, denoised_clip_value)`。
4. `_noise_std → compute_residual_sigma → noise_head.predict_delta` 使用同一个
   TCN，`log_sigma=log(base_sigma)+residual_scale*g`，在 log 空间按 sigma 边界
   限幅后再取指数；同时保持现有 sampling floor。没有 residual-output tanh，
   `residual_scale` 仅为线性增益，默认 1.0；噪声输入只有观测、时间、当前 Xt
   和 horizon 位置。
5. NLL 为 `0.5*((target-mean.detach())/sigma)**2 + log(sigma) + 0.5*log(2π)`，
   对 batch、被监督的 step、H、Da 求平均；关键似然运算使用 float32。
   仅监督 `range(K-ft_denoising_steps,K)`，不训练固定噪声 step。

PPO 的 `loss(..., learn_exploration_noise=False)` 只 detach sigma 分支，保留
均值梯度；其 optimizer 不含 noise 参数。NLL 的编码和当前均值使用 `no_grad`，
条件特征 detach，独立 optimizer 仅含 TCN。运行时检测梯度泄漏、参数版本变化、
非有限 loss/梯度、长期无更新、下界饱和及不可表达的残差比例。

## 配置与数据

`rl.noise_guidance` 默认值：

| 配置项 | 默认值 |
|---|---|
| `enabled`, `mode` | `true`, `expert_interpolation` |
| `dataset_path` | `null`，回退到原 `task.dataset.zarr_path`；路径不存在明确报错 |
| `batch_size` | `16` |
| `update_interval`, `gradient_steps` | `1`, `4` |
| `learning_rate`, `weight_decay`, `max_grad_norm` | `3e-6`, `0.0`, `1.0` |
| `initial_warmup_steps` | `0`（可选首次 rollout 前的 NLL 更新次数） |
| `sample_all_steps`, `log_diagnostics` | `true`, `true` |
| `num_workers` | `0`，为精确恢复数据及增强 RNG，其他值明确报错 |

配置不包含 `policy.action_plan`；`noise_head_type=residual_schedule`，
`noise_backbone=tcn`，`noise_conditioning_mode=default`，`noise_output_mode=full`，
`noise_parameterization=log_residual`，`residual_scale=1.0`，
`base_sigma_schedule=linear`，`freeze_obs_encoder=true`。P 网络和
DiT-X 参考注入的代码已删除。其余 PPO、Critic、reward、BC anchor、AMP、梯度累积、
rollout/reset 和评估参数继承现有配置；本实验要求 `ent_coef=0`。

数据沿用 ManiFlow Zarr：`data/point_cloud`、`data/state`、`data/action`、
`meta/episode_ends`。`RoboTwinDataset/SequenceSampler` 保留原 horizon、padding、
颜色增强和 train/val 划分，编码器截取前 `n_obs_steps` 个观测。不重新拟合
normalizer，不使用 lowpass 或执行窗口截断的动作，不收集在线专家标签。

当前 place_empty_cup 数据实际为动作 `[B,16,14]`、点云 `[B,16,1024,6]`、
状态 `[B,16,14]`，观测编码使用前两帧。原执行窗口从 `n_obs_steps-1` 开始，
在 `H=16,n_obs_steps=2,n_action_steps=16` 下实际返回 15 步；保留原行为，
NLL 仍使用完整 16 步。

resume 保存 Actor/Noise/Critic、三个 optimizer、AMP scaler、迭代/环境步数、
guidance 配置、噪声 RNG、数据 epoch/batch/增强 RNG 和全局 RNG；拒绝缺少
独立噪声状态或改变 Gaussian/策略语义的恢复。RL resume 格式为版本 2，并保存
噪声参数化标记；旧 bounded-noise RL checkpoint 明确拒绝加载，应使用原始
ManiFlow Actor 初始化新实验。只含 Flow Actor 的旧导出评估不受影响。
BC reference 沿用原逻辑，从
不可变的预训练 checkpoint 重建。物理环境内部状态未序列化；跨进程恢复时
沿用原环境重新创建行为，不承诺恢复相同物理轨迹。

## 训练、对照与评估

以下命令从仓库根执行，使用当前机器已存在的 checkpoint、数据和 Python。
完整 RL 训练需要可用的 CUDA 驱动和 RoboTwin/Curobo 环境：

```bash
cd /home/ljj/code/Robotwin2/RoboTwin/policy/RAFE
PYTHON_BIN=/opt/shared/Prog/anaconda3/envs/maniflow/bin/python \
EVAL_SCENE_SELECTION=rl \
bash scripts/train_rl_policy.sh \
  reinflow_rl_pointcloud_robotwin2_expert_interp \
  place_empty_cup demo_messy_ur5 50 expert-interp 0 0 \
  /media/Elements1/ljj/ManiFlow/outputs/place_empty_cup-maniflow_pointcloud_policy_robotwin2-0428_seed0/checkpoints/latest.ckpt \
  demo_messy_ur5 \
  rl.noise_guidance.dataset_path=/media/Elements1/ljj/ManiFlow/data/dataset/place_empty_cup-demo_messy_ur5-50.zarr \
  logging.mode=disabled
```

默认输出 `RAFE/outputs/<task>-<config>-<addition>_seed<seed>/`。可设置
`RAFE_OUTPUT_ROOT` 或 `RAFE_RUN_DIR`，定期评估会传递真实 run directory。
默认启用定期评估；`EVAL=true` 额外执行训练后的评估，`TRAIN=false EVAL=true`
可独立评估已有 run。`RL_EVAL_POLICY_NAME=RAFE.deploy_ppo_actor_policy` 可显式
评估带 Noise Head 的 resume checkpoint，此时相应设置 `EVAL_CKPT_TAG=latest_rl`。

训练场景默认 `rl.env.expert_check=false, rl.env.scene_validation.mode=cheap`，
仅检查合法性，不执行专家轨迹。RAFE 的 RL 脚本、训练后评估脚本和部署 YAML
均默认选择 `rl` 场景；也可显式设置 `EVAL_SCENE_SELECTION=rl`。
评估会检查初始状态合法性，再运行待评估策略，不以专家能否成功筛选场景。
共享 seed bank 中 `expert`、`rl`、`expert_fail` 和 `baseline_fail` 为不同组。

先单独评估 0428 预训练基线（100 个 RL 合法场景）：

```bash
cd /home/ljj/code/Robotwin2/RoboTwin
CUDA_VISIBLE_DEVICES=0 \
VK_ICD_FILENAMES="$PWD/policy/ManiFlow/scripts/nvidia_icd.json" \
/opt/shared/Prog/anaconda3/envs/maniflow/bin/python script/eval_policy.py \
  --config policy/ManiFlow/deploy_policy.yml --overrides \
  --config_name maniflow_pointcloud_policy_robotwin2 \
  --task_name place_empty_cup --task_config demo_messy_ur5 \
  --ckpt_setting demo_messy_ur5 --expert_data_num 50 \
  --training_seed 0 --seed 0 --policy_name ManiFlow \
  --addition_info 0428 --alg_name maniflow_pointcloud_policy_robotwin2 \
  --ckpt_tag 'epoch=0500-val_loss=0.373935' \
  --eval_scene_selection rl --eval_num_episodes 100 \
  --eval_scene_seed_file /media/Elements1/ljj/ManiFlow/outputs/eval_scene_seeds.json
```

这条命令对应已存在的
`/media/Elements1/ljj/ManiFlow/outputs/place_empty_cup-maniflow_pointcloud_policy_robotwin2-0428_seed0/checkpoints/epoch=0500-val_loss=0.373935.ckpt`。
RAFE 训练后评估沿用上面训练示例的 addition `expert-interp`：

```bash
cd /home/ljj/code/Robotwin2/RoboTwin/policy/RAFE
TRAIN=false EVAL=true EVAL_SCENE_SELECTION=rl EVAL_CKPT_TAG=best \
PYTHON_BIN=/opt/shared/Prog/anaconda3/envs/maniflow/bin/python \
bash scripts/train_rl_policy.sh \
  reinflow_rl_pointcloud_robotwin2_expert_interp \
  place_empty_cup demo_messy_ur5 50 expert-interp 0 0 \
  /media/Elements1/ljj/ManiFlow/outputs/place_empty_cup-maniflow_pointcloud_policy_robotwin2-0428_seed0/checkpoints/epoch=0500-val_loss=0.373935.ckpt \
  demo_messy_ur5
```

RAFE 实验名称必须与训练一致；如果已有 run 存在其他目录，设置
`RAFE_RUN_DIR=/absolute/path/to/run`。`best` 是定期评估选出的导出 Actor，
`latest` 是最近保存的导出 Actor；仅有 `latest_rl.ckpt` 时应先使用对应的
PPO 部署入口。上述 0428 和 RAFE 命令使用同一份 RL 场景 bank，便于比较。

| 配置后缀 | 噪声更新 | Flow 更新 |
|---|---|---|
| `fixed_sigma` | 固定 linear schedule，`residual_scale=0,noise_lr=0` | PPO |
| `ppo_sigma` | 原 TCN 经 PPO 更新 | PPO |
| `expert_interp` | 同 TCN 经独立专家 Gaussian NLL 更新 | PPO |

替换命令的配置名即可运行对照。固定噪声保留同样 TCN 参数量，但零 residual
scale 使输出等于固定 schedule，参数无噪声梯度和更新。对照使用相同 checkpoint、
任务、seed、推理步数、Actor LR、rollout/PPO 预算和评估场景 bank。

`metrics.jsonl` 总是写入，W&B 按原配置可选：

- `global_step` 是实际执行的环境动作步数，用于画学习曲线和计算达到阈值的步数。
- `rollout/online_success_rate` 是含探索噪声的训练交互表现。
- `eval/exported_actor_success_rate` 是只导出微调 Flow Actor、沿用原
  `script/eval_policy.py → RAFE.deploy_policy → RobotRunner` 路径的成功率；
  `best.ckpt` 仍根据该评估选择。原 `eval/actor_success_rate` 等别名保留。
- `noise_nll_loss`、`noise_sigma_mean/min/max`、上下界比例、残差 MSE/绝对值、
  梯度范数、更新次数/耗时、逐 step sigma/MSE、可表达区间和饱和比例用于诊断。

## 小规模验证

最小自动化闭环（不需要 GPU 或物理仿真）：

```bash
cd /home/ljj/code/Robotwin2
/opt/shared/Prog/anaconda3/envs/maniflow/bin/python -m pytest RoboTwin/policy/RAFE/tests -q
```

真实预训练模型与专家数据，监督全部 10 个 step，更新、采样并检查完整恢复：

```bash
cd /home/ljj/code/Robotwin2
/opt/shared/Prog/anaconda3/envs/maniflow/bin/python \
  RoboTwin/policy/RAFE/scripts/check_expert_interpolation.py \
  --checkpoint /media/Elements1/ljj/ManiFlow/outputs/place_empty_cup-maniflow_pointcloud_policy_robotwin2-0428_seed0/checkpoints/latest.ckpt \
  --dataset /media/Elements1/ljj/ManiFlow/data/dataset/place_empty_cup-demo_messy_ur5-50.zarr \
  --all-steps --check-resume
```

真实环境 smoke：

```bash
/opt/shared/Prog/anaconda3/envs/maniflow/bin/python \
  RoboTwin/policy/RAFE/scripts/check_rl_simulation.py --backend serial --n-envs 1 --steps 1
```

已运行 A–E 自动化测试（真实 adapter/TCN/Zarr/optimizer/checkpoint，小型
Actor/Critic 和 toy rollout）、无动作参考网络结构检查、原始 ManiFlow
checkpoint 的严格加载与预测一致性、optimizer 迁移/恢复和评估配置回归。
原 ReferenceFlow 功能测试已从 RAFE 副本删除，因为相应功能已移除；原
ManiFlow 目录不受影响。

无 tanh 参数化、旧状态拒绝与原功能组合测试 **115 passed**。真实 181M 无参考 checkpoint 和专家 Zarr
再次通过全部 10 步 NLL 更新、采样与保存/恢复，确认
`no_action_reference_modules=true`；初次 NLL 为 `-1.3006`，噪声参数正常更新。

```bash
PYTHONPATH=RoboTwin/policy/RAFE/ManiFlow \
/opt/shared/Prog/anaconda3/envs/maniflow/bin/python -m pytest \
  RoboTwin/policy/RAFE/tests \
  RoboTwin/policy/RAFE/ManiFlow/tests/test_ppo_optimizer.py \
  RoboTwin/policy/RAFE/ManiFlow/tests/test_eval_checkpoint.py \
  RoboTwin/policy/RAFE/ManiFlow/tests/test_baseline_failure_selection.py \
  RoboTwin/policy/RAFE/ManiFlow/tests/test_rl_progress.py -q
```

训练脚本也已通过 `--cfg job` 检查，包含带 `=` 的 checkpoint 文件名，确认
Hydra override、RAFE 输出目录、无参考配置及部署 policy 名能正确解析。

当前 CUDA 不可用，真实仿真尝试失败于 `No CUDA GPUs are available`，继而
`CuroboPlanner` 导入失败。尚未验证完整物理训练、GPU AMP 和真实 Actor
成功率提升。新参数化的单批全 step 诊断约 89.7% 的逐元素残差低于 sigma 下限，
约 3.0% 高于 sigma 上限；这只是初次小批检查，不能说明长期训练趋势。
优先观察日志，再决定是否显式调整 sigma 范围；`residual_scale` 不再限制范围。实现未
添加正则或改变损失。该方法只检验残差尺度引导探索的假设，不预设它提高收益。
