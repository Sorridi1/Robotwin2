# 用 FPO++ 微调 ManiFlow，在 RoboTwin2 上评估

直接使用现有 **maniflow Conda 环境**。本目录实现 FPO++ 的 manipulation 优化目标，复用现有 ManiFlow 模型、RoboTwinRLEnv 和公共评估器；不修改本工作的源码、仿真任务或已安装的 Python 包。

用户指定的0428起点、训练命令、结果目录和保存逻辑见 [RUN_0428.md](RUN_0428.md)。下面0923示例保留为此前连通性验证的参考，不是固定的训练起点。

官方与实际训练参数对照、推荐起始值见 [PARAMETERS_0428.md](PARAMETERS_0428.md)。新增 `configs/fpopp_0428_recommended.yaml` 供显式选择；原默认配置保留，推荐值尚未通过完整训练证明最优。

模型结构和 normalizer 严格读取输入 checkpoint 自带的 cfg/权重，默认使用 BC 的 ema_model。checkpoint 自带 action-plan 时完整保留并冻结 plan predictor；无 plan 时保持无 plan。微调 flow 网络，冻结观测 encoder，不添加学习噪声、参考残差噪声、BC anchor 或成功率门控。该对比的含义是“相同 ManiFlow 起点，改用 FPO++ 微调目标”。

## 运行

从仓库根目录执行。下面使用实际存在的 0923 BC checkpoint 举例；正式实验应换成与本工作相同的 BC 起点。

```bash
cd /home/ljj/code/Robotwin2
conda activate maniflow

# 只检查 checkpoint 的配置，不加载完整模型或创建仿真。
bash RoboTwin/policy/FPOpp/train.sh \
  --checkpoint /media/Elements1/ljj/ManiFlow/outputs/place_empty_cup-maniflow_pointcloud_policy_robotwin2-0923_seed0/checkpoints/latest.ckpt \
  --dry-run

# 训练；GPU 号由 CUDA_VISIBLE_DEVICES 控制，程序中使用可见设备 cuda:0。
CUDA_VISIBLE_DEVICES=0 bash RoboTwin/policy/FPOpp/train.sh \
  --checkpoint /media/Elements1/ljj/ManiFlow/outputs/place_empty_cup-maniflow_pointcloud_policy_robotwin2-0923_seed0/checkpoints/latest.ckpt \
  --task-name place_empty_cup --task-config demo_messy_ur5 --seed 0 \
  --output-dir RoboTwin/policy/FPOpp/output/cup_seed0

# 从完整训练状态恢复；可增加迭代/动作预算。
CUDA_VISIBLE_DEVICES=0 bash RoboTwin/policy/FPOpp/train.sh \
  --resume RoboTwin/policy/FPOpp/output/cup_seed0/checkpoints/latest.ckpt \
  max_iterations=200 total_action_steps=200000
```

也可在不激活环境时设置 `PYTHON_BIN=/opt/shared/Prog/anaconda3/envs/maniflow/bin/python` 调用两个 shell 入口。入口自动将**当前 checkout**放到 Python 搜索路径最前，无需重新安装 maniflow。

配置在 [configs/fpopp_robotwin2.yaml](configs/fpopp_robotwin2.yaml)。可追加 `rollout.n_envs=2 fpo.mc_samples=8 ppo.actor_lr=3e-6` 等已有键；未知键报错。输出已存在 checkpoint 时必须显式 resume，避免覆盖。

## 评估与 best

接入原 `RoboTwin/script/eval_policy.py`，成功条件、场景筛选和公共种子机制保持一致：

```bash
bash RoboTwin/policy/FPOpp/eval.sh \
  RoboTwin/policy/FPOpp/output/cup_seed0/checkpoints/latest.ckpt \
  place_empty_cup demo_messy_ur5 10 0 \
  --eval_scene_selection rl --eval_num_episodes 100 \
  --eval_scene_seed_file /home/ljj/code/Robotwin2/RoboTwin/policy/FPOpp/output/shared_test_seeds.json \
  --eval_output_dir RoboTwin/policy/FPOpp/output/cup_seed0/test_evaluations/first_test
```

位置参数最后两个是评估 seed 和物理 GPU 号；`visible` 表示沿用外部 CUDA_VISIBLE_DEVICES。双方必须使用**同一个**测试池文件、同一个 task/config/mode 分组、同一个场景前缀。本工作原脚本可通过 `EVAL_SCENE_SELECTION=rl` 和 `EVAL_SCENE_SEED_FILE=上面的同一路径` 复用该池。新建池只为分开 train/validation/test，不能给两种方法各建一批不同场景。

默认不做周期评估，只保存 latest。需要验证集 best 时，训练命令追加：

```text
validation.interval=20 validation.seed_file=/home/ljj/code/Robotwin2/RoboTwin/policy/FPOpp/output/shared_validation_seeds.json
```

validation 默认候选 seed=20，test 示例 seed=10，训练 seed=0。仍须核对实际 seeds；公共池已有组时，修改 eval_seed 不会替换旧组。程序记录真实训练 seeds，检查已有评估池交集；新建评估池和其它方法的训练集合也应在最终汇总时交叉检查。`best.ckpt` 是按 validation 成功率选择的 actor 导出，不能用它恢复 optimizer；resume 使用 `latest.ckpt`。

周期评估启用后，默认也执行最后一轮评估（`validation.run_final=true`）。每次评估之前先保存已完成的训练更新，评估成功后再保存 best 分数。独立评估遇到同轮次/同模式的已有结果时拒绝覆盖；训练内的重试自动写入本 run 的 `evaluation_retries` 新子目录。

默认评估保留随机初始 latent 的 ODE，episode 内使用独立策略 RNG，episode 开始按 sampling_seed+episode_index 重置。`--zero_latent True` 是另一个明确标注的推理协议，不与默认成绩混用。结果写入所选输出目录的 `eval_results/epoch_N/scene_rl/`，该模式目录内另保存 `_fpopp_protocol.json`；模型 checkpoint 和正式训练目录保持不变。

## 算法与环境约定

第 20 轮 63% 的排查记录见 [2026-11-04 审计记录](AUDIT_20261104.md)。临时增加的环境配置变化检测、快照归档及同协议 BC 入口已撤回；保留原有训练配置保存和评估结果防覆盖逻辑。

- FPO++：每个 chunk 8 组 MC；对动作维求均值，old loss 按位置 cap=4，再按已执行窗口求和；每组 MC 独立计算和 clip ratio，最后平均。modified Huber delta=0.5、STE logratio cap=5、PPO trust region；对应官方 manipulation 分支，不默认启用 ASPO。
- 同一轮更新固定 action target、观测条件、t、eps、old loss；MC 分片反传限制显存，梯度累积按实际 minibatch 大小加权。
- 默认 AMP 使用 BF16；不支持 BF16 的 GPU 可用 `ppo.use_amp=false`，可选 FP16 则由 GradScaler 处理溢出跳步并记录计数。
- ManiFlow 时间方向为 noise→data，CFM 使用 dt=0 瞬时速度；采样保留 checkpoint 的有限 dt/ODE 步数。与原官方操作实验相比，本实现使用现有 RoboTwin **chunk transition GAE**，并明确保留此移植差异。
- 直接复用当前 ant_style 奖励、cheap 场景检查、qpos 控制与自动 reset pool。success/truncation 都切断 GAE，不使用 reset 后观测给上一 episode bootstrap。每轮 reset，resume 从保存的 next seed 重建新场景，不声称物理状态逐位恢复。
- `H=16, n_obs_steps=2, n_action_steps=16` 的有效窗口是 `[1:16]`，实际最多15步；没有补动作或改变 horizon。提前成功/超时只计算执行部分的 loss。
- 默认保留原生 ManiFlow 不裁剪的 ODE。现有 PPO actor 的训练曾使用 intermediate=1.5/final=1.0；需要对齐该训练协议时，双方明确采用相同设置，并给本命令追加 `actor.intermediate_clip=1.5 actor.final_action_clip=1.0`。实际裁剪和执行 target 保持一致，并记录配置与裁剪比例。
- 预算按真实 executed_action_steps 统计；在向量环境的一个 batch 后检查停止条件，最多超出一个 batch 的动作数，日志保留实际值。比较学习曲线时用实际动作数，并固定 checkpoint 选择规则与调参预算。

## 验证

```bash
python -m pytest RoboTwin/policy/FPOpp/checks -q

python RoboTwin/policy/FPOpp/scripts/check_environment.py --local-source --imports

# 真实 checkpoint + GPU actor 更新，输入是 synthetic observations，不创建仿真。
CUDA_VISIBLE_DEVICES=0 python RoboTwin/policy/FPOpp/scripts/smoke.py \
  --checkpoint /path/to/the/same/bc.ckpt --device cuda:0

# 真实 RoboTwin：1 env、2 chunk、1 次更新及保存；使用独立 smoke seed/临时输出。
CUDA_VISIBLE_DEVICES=0 python RoboTwin/policy/FPOpp/scripts/smoke.py \
  --checkpoint /path/to/the/same/bc.ckpt --device cuda:0 --env
```

CPU 测试使用小型的真实 ManiFlow 网络和 fake simulator 验证完整训练/部署链路，另检查公共评估报告的 complete/aborted 状态与 best 更新边界。真实 GPU actor smoke 已用 0923 checkpoint 验证：10步 ODE、输出 `[2,15,14]`、保留 plan、有限非零梯度。

真实 RoboTwin2 smoke 也已完成：1个环境、2个 chunk、30个实际动作、一次 BF16 actor/critic 更新、0次溢出，生成完整可恢复 checkpoint。逐项比较确认 encoder、plan predictor 和 normalizer 保持不变，flow 参数已更新。报告位于 `/tmp/fpopp-robotwin-smoke-bf16-20261103/report.json`。这只验证训练链路，不代表完成对比实验或获得有效成功率结论。

该 checkpoint 已通过原公共评估器完成1个真实场景（seed=1100000，200步上限，任务失败0/1，报告 status=complete），验证了加载、动作执行、失败计入分母和结果写入。报告在 `/tmp/fpopp-public-eval-smoke-20261103/eval_results/epoch_1/scene_rl/_scene_selection.json`；该单场 smoke 不能用于性能比较，且没有写入正式评估池。

相关材料：[环境与路径说明](ENVIRONMENT.md)、[实现/复核 prompt](IMPLEMENTATION_PROMPT.md)。官方依据：[项目](https://hongsukchoi.github.io/fpo-control/)、本地 `/home/ljj/code/fpo-control/manipulation_experiments`，检查时上游 HEAD 为 `8f460e4415a2a6debfbeb689093f5e36140bd178`。这是在 ManiFlow/RoboTwin2 上比较微调方法，不声称复现论文原始 robosuite 成绩。
