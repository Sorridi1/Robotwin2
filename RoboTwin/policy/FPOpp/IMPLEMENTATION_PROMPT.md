# 实现与复核 prompt

请在 `/home/ljj/code/Robotwin2/RoboTwin/policy/FPOpp` 中实现/复核“使用 FPO++ 方法微调现有 ManiFlow，并在 RoboTwin2 上比较”的代码。直接使用已有 maniflow Conda 环境，不新建环境、不修改已安装包；所有新增实现限制在本目录。

## 输入与修改边界

读取 `/home/ljj/code/fpo-control/manipulation_experiments/src/flow_model.py`、`finetune_online_rl.py`、README、docs/reproduce.md 及操作任务 sweep，以其 manipulation PPO 分支为算法依据。上游 HEAD 为 `8f460e4415a2a6debfbeb689093f5e36140bd178`。

读取本地 ManiFlow 的 policy/maniflow_pointcloud_policy.py、model/rl/pointcloud_critic.py、env/robotwin/robotwin_rl_env.py、common/robotwin_env_pool.py、common/ppo_buffer_pointcloud.py 和原 RL workspace/config；只读复用。

读取 RoboTwin/script/eval_policy.py、eval_scene_selection.py、eval_seed_bank.py 和 ManiFlow 部署接口；通过新增 FPOpp 接口接入公共评估器。不要改任务、奖励、成功条件、现有源码、用户暂存改动、checkpoint 或种子池。

准确源码路径为 `/home/ljj/code/Robotwin2/RoboTwin/policy/ManiFlow/ManiFlow`，不含 `.0`。入口相对文件位置定位源码，优先加入本进程及子进程搜索路径，处理失效的旧 editable 安装；不要求重装包。

## 对比定义

相同 ManiFlow BC 起点，改用 FPO++ 的微调目标。严格从输入 checkpoint 内 cfg/权重恢复结构与 normalizer，默认读 ema_model。checkpoint 有 plan 就保留并冻结 plan predictor，没有就维持没有；不要强制换成无 plan checkpoint 或随机补充新模块。

冻结观测 encoder，微调 flow 网络；不引入学习噪声、reference residual、BC anchor 或基于成功率的 actor 跳过/降权。不能用本工作的高斯去噪链 log-prob 伪装 FPO++ ratio。明确这是算法移植对比，不声称原论文仿真成绩复现。

## 核心目标

1. 原始 FPO-control 使用 data→noise 的时间方向，本地 ManiFlow 使用 noise→data。采用本地 `x_t=(1-t)*eps+t*a`，速度目标 `a-eps`。
2. CFM 计算用瞬时速度条件 dt=0：relative target_t=0，absolute target_t=t；ODE 采样沿用 checkpoint 的有限 dt/采样步数。
3. 一个 chunk 采样 N=8 组 `(t,eps)`。t 在 chunk 的位置间共享；eps 为完整 `[B,N,H,D]`。同一轮所有 PPO epoch 重用完全相同的 t、eps、action target、观测条件与 old loss。
4. modified Huber delta=.5：小误差为 e²，大误差为 `2*delta*abs(e)-delta²`，无0.5系数。先对动作维 D 求均值。
5. 对 old loss 的每个位置作 cap=4，current loss 不同样 cap；再按实际执行 mask 对 chunk 求和。默认不对 chunk 求均值。
6. 保留 N 维，`logratio=old-new` 经 `x+(clamp(x,-5,5)-x).detach()`，ratio=exp(logratio)，逐 MC 样本进行 PPO clipping 后才平均。不要先平均 MC loss 再 exp。
7. old-only cap 会让饱和位置在相同权重时 ratio 也不为1。必须忠实复现，并分别测试未触及 cap 的单位 ratio 和触及 cap 的例外。
8. CFM action target、观测条件与 old loss detach。MC 分片分别反传，以样本占比加权；梯度累积按真实 batch 大小归一化。

## 环境与动作

- 复用相同 RoboTwinRLEnv、cheap validator、奖励与 sum 聚合、qpos 控制、pointcloud/agent_pos 输入和 normalizer。
- 保留完整 horizon 做 DiTX 前向；actor loss 只覆盖实际执行窗口。历史位置、成功/超时后的未执行动作不进入 loss。
- 现有 H=16/To=2/n_action_steps=16 只返回15步，不补动作。记录实际执行数，对齐真实动作预算。
- 采样为随机 Gaussian 初始 latent + 确定性 ODE，无逐步 SDE 噪声。默认保留 native ManiFlow 不裁剪协议；对齐旧 PPO 的裁剪时显式配置并记录双方相同设置。CFM 有效 action target 必须对应实际执行的裁剪后动作。
- 使用本地 chunk transition GAE，gamma=.99/lambda=.95；成功和超时都切断递推，不用自动 reset 后的观测给上一 episode bootstrap。明确这与官方逐动作采集/聚合实现的时间尺度差异。
- buffer 自己持有 CPU 副本，不能引用 worker 下一步会覆盖的共享内存；不混入跨 episode 的 chunk。

## 训练、恢复和评估

独立配置和 train/eval shell 入口，checkpoint/任务/GPU/seed/output 均可指定。保存 flow/normalizer、critic、optimizer、AMP scaler、奖励缩放、iteration、真实动作计数、训练 seeds、RNG 和 next seeds。resume 严格限制改变算法/环境配置，允许增加预算；物理场景重新 reset，不承诺逐位恢复。

新增 get_model/eval/reset_model，使用原公共 evaluator。model 暴露 run_dir、epoch；开始补齐观测历史，成功/超时立即停，episode 结束清空缓存。记录随机 latent 与 zero latent 的不同推理协议。

双方共享同一 test bank 和筛选模式。周期 best 只用独立 validation bank，测试集不参与选择。记录并检查实际 train/validation/test seeds；公共池已有组时换 eval_seed 不会换场景。正常失败计入分母，仿真异常不悄悄换 seed。

## 验证与交付

复核/补充本目录 checks：modified Huber/STE 梯度、官方 reduction 顺序、MC 非线性与分片等价、固定随机变量、执行 mask、chunk GAE 和 auto-reset 边界、严格 normalizer/plan 加载、小型真实 ManiFlow 的 collect/update/save/resume、原评估接口的提前停止。

在已有环境中验证真实 checkpoint 的 GPU ODE/CFM 更新；需要区分 synthetic observations、fake simulator 和真实 RoboTwin rollout，不能把前两项称作仿真实测。真实 smoke 使用独立临时种子/目录，不触碰正式 test bank，不启动大规模训练。

最终给出能直接执行的 train/resume/eval 命令、配置与改动范围、实际通过的测试和未完成验证。保持现有工作代码与 Python 安装不变，不自动提交。
