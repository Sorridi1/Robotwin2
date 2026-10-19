# RL 仿真优化

实施要求见 [rl_simulation_optimization_prompt.md](rl_simulation_optimization_prompt.md)。
场景筛选已进一步改为默认 cheap 检测；API、缓存兼容与实测见 [scene_validation.md](scene_validation.md)。

## 当前默认行为

RL 配置启用 `subprocess` 后端，4 个常驻 spawn worker 并发执行环境，主进程继续批量推理并同步收集 PPO 样本。点云和关节历史通过共享内存传递；写入 buffer 的观测是独立副本。worker 本地自动 reset，同时返回 reset 前的 reward、terminated、truncated 和 info。

RL 只生成头部的带颜色点云与关节状态，关闭腕相机和独立 RGB 图像导出。点云的 RGB 通道保留。`get_obs()` 仍在每个执行动作后调用，`n_obs_steps=2` 的时间间隔保持原状。普通采集和评估默认不启用这些 RL 参数。

`render_on_observation_only=true` 只跳过不需要图像时的中间渲染同步，每次 `get_obs()` 都先强制同步。viewer、录像、数据保存或动态随机光照启用时保留原路径。默认保持 `rt / 32 samples / path depth 8 / oidn` 的视觉设置；可显式覆盖采样数，但视觉质量变化需要单独评估。

正常场景关闭按 `clear_cache_freq` 清理缓存；相机分配失败时强制清理。PPO、控制器、物理步长、稳定性检查、成功判断及 checkpoint 格式不作修改。

## 启动与回退

在 `RoboTwin/policy/ManiFlow` 目录、`maniflow` 环境执行原来的训练命令即可。脚本现在支持第九个参数之后的 Hydra overrides，例如：

```bash
EVAL=false bash scripts/train_rl_policy.sh \
  reinflow_rl_pointcloud_robotwin2 place_empty_cup demo_messy_ur5 50 \
  coarse_parallel 0 1 \
  /media/Elements1/ljj/ManiFlow/outputs/place_empty_cup-maniflow_pointcloud_policy_robotwin2-0923_seed0/checkpoints/latest.ckpt \
  demo_messy_ur5 \
  rl.rollout.backend=subprocess rl.rollout.n_envs=4
```

`EVAL=false` 仅关闭脚本末尾的额外评估；训练中的周期评估由 `rl.eval.enabled` 控制。性能对照时可追加 `rl.eval.enabled=false`，正式消融三组应统一评估预算。

单独回退串行：`rl.rollout.backend=serial`。回退原传感器/同步/专家检查路径可追加：

```text
rl.rollout.backend=serial
rl.env.head_pointcloud_only=false
rl.env.render_on_observation_only=false
rl.env.expert_seed_cache_mode=off
rl.env.scene_validation.mode=expert
rl.env.clear_cache_freq=1
```

同一 rollout 的策略版本固定。并行 worker 使用独立 RNG，reset 不再重置主进程的 NumPy / torch RNG，因此同一个训练 seed 不保证与旧串行版本逐位一致。噪声头三组消融应使用相同后端、优化设置、checkpoint 和训练 seed 集合。更改环境数会改变每轮数据量；比较吞吐量时也要报告 transition 总数。

## 场景验证缓存与预验证

缓存位于 `rl.env.expert_seed_cache_dir`，按验证模式、选项/阈值、配置/代码指纹和 seed 存储。默认 cheap 模式不执行专家，通过检查后保留同一个初始场景。命中有效 seed 后仍检查新场景；缓存的无效 seed 仅在同一验证模式内跳过。没有固定小 seed 池反复循环。

- `readwrite`：复用同模式记录，未命中时执行该模式的检查并持久化。
- `readonly`：必须事先完成对应 seed 的验证；缺少条目明确报错。
- `off`：每次按原路径验证。

缓存包含初始化元信息或 expert 模式执行的 `episode_info`。进程锁和原子替换保护文件完整性。显式判定失败/场景不稳定可缓存；未知运行时异常不缓存为不可完成。

指纹覆盖任务参数、任务 YAML、机器人配置、相机配置以及 envs 下 Python 源码和 RL wrapper。更改 mesh、URDF、贴图、外部规划库或其他未纳入哈希的资源后，须更新 `rl.env.expert_seed_cache_namespace` 或换缓存目录。缓存并不保证外部规划器在所有平台上逐位确定。

离线预验证不加载 actor。例如训练 seed 0、worker 0，从 100000 开始检查 100 个候选 seed：

```bash
CUDA_VISIBLE_DEVICES=1 \
VK_ICD_FILENAMES="$PWD/scripts/nvidia_icd.json" \
CUROBO_TORCH_COMPILE=0 CUROBO_TORCH_COMPILE_DISABLE=1 \
python scripts/prevalidate_rl_seeds.py \
  --task-name place_empty_cup --task-config demo_messy_ur5 \
  --training-seed 0 --env-index 0 --count 100
```

其他 worker 分别使用 `--env-index 1/2/3`，默认起点为 101000/102000/103000。`--count` 是候选数，包含不通过的 seed；足够长的训练需要继续扩展范围。可用 `--start-seed` 指定范围，用 `--cache-dir` 更换位置。若训练覆盖了环境参数，预验证也须传入相同的 Hydra overrides，否则会产生不同指纹。

默认 cheap 模式对新 seed 也只运行轻量检查。显式选择 `scene_validation.mode=expert` 时，新 seed 仍需要原有专家工作；旧模式缓存收益主要来自多个消融/重启复用验证结果。

## 计时与小规模验证

终端每个 iteration 使用同一条进度条，从 `reset` 切换为 `rollout`、再切换为 `update`，结束后保留最后一行。`succ=S/N` 表示本轮已结束 episode 中成功数/总数，不是最近一个动作批次的瞬时成功数；尚无 episode 结束时显示 `0/0`。`step` 是累计实际执行的动作数，`(+K)` 是刚完成批次的动作数；`env_step` 是各环境返回的 episode 动作计数（不一致时显示最小值..最大值）。rollout 的 `n/total` 是并行动作批次数；update 阶段是 minibatch 数，`opt=n/total` 单独显示梯度累积后的优化器更新次数。KL 提前停止保留实际计数并标记 `KL-stop`，不伪装为全部完成。worker 不单独打印进度条。

训练记录：`time/rollout_sec`、`time/ppo_update_sec`、`rollout/actor_inference_sec`、`rollout/env_wall_sec`、`rollout/initial_reset_wall_sec`，以及 `rollout/worker_*_sum` 的环境/重置/专家耗时和缓存命中数。worker 累计耗时可以大于并行墙钟，不能当作总耗时直接相加。

不加载模型的小规模真实仿真测试：

```bash
CUDA_VISIBLE_DEVICES=1 \
VK_ICD_FILENAMES="$PWD/scripts/nvidia_icd.json" \
CUROBO_TORCH_COMPILE=0 CUROBO_TORCH_COMPILE_DISABLE=1 \
python scripts/check_rl_simulation.py --backend subprocess --n-envs 2 --steps 2
```

它执行保持当前位置的关节命令，验证进程、相机、TOPP、物理步、点云、共享内存和退出清理，不等价于完整任务成功率验证。`--backend serial`、`--legacy-sensors` 可用于短程对照，`--expert-check` 测试缓存路径，`--output /tmp/report.json` 保存报告。

## 减少中间观测会怎样影响训练

本次保留逐动作完整观测。对于预先生成、chunk 内不重新决策的动作序列，如果中间视觉帧完全没有被消费，且保留最后两个相邻动作后的观测、逐动作 reward、终止判断以及实际物理轨迹，那么省略未消费帧在理想条件下不改变 PPO transition 定义。

但当前控制代价读取上一动作后的关节状态；需要用轻量状态读取替代，不能直接用旧图像对应的旧关节状态。只保留 chunk 最后一帧或首尾帧会改变两帧输入的时间语义。提前结束、短 chunk、时间截断和未来的 terminal bootstrap 都必须保留对应终止观测语义。

此外渲染可影响随机数消耗、随机光照和点云颜色噪声；相机/FPS 也可能具有状态。因此即使数学上未消费中间帧，实际轨迹也不保证逐位相同。后续若实现跳帧，应先固定初始场景与动作，对照关节/物体状态、每步 reward、done 标志和最终两帧，再比较多 seed 的训练成功率。

## 初版并行/渲染优化验证结果

- `conda run -n maniflow python -m pytest -q tests`：26 passed，覆盖真实 spawn/共享内存、并发屏障、worker 异常/超时回收、workspace 串并行 buffer 一致性、缓存并发/只读/失效，以及逐动作观测和渲染保护条件。
- GPU 3 上两个真实 SAPIEN worker 完成 reset、TOPP、物理步和有限值点云检查，并正常关闭。
- 对 seed 100000 完成真实专家预验证；后续训练环境的 readonly reset 报告 `expert_cache_hits=1`，没有 `expert_check_sec`。
- 保持关节位置的短动作测试中，原传感器串行路径、优化路径和并行路径的关节状态、reward、terminated/truncated 一致。
- Python 编译、shell 语法、`git diff --check` 通过。

这些检查验证了实现和短程控制语义，不代表完整任务成功率或训练加速倍数。报告暂存于 `/tmp/maniflow-serial-legacy-smoke.json`、`/tmp/maniflow-subprocess-smoke.json`、`/tmp/maniflow-expert-cache-hit-smoke.json`。
