# 在线 RL 场景验证

评估也可复用相同检查，或选择专家失败但检查合法的困难子集，见 [评估场景筛选](eval_scene_selection.md)。原评估默认仍保持专家成功筛选。

默认 reset 流程现在是 `seed -> setup_demo -> cheap validation -> 保留当前场景 -> get_obs -> rollout`。
合法场景只初始化一次，不执行 scripted expert；检查失败时关闭该候选场景，递增 seed 再试。
目标是排除明显非法状态，不证明任务可解，也不筛选容易成功的场景。

## API 与配置

独立模块：`maniflow/env/robotwin/scene_validator.py`。

```python
SceneValidationResult(valid: bool, reason: str, metrics: dict)
CheapSceneValidator(config).validate(task_env) -> SceneValidationResult
```

可选 task hook：`cheap_validate_scene(config) -> (valid: bool, reason: str, metrics: dict)`。
`place_empty_cup` 已实现示例；其他 task 无须实现 hook，仍运行通用检查。
轻量元信息由 `get_scene_metadata()` 提供，`place_empty_cup` 无须执行 expert 就能提供 cup/coaster 标签。
当前 `language_conditioned=false` 时，cheap/none reset 不生成 instruction。

```yaml
rl:
  env:
    scene_validation:
      mode: cheap                 # none / cheap / expert
      max_seed_tries: 100
      camera_allocation_retries: 3
      check_finite_state: true
      check_joint_limits: true
      joint_limit_tolerance: 0.05
      reject_initial_success: true
      check_initial_penetration: true
      max_initial_penetration: 0.03
      enable_task_specific_check: true
      log_rejection_reason: true
      task_specific:
        workspace_min: [-0.65, -0.40, -0.10]
        workspace_max: [0.65, 0.40, 0.75]
```

新字段 `scene_validation.mode` 优先；没有该字段时，旧 `expert_check=true/false` 分别映射为 `expert/none`。
旧布尔参数仍可解析，但在新的默认 YAML 中单改它不会覆盖 `scene_validation.mode`。
`expert` 保留专家执行、`plan_success && check_success()` 筛选及验证后重建。
`none` 只初始化，不调用专家或 cheap validator；`UnStableError` 仍拒绝重试，其他初始化异常记录后抛出。

## 具体检查与限制

| 检查 | 当前实现 | 拒绝原因 |
|---|---|---|
| 初始化稳定性 | 处理 setup_demo 原有稳定性检测抛出的 UnStableError | unstable_setup |
| 机器人状态 | 两侧 articulation 的 qpos、qvel、root pose 全为 finite | nonfinite_robot_state |
| 关节限位 | 使用 get_qlimits，逐侧忽略无限边界，允许配置容差 | joint_limit_violation |
| 物体位姿 | scene 的 rigid actors 和 articulations 的位置/四元数为 finite | nonfinite_actor_pose |
| 严重穿透 | 使用 contact point.separation；仅当负 separation 的深度超过阈值才拒绝 | severe_penetration |
| 初始成功 | 原始场景调用 check_success，不执行动作 | initial_success |
| place_empty_cup 几何 | cup/coaster root pose 有限、位于宽松的桌面相对包围盒中 | task_specific_invalid |

当前安装的 SAPIEN 真实提供上述状态 API 与 contact `separation`。正常物体/桌面接触合法。
如果其他版本没有 contact separation，穿透检查只报告 `penetration_check_supported=false`，不根据接触数量拒绝。
接触检查依赖物理引擎已生成的 contact，不是完整网格穿透证明。

未实现完整 IK、motion planning、任务轨迹可解性证明，也不按距离难度、reward 或任何策略成功率筛场景。
原有 setup_demo 的 2500 子步稳定性检查及机器人/规划器初始化仍保留；cheap validator 自身不推进仿真、不渲染、不执行机器人动作。

容差单位随关节类型为弧度或米；穿透阈值为 0.03 米。
workspace 是相对实际 table pose 的三维范围，明显宽于原始 spawn 范围，不限制 cup/coaster 的相互距离。
这些 sanity 阈值均可配置；本次实测没有为通过测试而放宽或关闭检查。更换机器人/任务后应检查对应几何范围。

## 保留缓存

沿用现有 `expert_seed_cache_dir/mode/namespace` 字段和进程锁、原子写入；名称为历史兼容。
缓存指纹新增 validation mode、检查选项/阈值、validator 源码，继续覆盖任务与仿真相关代码/配置。
旧 expert 失败记录不与 cheap 模式共享；阈值变更也使用不同缓存。

cheap 模式可以跳过已记录为非法的 seed；命中有效 seed 时仍 setup 一次并重新检查新物理状态，不凭缓存直接放行。
expert 模式命中时可复用专家验证结果。资源分配等未知异常不写为非法 seed。
readonly 缺少记录明确报错。none 模式不使用筛选缓存。

预验证入口默认也改为 cheap：

```bash
CUDA_VISIBLE_DEVICES=1 \
VK_ICD_FILENAMES="$PWD/scripts/nvidia_icd.json" \
CUROBO_TORCH_COMPILE=0 CUROBO_TORCH_COMPILE_DISABLE=1 \
python scripts/prevalidate_rl_seeds.py --mode cheap \
  --task-name place_empty_cup --task-config demo_messy_ur5 \
  --env-index 0 --count 100
```

在线 cheap 模式已经很轻，无需先离线跑完验证才能开始 readwrite 训练。
用于旧方案对照时，把原训练命令最后追加 `rl.env.scene_validation.mode=expert`；完全关闭检查用 `...mode=none`。

## 统计

`RoboTwinRLEnv.last_validation_info` 暴露：mode、attempts、rejected_count、accepted_seed、validation_time_sec、setup_time_sec、setup_demo_count、play_once_count、cache_hits/misses、rejection_reasons、setup_exception_types、last_metrics。
这些统计也由 `last_reset_info['validation']` 跨 worker 返回，并汇总到 W&B 的 `rollout/scene_validation/*`。
合法 episode 不逐条打印；有拒绝时仅汇总原因。未知初始化异常保留异常类型与 traceback。

## 实际验证结果

- 完整测试：45 passed；覆盖 cheap 不调用 play_once、合法场景只 setup 一次、非法候选关闭与 seed 重试、旧 expert/none、缓存隔离/重新检查、finite/限位/接触/初始成功及任务 hook。
- 真实 PPO smoke：加载 0923 的现有 checkpoint，当前 PPO 配置、两个 subprocess 环境、每环境两步，成功收集 4 条 transition、60 个实际动作；优势值有限，存储链的 log-prob 重算一致。不更新模型、不执行完整训练。
- 相同任务、相同 seed 100000/100001/100002、各三次 reset，关闭缓存、相同渲染与清理设置，GPU 3 上顺序实测：

| 模式 | 三次总耗时 | 平均耗时 | 后两次平均 | setup_demo 总次数 | play_once 总次数 |
|---|---:|---:|---:|---:|---:|
| expert | 29.572 s | 9.857 s | 5.013 s | 6 | 3 |
| cheap | 26.809 s | 8.936 s | 2.914 s | 3 | 0 |

cheap 检查本身耗时约 1.46–2.02 ms。每组第一次 reset 包含 CUDA/规划器/场景冷启动，且是顺序小样本测量，不能据此推断完整训练加速比例。
三个自然场景和 smoke 中未触发拒绝；受控测试实际验证了非法状态/候选拒绝和重新初始化，没有用放宽阈值掩盖失败。
原始报告：`/tmp/maniflow-scene-validation-report.json`。

可复现命令（从 `RoboTwin/policy/ManiFlow` 运行）：

```bash
CUDA_VISIBLE_DEVICES=1 \
VK_ICD_FILENAMES="$PWD/scripts/nvidia_icd.json" \
CUROBO_TORCH_COMPILE=0 CUROBO_TORCH_COMPILE_DISABLE=1 \
python scripts/check_scene_validation.py --modes cheap expert --resets 3 \
  --smoke-checkpoint /media/Elements1/ljj/ManiFlow/outputs/place_empty_cup-maniflow_pointcloud_policy_robotwin2-0923_seed0/checkpoints/latest.ckpt \
  --output /tmp/maniflow-scene-validation-report.json
```

## 本次文件范围

- 新增 `scene_validator.py`、`tests/test_scene_validation.py` 和本说明。
- `robotwin_rl_env.py`：验证模式、单次 setup 的 cheap reset、错误处理、模式隔离缓存、统计。
- `robotwin_seed_cache.py`：增加 lookup/store 接口，保留并发锁与原子写入。
- `reinflow_rl_pointcloud_robotwin2.yaml`：显式 cheap 默认及可配置阈值。
- `_base_task.py` / `place_empty_cup.py`：轻量元信息 hook、杯子任务几何 hook。
- RL workspace：仅汇总场景验证统计。
- 预验证/仿真检查脚本更新模式参数；新增 `scripts/check_scene_validation.py` 提供真实 reset 对照和 PPO smoke。
- PPO/ManiFlow/noise head/critic/BC/action chunk 实现均未修改。

cheap 会保留原先因专家失败被排除的物理合法场景，这是本次有意的训练分布变化；与旧实验比较时需明确标注验证模式。
