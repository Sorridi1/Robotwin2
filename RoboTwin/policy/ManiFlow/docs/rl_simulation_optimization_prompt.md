# RL 仿真优化实施 prompt

在现有 ManiFlow + PPO RoboTwin 训练代码上实现以下优化，并提供可运行的配置、使用方法和验证结果。

后续场景筛选要求已更新：在线 RL 默认采用 cheap 场景检测并保留同一合法场景；专家验证只用于显式 expert 对照模式。具体 API、检查条件和验证结果以 [scene_validation.md](scene_validation.md) 为准，下面的专家缓存要求保留为兼容模式。

## 目标与边界

- 保留每个动作后的完整观测、两帧历史、reward、成功判断、TOPP 和物理步长；本次不实现 chunk 内跳帧。
- 保留 action-plan / default / plan / reference_flow 的算法实现和已有 checkpoint 接口。
- 优化只通过 RL 参数启用；普通数据采集、监督训练、评估默认保持原有渲染行为。保留串行回退路径。
- 保留工作区现有用户改动，不修改共享 demo_messy_ur5.yml 来启用 RL 优化。

## 一、减少无用相机与渲染工作

- RL 可选只采集头部 RGB 点云和关节状态；关闭无用腕相机和独立 RGB 数组导出，保留点云颜色。
- 去掉 RL wrapper 中未被 actor / critic 使用的 head_cam 转换和历史堆叠。
- 无 viewer、无录像、无数据保存、无动态随机光照时，可延迟中间渲染同步至 get_obs；每次观测前强制同步。
- 将当前硬编码的 ray-tracing 设置参数化；默认维持 rt / 32 samples / depth 8 / oidn。降低质量须由用户显式配置。
- 让 clear_cache_freq 真正控制正常关闭时的清理频率；分配失败时仍强制清理和重建。

## 二、缓存专家验证

- 持久化按 seed 的专家验证结果和 episode_info；缓存必须绑定任务、有效场景/机器人配置以及仿真相关源码指纹。
- 相同 seed、相同指纹命中后跳过专家执行，实际 rollout 仍重新初始化原始场景并进行原有稳定性检查。
- 支持 off / readwrite / readonly 模式；readonly 缺失应明确报错，不应悄悄取消验证。
- 使用进程锁与原子文件替换，避免并发写坏；不把资源异常等暂时性失败永久缓存为不可完成 seed。
- 提供离线预验证入口。冷缓存遇到新 seed 仍需验证，不宣称首次训练可以免费省掉这部分计算。

## 三、并行环境采样

- 使用 spawn 创建常驻环境进程；每个进程独立创建 SAPIEN / CUDA 资源和随机状态。
- 主进程批量进行 actor / critic 推理，将动作同时发给所有 worker，等待本轮结果后写入原有 PPO buffer。
- worker 在 episode 完成时本地 reset，终止标志和 reward 保持原始 transition 语义。
- 点云和关节观测支持共享内存传输；进程异常、超时、训练退出和周期评估都必须回收进程与共享内存。
- 保持同一 rollout 内策略固定，不引入跨策略版本的异步样本。提供 serial / subprocess 配置切换。
- 记录 rollout、actor 推理、环境 step/reset、PPO update 等基础耗时，区分并行墙钟与 worker 累计耗时。

## 验证与交付

- 用轻量假环境验证串行/并行 transition、自动 reset、共享内存、异常传播及清理；验证缓存指纹、并发写和缓存命中。
- 验证渲染延迟只减少同步次数，每个动作的观测仍生成；验证动态随机光照保留原路径。
- 执行现有测试与必要的真实 SAPIEN 小规模对照，不启动完整训练。不报告未经测量的加速倍数。
- 文档包含训练启动、回退、预验证命令，以及减少中间观测对训练影响的条件与限制。
