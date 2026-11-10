# Duco 擦白板：RoboTwin 单臂任务与 ManiFlow 推理

## 实现范围与当前状态

这是现有 **maniflow Conda 环境 / RoboTwin SAPIEN 3.0.0b1** 中的独立单臂任务，
不是另一个 ManiSkill 环境，也不连接真实机器人。

- `envs/duco_wipe_whiteboard.py`：继承 RoboTwin `Base_Task` 的任务注册。
- `duco_sim/scene.py`：单臂物理场景、7 维动作、相机、接触擦除、成功判定。
- `duco_sim/robot.py`：原 URDF 的六关节 FK、可追溯工具安装修正、临时 URDF。
- `duco_sim/policy.py`：严格加载指定 checkpoint、EMA、原训练归一化器与数据契约。
- `script/duco_wipe.py`：独立开发/原始回放/模型推理入口。
- `task_config/duco_wipe_whiteboard.yml`：Duco 专用配置。

没有修改已有任务、Robot 类、训练/RL loop、原始 URDF/meshes、数据集或 checkpoint，
没有安装、升级或降级任何依赖。新增入口不经过现有双臂 `eval_policy.py`，避免将 7 维
动作误当作双臂 14/16 维动作。未来若要进入统一评估器，应显式扩展单臂分支。

**推理链路已跑通，不表示预训练策略已经完成擦白板。** 第 50 条场景的 300 步
模型闭环检查仍为 `task_success=false`、覆盖率 0；报告不把运行结束当作成功。
加入严重跟踪偏差停止保护后，同一检查在第 218 步停止，原因是
`excessive_joint_tracking_error`，不是正常完成或成功抓取。

## 启动

```bash
conda activate maniflow
cd /home/ljj/code/Robotwin2/RoboTwin
```

SSH/headless 模型推理（当前已验证的运行方式）：

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_duco_wipe.sh infer \
  --demo 50 --headless --steps 300
```

默认读取用户指定的训练目录：

```text
/media/Elements1/ljj/ManiFlow/outputs/2026.08.27/20.07.11_train_maniflow_pointcloud_policy_wipe_whiteboard
```

目前该目录中唯一的 checkpoint 为：

```text
checkpoints/epoch=0500-val_loss=0.272452.ckpt
```

若未来目录有多个 checkpoint，必须使用 `--checkpoint /absolute/path/to/file.ckpt` 显式选择；
不会按文件名随意挑选。也可传 `--run-dir` 指向另一训练目录，但契约必须仍匹配单臂 Duco。

原始关节记录的物理回放对照，不调用策略，也不生成 task 修正轨迹：

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_duco_wipe.sh replay \
  --demo 50 --headless --steps 270
```

只验证原始 RGB-D 到 checkpoint 的推理，不启动仿真：

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_duco_wipe.sh check --demo 50
```

### 官方 GUI 与已知限制

已实现的是官方 `sapien.utils.Viewer`，保留默认 Scene/Entity/Articulation/Control
面板与鼠标自由视角。没有用 OpenCV、Pygame 或固定 RGB camera 替代 GUI。

**当前 GUI 尚未通过连续渲染验证。** 在现有 `:2` 桌面、GPU 2 上 `can_present=True`，
但旧 beta Viewer 连续渲染若干帧后报 `vk::DeviceLostError`。一个完全不含 Duco、
只显示方块的官方 Viewer 最小例子也复现了此问题，raster/RT 均复现。
这和之前的 `can_present=False` 不是同一个故障。没有为此修改依赖或驱动。

因此，目前建议先使用上面的 headless 入口。下面是 GUI 入口，保留给后续修复验证，
**不能视为已验证可用的指令**：

```bash
# 从自己的 Linux 桌面终端执行，继承该终端实际的 DISPLAY，不要随意设置 :2。
CUDA_VISIBLE_DEVICES=2 bash script/run_duco_wipe.sh view --demo 50

# 运行策略并打开官方 GUI（当前同样受 DeviceLost 限制）
CUDA_VISIBLE_DEVICES=2 bash script/run_duco_wipe.sh infer --demo 50 --steps 300
```

不能呈现时会明确报错，不做图像窗口回退。headless 与 GUI 传感器使用相同相机参数，
GUI 的自由视角与策略的固定传感器视角是两件不同的事。

## 与 checkpoint 完全一致的数据接口

训练契约从原 Zarr 的 `.zattrs` 读取，实际预处理直接复用
`maniflow.common.rise_duco_data.RGBDPointCloudPreprocessor`，不复制第二套算法。

- 点云：`1024 x 6`，XYZ 单位米，RGB `[0,1]`。
- 坐标：OpenCV 相机光学坐标，x 向右、y 向下、z 向前；不是 world/base 点云。
- 原相机内参、1280x720、深度毫米、5 mm voxel、裁剪范围与 FPS 采样均沿用训练契约。
- 状态：六个实际关节角（rad）加上最近夹爪命令（0 闭合 / 1 张开），与训练数据定义一致。
- 动作：六个绝对关节目标（rad）加夹爪目标，不是 TCP、delta 或归一化关节角。
- 观测历史为两帧，初始观测按原运行方式补齐；之后使用最新仿真观测更新。
- 从 checkpoint 的 `cfg.policy` 创建网络，严格载入 `ema_model` 与保存的 normalizer。
- 控制 10 Hz、物理 240 Hz，与真实数据约 10 Hz 的时序对应。

模型设置为 horizon 16、n_obs_steps 2。现有 `predict_action` 从索引 1 截取，
**实际返回 15 个动作**；不补造第 16 个。默认每次执行其中 4 步后用新观测重推，
可传 `--action-steps 0` 执行完整返回 chunk，再重推。

模型动作不经过 IK、抓取路点重建或演示轨迹替换。只做显式仿真保护：URDF 限位、
夹爪 `[0,1]`、每步关节目标最大变化 0.30 rad；严重/持续跟踪误差会停止。
原预测与真正应用的目标都会保存，不能把安全限幅说成原封不动执行。

## 场景和末端装配不是实物标定

场景复用 `/home/ljj/code/maniskill` 中现有的**纯数据/几何模块**，只读取 RGB-D、
部署文件内的标定数值和共享物体配置，不导入 ManiSkill，也不调用硬件接口。
因此当前这两个代码目录都是运行依赖；不能随意移走几何模块或原 meshes。

默认场景按 demo 的原始 RGB-D 重建；物体尺寸使用共享近似配置，包含明确记录的静态
防穿模位置修正。`--scene-profile` 可以显式加载匹配该 demo 的测量/修正后 profile。
模型推理时，数据集只提供初始场景/关节状态以及固定装配估计，不提供执行轨迹。

### 固定工具估计修正

之前原始回放中夹爪开合轴相对黑板擦短轴约差 90°，且手指悬空。
默认 `--tool-mount estimated` 在临时 URDF 中一起调整掌部 visual/collision 和两个
手指 joint 的安装变换，不只是旋转显示 mesh，不改变六个机械臂关节角。

当前修正是从第 50 条首次抓取拟合的**同一个固定变换**，所有 demo 共用：

```text
xyz_m   = [0.0102308082, 0.0301231265, 0.1218626787]
rpy_deg = [0, 0, 90]
verified = false
```

这不是硬件标定：场景定位误差、原 URDF 运动学误差可能被拟合吸收；第 50 条本身
还参与了估计，不能拿它证明泛化或独立评估性能。它只修复目前明显的装配不匹配，
没有证明真实转接件长度就是 121.9 mm。暂不添加未经测量的转接件外形。

查看完全未经装配修正的原始 URDF：

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_duco_wipe.sh replay \
  --demo 50 --headless --steps 270 --tool-mount raw
```

获得实测安装变换后，用单独 JSON 替换估计：

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_duco_wipe.sh infer \
  --demo 50 --headless --tool-calibration /absolute/path/to/measured_duco_mount.json
```

模板位于 `task_config/duco_tool_calibration.example.json`。这里填写的是保留原 mesh origin
后补入的 `link_6 -> duco_tool_mount` 装配修正，**不是控制器 TCP 的位置姿态**。
原始 URDF 和 meshes 始终只读，派生 URDF 在本次进程临时目录内生成。

## 任务物理与成功判定

- 黑板擦是独立动态刚体，通过夹指摩擦抓取；不绑定、不焊接、不随手 teleport。
- 双侧夹指接触、抬升至少 3 cm 且速度合理，连续满足 0.2 秒后才确认抓取；
  早先接触加上后来被弹飞，不会被当作成功抓取。
- 擦除必须同时有黑板擦/板面真实接触力、底面接近笔迹和实际覆盖。不是按时间或
  规划路径直接将图案变白。小段笔迹覆盖后隐藏，属于覆盖代理，不是颜料物理。
- 总成功要求确认抓取、覆盖至少 95%、擦的全部盒角落在托盘内、落在盘底且已释放。
- 原始回放结束、模型跑满步数，都不自动意味着任务成功。

## 输出与已做验证

默认写入新的 `outputs/duco_wipe/<timestamp>/`，不覆盖已有输出，也不会自动读取历史轨迹。
可用 `--output /path/to/new_directory` 改位置，目录必须尚不存在。

- `report.json`：checkpoint、数据契约、场景、固定工具估计、接触指标和停止原因。
- `duco_rollout.npz`：模型/源轨迹请求目标、实际应用目标、实际关节状态、擦的姿态和时间。
  `measured_state` 的第 7 维按训练定义是夹爪命令，不是假称测得的开口；
  实际 8 关节状态另存于 `measured_qpos`，顺序见 `active_joint_names`。
- `duco_initial_rgb.png` / `duco_final_rgb.png`：策略传感器的诊断图，不是 GUI 替代物。
- `duco_initial_pointcloud.npy`：初始策略输入点云。

已验证：9 项纯几何/关节/URDF 保护测试通过；指定 checkpoint 严格载入，真实 RGB-D
推理得到有限的 `15x7` 动作；第 50 条原始关节物理回放抓取确认、抬升约 6.3 cm、
覆盖 100%，但未落入托盘，故总任务仍失败；真实模型 300 步闭环运行完成，但没有稳定
抓取和擦除，覆盖 0。旧 beta 的连续原生 GUI 渲染仍有上文所述故障。
增加立即停止保护后的结果为 218 步、最大关节跟踪误差约 0.79 rad、未确认稳定抓取。
验证报告归档在 `outputs/duco_wipe_validation/20261021/`，其中 `policy_guarded.json`
是保护版本，`policy_300_steps.json` 是较早的完整步数诊断；两者都不是成功案例。

测试命令：

```bash
python -m unittest discover -s tests -p test_duco_robotwin.py -v
```

下一步应优先实测末端装配与场景尺寸/位置，并对齐真实与仿真的相机 RGB-D 分布。
本次只完整验证第 50 条，不代表 100 个场景或多随机种子的策略性能已经验证。
原 RGB-D 场景估计器对第 9 条初始托盘存在检测失败的已知限制，加载会明确报错。
目前不应将该环境里的成功率解释为预训练模型的实物能力，也不应通过偷偷替换成
脚本轨迹来提高“模型成功率”。
