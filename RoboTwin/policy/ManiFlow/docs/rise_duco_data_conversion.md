# RISE/DUCO to ManiFlow data conversion

## 实施 Prompt

以下 Prompt 是第一版 DUCO 数据转换的冻结实施规范：

> 将指定目录中已经同步的 RISE 示范转换为 ManiFlow Zarr replay-buffer 格式。
> 不使用力/力矩数据，将设备定义为一台六轴 DUCO 机械臂和一个二值夹爪。
> 通过整数毫秒时间戳文件名严格对齐 RGB、Depth、Joint 和 Gripper，任何模态
> 缺帧都必须报错，禁止静默取交集。RGB 必须按 RGB 而非 OpenCV BGR 解码；
> 深度单位为毫米，使用 depth_scale=1000；相机 243222076209 使用指定内参。
> 生成 OpenCV 相机光学坐标系点云（x 向右、y 向下、z 向前），按声明的工作区
> 裁剪，以 5 mm voxel 下采样，再确定性采样为 1024 个 XYZRGB 点。
> state[t] 定义为 6 个弧度制关节角加 gripper/1000，其中 0=闭合、1=张开；
> action[t] 默认定义为 t+1 帧的同一 7 维量，同时允许显式调整 action offset。
> 必须写入 data/point_cloud、data/state、data/action、meta/episode_ends，另外
> 保存源时间戳和 episode ID 以便追溯。完整预处理契约必须写入 Zarr attrs。
> 转换完成后检查 shape、有限值、时间戳对齐和累计 episode 边界。除非显式指定，
> 不得覆盖已有输出；覆盖时保留带时间戳的备份。未来在线推理必须导入完全相同的
> 预处理实现，不能复制出第二套处理逻辑。

## Data contract

- Point cloud: `float32 [N, 1024, 6]`, XYZ in metres and RGB in `[0, 1]`.
- Point frame: OpenCV camera optical frame (`x right, y down, z forward`).
- State: `float32 [N, 7]`, `[q1..q6, gripper]`.
- Action: `float32 [N, 7]`, next-frame joint/gripper target by default.
- Episode boundary: `int64 [E]` cumulative `meta/episode_ends`.
- Force/torque: deliberately excluded in schema version `rise_duco_maniflow_v1`.

The camera frame is used in V1 because the checked-in DUCO camera extrinsic is
hard-coded and has not yet been validated against this dataset. During the
simulation phase, RoboTwin/SAPIEN point clouds must be converted into this same
camera optical frame.

## Source validation

This command does not need Open3D or Zarr:

```bash
python3 scripts/process_rise_duco_data.py \
  --input '/media/Elements1/users/ysj/rise_datasets/wipe-v1-one-pass-clean-whiteboard@100' \
  --scan-only
```

Missing metadata is reported as a warning because frame timestamps are carried
by every modality filename. Pass `--strict-metadata` if metadata and
`timestamp.txt` must be mandatory.

## Smoke conversion

Run in the ManiFlow environment:

```bash
conda run -n maniflow python scripts/process_rise_duco_data.py \
  --input '/media/Elements1/users/ysj/rise_datasets/wipe-v1-one-pass-clean-whiteboard@100' \
  --output /tmp/wipe-duco-smoke.zarr \
  --max-episodes 1 \
  --max-frames-per-episode 3
```

## Full conversion

Choose an explicit output location with sufficient free space:

```bash
conda run -n maniflow python scripts/process_rise_duco_data.py \
  --input '/media/Elements1/users/ysj/rise_datasets/wipe-v1-one-pass-clean-whiteboard@100' \
  --output '/path/to/wipe-duco-camera-pointcloud-100.zarr'
```

Use `--overwrite` only intentionally. The old output is renamed to a timestamped
backup rather than deleted.

## ManiFlow training configuration

Use the DUCO task config and override the dataset path:

```bash
cd RoboTwin/policy/ManiFlow/ManiFlow/maniflow/workspace
python train_maniflow_robotwin2_workspace.py \
  --config-name=maniflow_pointcloud_policy_robotwin2.yaml \
  task=duco_real_pointcloud \
  task.dataset.zarr_path=/absolute/path/wipe-duco-camera-pointcloud-100.zarr
```

Keep `n_obs_steps=2` and `horizon=16` for the first baseline. The source rate is
approximately 10 Hz, so the default 16-step horizon spans about 1.6 seconds.
