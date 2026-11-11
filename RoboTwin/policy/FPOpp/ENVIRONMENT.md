# 现有环境与路径

直接在已有 `maniflow` Conda 环境运行，不需要创建新环境，也不需要安装官方 robosuite、DexMimicGen、LeRobot 或其 requirements。新增优化目标只用本环境已有的 PyTorch、Hydra 等依赖。

## 路径核对

当前工作区的准确路径是：

```text
/home/ljj/code/Robotwin2/RoboTwin/policy/ManiFlow/ManiFlow
```

它有大写 R，且没有 `.0`。2026-11-03 实测 `/home/ljj/code/robotwin2.0` 和 `/home/ljj/code/Robotwin2.0` 均不存在。旧 editable 安装的 direct_url.json 确实记录小写 `robotwin2.0`，不是将当前路径手工转小写后的推测。

FPO++ 的 `train.py`、`eval.sh` 和部署接口按自身文件位置寻找当前 ManiFlow 源码，并只在当前进程/子进程的 Python 搜索路径中优先使用它。**没有修改现有 editable 安装、创建软链接、重命名文件夹或安装包。** 旧安装路径不会阻止本目录入口运行。

## 已有版本与检查

解释器：`/opt/shared/Prog/anaconda3/envs/maniflow/bin/python`，Python 3.10.19。

| 包 | 安装版本 |
|---|---|
| torch / torchvision | 2.4.1 / 0.19.1 |
| numpy | 1.26.4 |
| sapien / mplib | 3.0.0b1 / 0.2.1 |
| pytorch3d | 0.7.8 |
| nvidia-curobo | 0.8.0.post1.dev2+dirty |
| hydra-core / omegaconf | 1.2.0 / 2.3.0 |
| gymnasium | 0.29.1 |
| diffusers / timm | 0.27.2 / 1.0.22 |
| zarr | 2.12.0 |

```bash
cd /home/ljj/code/Robotwin2
conda activate maniflow
python RoboTwin/policy/FPOpp/scripts/check_environment.py --local-source --imports
CUDA_VISIBLE_DEVICES=0 python RoboTwin/policy/FPOpp/scripts/check_environment.py --local-source --cuda
```

不加 `--local-source` 时检查原安装来源，会如实报告旧 editable 路径失效。加该选项后，检查方式与 FPO++ 入口一致。包元数据/导入检查不等于完成真实仿真。

默认当前 shell 可能仍是 Python 3.12 的 Conda base，应先激活 maniflow，或对 shell 入口设置 `PYTHON_BIN=/opt/shared/Prog/anaconda3/envs/maniflow/bin/python`。

FPO++ 入口继承 GPU 可见性，并为 Vulkan 和 CuRobo 设置与本工作相同的默认值：

```text
VK_ICD_FILENAMES=当前仓库/RoboTwin/policy/ManiFlow/scripts/nvidia_icd.json
CUROBO_TORCH_COMPILE=0
CUROBO_TORCH_COMPILE_DISABLE=1
TOKENIZERS_PARALLELISM=false
```

沙箱内可能看不到 CUDA，不能仅据此判断主机驱动损坏；本次在可访问 GPU 的进程中已核实主机 A6000 可用，并完成真实 checkpoint 的 GPU 更新。

真实场景 smoke 中，FP16 在初始 GradScaler 的缩放下出现梯度溢出；默认改为 BF16 后，已完成两次 chunk 的采集、30个动作和一次完整更新/保存。BF16 不需要 GradScaler；可选 FP16 保留动态缩放和溢出跳步日志，FP32/BF16 的非有限梯度会明确报错。

## RoboTwin 仿真环境

继续复用同一个 `RoboTwinRLEnv`，不修改任务和成功条件：

- 相同机器人、点云/关节观测、qpos 动作、normalizer 和历史窗口。
- 相同 reward_mode/系数、场景 cheap validator、渲染设置和任务步数上限。
- 新增 FPO++ actor/loss/buffer/workspace；不改旧 PPO actor、噪声模型或公共 pool。
- 新增 `FPOpp.get_model/eval/reset_model`，接入原 `script/eval_policy.py`。
- 默认不做超时 bootstrap，符合现有 pool 的自动 reset 返回语义；不支持通过简单开关给 reset observation 估计 terminal value。

切换任务或机器人时，传入对应 checkpoint、task_name、task_config，并确认实际 shape_meta、资产和奖励来源相符。当前 `demo_messy_ur5` 是双 ur5-wsg，随机化开关实际关闭，不按配置名称推测难度。

## 比较时的固定项

使用与本工作相同的 BC checkpoint、model/EMA 起点和推理协议。默认严格保留 checkpoint 的 plan 架构，不要求额外训练无 plan 模型。plan predictor 与 encoder 冻结，仅使用 FPO++ 更新 flow 网络。

`H=16/To=2/n_action_steps=16` 实际最多15步，双方一致。按真实执行动作数对齐预算。若要对齐现有 PPO 训练裁剪，显式设置 `actor.intermediate_clip=1.5 actor.final_action_clip=1.0`，不改变公共代码。

双方共享同一冻结 test bank；独立的 validation bank 用于 best 选择。旧公共池可能与训练 seed 重叠，且已有组不会因修改 eval_seed 换种子。程序保存 `training_scene_seeds.json` 和逐场景评估报告，最终汇总时核对各方法实际 train/validation/test 交集。

环境快照可以保存到被忽略的生成目录：

```bash
mkdir -p RoboTwin/policy/FPOpp/output/environment
conda env export > RoboTwin/policy/FPOpp/output/environment/conda.yaml
python -m pip freeze > RoboTwin/policy/FPOpp/output/environment/pip-freeze.txt
python RoboTwin/policy/FPOpp/scripts/check_environment.py --local-source \
  > RoboTwin/policy/FPOpp/output/environment/preflight.json
```
