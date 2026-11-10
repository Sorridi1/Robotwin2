# 0428 微调参数复核与建议

当前默认值可作为保守起点，但不是经过性能调优的 FPO++ 最优值。新增 [fpopp_0428_recommended.yaml](configs/fpopp_0428_recommended.yaml) 作为明确的推荐起始配置；原配置、原方法、已有实验和结果均未修改。以下推荐来自官方操作实验与本地实际训练设置的比较，尚未通过完整训练证明性能更好。

**2026-11-04 复核更新：**该起点已出现第 20 轮验证 63% 及明显的离线示范行为漂移，不能继续视作已验证合理的参数组合。本地 actor AdamW 默认 eps=1e-8，官方为1e-5；实际优化器状态显示此差异会明显影响更新幅度。进一步诊断及单因素对照建议见 [审计记录](AUDIT_20261104.md)。本次没有改动配置文件数值或运行中的训练。

## 对照依据

官方以最终运行配置为准：

- `/home/ljj/code/fpo-control/manipulation_experiments/docs/reproduce.md` 的 main benchmark。
- `scripts/run_main_benchmark.sh` 的 SHARED_ARGS 和 FPOP_PARAMS。
- `finetune_online_rl.py` 的 FlowPPOConfig 以及未被启动参数覆盖的默认值。
- [论文 B.3](https://arxiv.org/html/2602.02481v2#A2.SS3)：最终操作实验 actor LR=1e-5、critic LR=1e-4、GAE lambda=.99。

不能将 sweeps 中的候选值误认为最后采用值。例如 Can 的 sweep 包含 clip=.01/.02，最终主实验用.02；Square 最终关闭两种 cap，不能宣称所有任务都固定 cap=4/5。

本地参考 `residual_only_1027_seed0` 和 `raw_reinflow_xt_tcn_1028_seed0`，从 `.hydra/config.yaml` 和 `checkpoints/latest.ckpt` 内 cfg 交叉确认。初次推荐时仅有小规模 smoke；后续真实训练证据以上述复核更新和审计记录为准。

## 主要参数

| 参数 | 官方最终操作实验/未覆盖默认 | 我们实际训练 | 当前 FPOpp 默认 | 推荐0428起点 |
|---|---|---|---|---|
| actor LR | 1e-5 | 3e-6 | 3e-6 | **1e-5** |
| critic LR | 1e-4 | 5e-4 | 5e-4 | **5e-4** |
| update epochs | 10 | 2 | 2 | **4** |
| 微批次 × 梯度累积 | 单GPU时约375×1个chunk | 4×4=16 | 4×4=16 | **4×8=32** |
| 每轮名义环境chunk数 | 30×1600/16=3000 | 4×64=256 | 4×64=256 | **4×64=256** |
| PPO clip coefficient | .01/.02/.03，随任务 | .005，高斯链ratio | .01，CFM ratio | **.02，CFM ratio** |
| MC samples | 8 | 不适用 | 8 | **8** |
| MC 分片大小 | 不等同于环境batch | 不适用 | 2 | **2** |
| Huber delta | .1/.5/1，随任务 | 不适用 | .5 | **.5** |
| old loss cap / logratio cap | 多数4/5；Square None/None | 不同log-prob裁剪 | 4/5 | **4/5** |
| max grad norm | 1/5/25，随任务 | 1 | 1 | **1** |
| gamma | .99/.995/.999，按动作GAE | .99，按chunk | .99，按chunk | **.99，按chunk** |
| GAE lambda | .99 | .95 | .95 | **.95** |
| critic-only warmup | 代码条件保留初始value-only阶段 | 4轮 | 4轮 | **4轮** |
| value loss coefficient | 1 | .5 | .5 | **.5** |
| ODE steps | 10 | 10 | checkpoint决定 | **10，来自0428** |
| 中间/最终动作裁剪 | 随模型设置 | 1.5/1.0 | None/None | **1.5/1.0** |
| 训练预算 | 原始任务500万或800万动作 | 100轮 | 默认100轮或10万动作 | **100轮或40万动作** |

官方FPO++五个任务的最终关键值：

| 参数 | Can | Square | Box | Tray | Threading |
|---|---|---|---|---|---|
| clip | .02 | .01 | .03 | .03 | .01 |
| max grad norm | 5 | 25 | 5 | 1 | 1 |
| Huber delta | .5 | 1 | .1 | 1 | .1 |
| old cap / logratio cap | 4/5 | None/None | 4/5 | 4/5 | 4/5 |

推荐用 Can 的 clip=.02、Huber=.5、cap=4/5 作为杯子放置任务的初始候选，这是迁移判断，不是官方对 RoboTwin2 的结论。保留 max grad norm=1，属于官方实际用过的设置，也与当前大模型训练保持一致；必要时可单独把5纳入验证。

## 为什么这样调整

1. **提高actor学习率，但不直接照抄10个epoch。** 1e-5是论文最终操作实验的actor LR；原3e-6更保守，未经过实验证明足够。我们的每轮只有256个chunk，直接复用10遍可能更易对这批数据拟合，因此先用4遍。
2. **有效batch从16改为32，显存峰值仍由微批次4和MC分片2控制。** 完整256个有效chunk时，原配置每轮为 `256/16×2=32` 次优化器更新，推荐配置为 `256/32×4=32` 次。这样保持名义更新次数，增加每次更新的环境样本数和数据复用；不宣称与官方375个chunk的batch完全等价。
3. **critic不照搬官方的小MLP设置。** 官方critic对actor视觉特征接MLP；这里是本工作独立的点云critic。先保留已使用的5e-4和4轮warmup，避免同时改变整个critic训练方案。1e-4可作为后续独立候选，不能仅因论文使用它就认定更适合本地critic。
4. **gamma/lambda保持共同时间口径。** 官方先逐动作计算GAE再聚合chunk，本实现与本工作以chunk为transition。直接复制.999/.99会改变有效时间尺度；主对比保留.99/.95。
5. **保持共同环境设置。** ant_style奖励、progress scale=1、success bonus=1.5、delta control cost=.001、奖励运行缩放、cheap validator、qpos控制与本工作一致。FPO++不增加学习噪声、reference residual、BC anchor、成功率门控。

Adam betas当前仍为本工作使用的(.9,.95)，actor weight decay=0、critic weight decay=1e-5；官方操作actor使用(.9,.99)、eps=1e-5、weight decay=1e-6。推荐配置有意沿用本地优化器实现，不能称为原版官方参数的逐项复刻。

注意MC的8次采样不是8倍独立环境数据：有效环境batch仍为32个chunk，每个chunk共享同一个advantage。PPO clip作用于CFM surrogate ratio，不能与本工作高斯链的.005直接按倍数比较。当前FPOpp没有精确KL早停接口，不应添加 `target_kl=.02` 或将ratio统计称为精确KL。

## 运行与验证选择

```bash
cd /home/ljj/code/Robotwin2
conda activate maniflow

CUDA_VISIBLE_DEVICES=0 bash RoboTwin/policy/FPOpp/train.sh \
  --config RoboTwin/policy/FPOpp/configs/fpopp_0428_recommended.yaml \
  --output-dir /media/Elements1/ljj/ManiFlow/outputs/place_empty_cup-fpopp-0428_recommended_seed0
```

配置已经指定0428第500轮ema_model、任务、seed=0、clip与评估参数。输出目录与此前建议的run不同；如目录已有checkpoint，仍要求resume或换目录。只检查配置可在命令末尾加 `--dry-run`，不会启动训练。

主参数候选建议控制为四组：`actor_lr ∈ {3e-6, 1e-5}` × `clip_coef ∈ {.01,.02}`，其余使用推荐配置。每组给相同预算、相同validation池；按validation选定后再跑seed=0/1/2与最终测试，不用test池调参。资源有限时可先等预算跑20轮筛选，不能把该小预算筛选声称为完整调参结果。

当前已有loss/ratio/clip_fraction、实际动作计数、溢出计数和验证成功率日志。观察跨epoch ratio与clip_fraction是否快速饱和，并同时看独立验证成功率；初始ratio≈1、actor loss≈0本身不代表未学习。若验证持续退化，先验证较小LR/clip候选，不按训练reward单独挑模型。

**初始化尚有一个实际差异：**已核对的10/27、10/28本工作训练checkpoint内cfg都指向0924 BC，0428不是这些旧实验的相同起点。按用户要求本推荐继续使用0428；若要严格比较微调方法，需把本工作对应分支也从相同0428权重重新训练，或明确报告两种初始化及各自BC成绩。
