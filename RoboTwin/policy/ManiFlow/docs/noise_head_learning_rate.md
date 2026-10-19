# 噪声头独立学习率

现有训练中 actor 和噪声头共用 `3e-6`。`raw_rl_0928` 训练至第 99 轮时噪声均值比基础 schedule 增加约 3.17%，说明噪声头有更新，但幅度较小。这版优先加快噪声头优化，不强制增加噪声，也不修改 reference-flow 输入、残差公式、PPO clip、熵系数、梯度裁剪、warmup、环境或评估逻辑。它是待实际训练验证的设置，不保证提高成功率或解决偏向全局增益的问题。

```yaml
rl:
  ppo:
    actor_lr: 3.0e-6
    noise_lr: 3.0e-5
```

两组参数仍由同一个 AdamW 更新，保持原来的 AMP、梯度累积、联合 actor/noise 梯度裁剪和 KL 提前停止逻辑。冻结的观测编码器、plan predictor 和参考 actor 不加入优化器。critic-only warmup 或其他 actor 更新跳过条件仍同时跳过噪声头。

`noise_lr` 缺失或为 `null` 时使用 `actor_lr`，兼容归档配置。启动及恢复后会打印实际学习率，W&B 记录 `lr/actor`、`lr/noise` 和 `lr/critic`。

终端 `sigma` 改为本轮 rollout 实际采样的均值，显示六位小数，并显示同轮的 `|delta|`；warmup 不再错误显示零噪声。`loss/sigma_mean` 等已有 W&B 指标保持原口径，不新建 noise_diagnostics。

## 恢复旧训练

旧版 optimizer 只有一个参数组，顺序是可训练 base actor 参数、然后噪声头参数。恢复时按这个原顺序拆成两组，保留每个参数的 Adam 动量和更新计数，不丢弃 optimizer state。参数数量或动量形状不匹配会明确报错，不静默跳过。

恢复后 actor/noise 学习率使用**本次配置**，不会被 checkpoint 中旧学习率覆盖；其他 optimizer 状态保持。新 checkpoint 使用两参数组，若要回退到旧代码，旧代码不能直接读取新 optimizer 分组；部署/评估用的 actor 导出格式不变。

不建议覆盖已完成实验的输出目录。直接尝试时使用一个明确的新实验名；这不是自动添加时间戳子目录。

## 启动示例

在原训练环境、`RoboTwin/policy/ManiFlow` 目录执行：

```bash
bash scripts/train_rl_policy.sh \
  reinflow_rl_pointcloud_robotwin2 \
  place_empty_cup 1016 50 raw_ref_noise_lr1016 0 1 \
  /media/Elements1/ljj/ManiFlow/outputs/place_empty_cup-maniflow_pointcloud_policy_robotwin2-0924_seed0/checkpoints/latest.ckpt \
  demo_messy_ur5 \
  policy.action_plan.target_mode=identity \
  rl.actor.noise_conditioning_mode=reference_flow \
  rl.ppo.actor_lr=3e-6 rl.ppo.noise_lr=3e-5
```

只启动这一组即可，不要求对比实验。现有默认四进程环境并行和 cheap 场景检测不变。已经运行的进程不会热更新；代码修改不会启动、停止或重启训练。
