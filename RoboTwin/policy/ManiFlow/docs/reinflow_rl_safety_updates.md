# ReinFlow RL Safety Updates

Date: 2026-05-13

## Background

The first RoboTwin2 ManiFlow PPO fine-tuning run showed a clear degradation pattern:
the pretrained ManiFlow point-cloud actor had a high success rate, but later PPO
iterations became mostly failed rollouts. Because failed rollouts were still used
for actor updates, the final `latest.ckpt` could be worse than earlier actors.

This update makes the RL stage more conservative and changes deployment export to
prefer the best observed actor instead of the last actor.

## Changed Files

- `ManiFlow/maniflow/workspace/train_reinflow_rl_robotwin2_workspace.py`
- `ManiFlow/maniflow/config/reinflow_rl_pointcloud_robotwin2.yaml`
- `docs/reinflow_rl_safety_updates.md`

## Checkpoint Logic

The RL workspace now tracks a best checkpoint metric:

```yaml
rl:
  best_checkpoint:
    enabled: true
    metric: rollout/success_rate
    mode: max
    tag: best_success
    update_latest_alias: true
    min_delta: 1.0e-6
```

When the metric improves, the workspace exports:

- `checkpoints/best_success.ckpt`
- `checkpoints/latest.ckpt`

`latest.ckpt` is now a deploy-compatible alias of the best actor, not necessarily
the final training iteration. This keeps the existing RoboTwin2 evaluation path
compatible while avoiding accidental evaluation of a collapsed last actor.

The best actor is exported before the PPO update for that iteration, so the saved
weights correspond to the actor that actually produced the rollout metric.

The RL resume checkpoint remains:

- `checkpoints/latest_rl.ckpt`

Use `latest_rl.ckpt` only for continuing RL training. Use `latest.ckpt` or
`best_success.ckpt` for deployment/evaluation.

`rl.resume` and WandB `logging.resume` now default to `false` so a new run starts
from the pretrained actor instead of accidentally resuming a previous collapsed
RL state. To continue an interrupted RL run, override `rl.resume=true` or set
`rl.resume_path`.

## Failed-Rollout Actor Update Protection

The workspace now decides whether to update the actor after each rollout:

- Critic warmup iterations update critic only.
- If `rollout/success_rate == 0`, actor update is skipped.
- If `rollout/rollout_raw_reward_sum == 0`, actor update is skipped.
- The critic still updates, so value learning can continue.

Relevant config:

```yaml
rl:
  ppo:
    skip_actor_update_on_zero_success: true
    skip_actor_update_on_zero_reward: true
    min_success_rate_for_actor_update: 0.0
    min_raw_reward_for_actor_update: 0.0
```

New logs include:

- `rl/actor_update_enabled`
- `rl/actor_update_skip_code`
- `rl/consecutive_zero_success_iters`
- `rl/saved_best_actor_checkpoint`
- `rl/best_checkpoint_metric`
- `loss/actor_update_enabled`

Skip codes:

- `0`: actor update enabled
- `1`: critic warmup
- `2`: zero-success rollout
- `3`: zero-reward rollout

## Conservative RL Defaults

The default RL settings were made more conservative to reduce drift from the
pretrained actor:

```yaml
rl:
  actor:
    min_sampling_denoising_std: 0.001
    min_logprob_denoising_std: 0.005
    max_logprob_denoising_std: 0.02
    actor_old_device: cuda:0

  ppo:
    actor_lr: 1.0e-6
    reward_scale_running: false

  bc_anchor:
    enabled: true
    coeff: 0.05
```

Rationale:

- Lower actor learning rate reduces destructive PPO updates.
- Smaller exploration noise reduces failed rollouts from a strong pretrained actor.
- Stronger BC anchor keeps the fine-tuned actor close to the pretrained policy.
- Disabling running reward scaling avoids unstable scaling in sparse-reward runs.
- Keeping `actor_old` on GPU makes BC anchor faster. If GPU memory becomes tight,
  override `rl.actor.actor_old_device=cpu`.

## Evaluation

Existing evaluation scripts can still load `latest.ckpt` because it remains a
plain ManiFlow actor checkpoint with:

- `payload["state_dicts"]["model"]`
- `payload["state_dicts"]["ema_model"]`

For example:

```bash
cd /home/ljj/code/Robotwin2/RoboTwin

python script/eval_policy.py --config policy/ManiFlow/deploy_policy.yml \
  --overrides \
  --config_name reinflow_rl_pointcloud_robotwin2 \
  --task_name place_empty_cup \
  --task_config demo_messy_ur5 \
  --ckpt_setting demo_messy_ur5 \
  --expert_data_num 50 \
  --training_seed 0 \
  --seed 0 \
  --policy_name ManiFlow \
  --addition_info rl_ft \
  --alg_name reinflow_rl_pointcloud_robotwin2
```

## Remaining Risks

- A zero-success first rollout can still be exported as the initial best if no
  better rollout is observed later. This guarantees a deploy checkpoint exists,
  but it does not guarantee improvement over the pretrained actor.
- The best metric is based on noisy training rollouts, not a full deterministic
  100-episode evaluation.
- Sparse success reward still has high variance. If a task implements
  `stage_reward()`, `rl.reward_mode=stage_delta` may provide better learning
  signal.
