from typing import Dict, Tuple

import numpy as np
import torch


class RunningRewardScaler:
    def __init__(self, n_envs: int, gamma: float = 0.99, eps: float = 1e-8):
        self.n_envs = n_envs
        self.gamma = gamma
        self.eps = eps
        self.discounted_reward = np.zeros(n_envs, dtype=np.float32)
        self.count = eps
        self.mean = 0.0
        self.m2 = 1.0

    def __call__(self, rewards: np.ndarray, firsts: np.ndarray) -> np.ndarray:
        scaled = np.zeros_like(rewards, dtype=np.float32)
        for t in range(rewards.shape[0]):
            self.discounted_reward = self.discounted_reward * self.gamma * (1.0 - firsts[t]) + rewards[t]
            batch = self.discounted_reward
            batch_mean = float(np.mean(batch))
            batch_var = float(np.var(batch))
            batch_count = batch.shape[0]

            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.mean = self.mean + delta * batch_count / total
            self.m2 = (
                self.m2 * self.count
                + batch_var * batch_count
                + delta**2 * self.count * batch_count / total
            ) / total
            self.count = total
            scaled[t] = rewards[t] / (np.sqrt(self.m2) + self.eps)
        return scaled


class PPOBufferPointcloud:
    """CPU on-policy PPO buffer for ManiFlow point-cloud observations."""

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        n_obs_steps: int,
        point_cloud_shape: Tuple[int, int],
        agent_pos_shape: Tuple[int],
        denoising_steps: int,
        horizon_steps: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        reward_scale_running: bool = False,
        reward_scale_const: float = 1.0,
        bootstrap_on_truncated: bool = False,
        device: str = "cuda:0",
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.n_obs_steps = n_obs_steps
        self.point_cloud_shape = tuple(point_cloud_shape)
        self.agent_pos_shape = tuple(agent_pos_shape)
        self.denoising_steps = denoising_steps
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reward_scale_running = reward_scale_running
        self.reward_scale_const = reward_scale_const
        self.bootstrap_on_truncated = bootstrap_on_truncated
        self.device = device
        self.running_reward_scaler = (
            RunningRewardScaler(n_envs=n_envs, gamma=gamma) if reward_scale_running else None
        )
        self.reset()

    def reset(self):
        self.obs_trajs = {
            "point_cloud": np.zeros(
                (self.n_steps, self.n_envs, self.n_obs_steps, *self.point_cloud_shape),
                dtype=np.float32,
            ),
            "agent_pos": np.zeros(
                (self.n_steps, self.n_envs, self.n_obs_steps, *self.agent_pos_shape),
                dtype=np.float32,
            ),
        }
        self.chains_trajs = np.zeros(
            (
                self.n_steps,
                self.n_envs,
                self.denoising_steps + 1,
                self.horizon_steps,
                self.action_dim,
            ),
            dtype=np.float32,
        )
        self.reward_trajs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.raw_reward_trajs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.terminated_trajs = np.zeros((self.n_steps, self.n_envs), dtype=bool)
        self.truncated_trajs = np.zeros((self.n_steps, self.n_envs), dtype=bool)
        self.firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs), dtype=np.float32)
        self.success_trajs = np.zeros((self.n_steps, self.n_envs), dtype=bool)
        self.value_trajs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.logprobs_trajs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.advantages_trajs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.returns_trajs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.firsts_trajs[0] = 1.0

    def add(
        self,
        step: int,
        obs: Dict[str, np.ndarray],
        chains: np.ndarray,
        reward,
        terminated,
        truncated,
        value,
        logprob,
        success,
        raw_reward=None,
    ):
        reward = np.asarray(reward, dtype=np.float32).reshape(self.n_envs)
        raw_reward = reward if raw_reward is None else np.asarray(raw_reward, dtype=np.float32).reshape(self.n_envs)
        terminated = np.asarray(terminated, dtype=bool).reshape(self.n_envs)
        truncated = np.asarray(truncated, dtype=bool).reshape(self.n_envs)
        success = np.asarray(success, dtype=bool).reshape(self.n_envs)

        self.obs_trajs["point_cloud"][step] = obs["point_cloud"]
        self.obs_trajs["agent_pos"][step] = obs["agent_pos"]
        self.chains_trajs[step] = chains
        self.reward_trajs[step] = reward
        self.raw_reward_trajs[step] = raw_reward
        self.terminated_trajs[step] = terminated
        self.truncated_trajs[step] = truncated
        self.firsts_trajs[step + 1] = np.logical_or(terminated, truncated).astype(np.float32)
        self.success_trajs[step] = success
        self.value_trajs[step] = np.asarray(value, dtype=np.float32).reshape(self.n_envs)
        self.logprobs_trajs[step] = np.asarray(logprob, dtype=np.float32).reshape(self.n_envs)

    def normalize_reward(self):
        if self.running_reward_scaler is not None:
            self.reward_trajs = self.running_reward_scaler(
                self.reward_trajs,
                self.firsts_trajs[:-1],
            )

    @torch.no_grad()
    def update_adv_returns(self, last_obs: Dict[str, np.ndarray], critic: torch.nn.Module):
        critic_device = next(critic.parameters()).device
        last_obs_ts = {
            "point_cloud": torch.as_tensor(last_obs["point_cloud"], device=critic_device).float(),
            "agent_pos": torch.as_tensor(last_obs["agent_pos"], device=critic_device).float(),
        }
        next_value = critic(last_obs_ts).detach().cpu().numpy().reshape(1, self.n_envs)

        lastgaelam = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nextvalues = next_value[0]
            else:
                nextvalues = self.value_trajs[t + 1]

            if self.bootstrap_on_truncated:
                non_terminal = 1.0 - self.terminated_trajs[t].astype(np.float32)
            else:
                done = np.logical_or(self.terminated_trajs[t], self.truncated_trajs[t])
                non_terminal = 1.0 - done.astype(np.float32)

            delta = (
                self.reward_trajs[t] * self.reward_scale_const
                + self.gamma * nextvalues * non_terminal
                - self.value_trajs[t]
            )
            lastgaelam = delta + self.gamma * self.gae_lambda * non_terminal * lastgaelam
            self.advantages_trajs[t] = lastgaelam
        self.returns_trajs = self.advantages_trajs + self.value_trajs

    @torch.no_grad()
    def update(self, last_obs: Dict[str, np.ndarray], critic: torch.nn.Module):
        self.normalize_reward()
        self.update_adv_returns(last_obs, critic)

    def make_dataset(self):
        obs = {
            "point_cloud": torch.as_tensor(self.obs_trajs["point_cloud"], device=self.device).float().flatten(0, 1),
            "agent_pos": torch.as_tensor(self.obs_trajs["agent_pos"], device=self.device).float().flatten(0, 1),
        }
        chains = torch.as_tensor(self.chains_trajs, device=self.device).float().flatten(0, 1)
        returns = torch.as_tensor(self.returns_trajs, device=self.device).float().flatten(0, 1)
        values = torch.as_tensor(self.value_trajs, device=self.device).float().flatten(0, 1)
        advantages = torch.as_tensor(self.advantages_trajs, device=self.device).float().flatten(0, 1)
        logprobs = torch.as_tensor(self.logprobs_trajs, device=self.device).float().flatten(0, 1)
        return obs, chains, returns, values, advantages, logprobs

    @torch.no_grad()
    def get_explained_var(self, values: torch.Tensor, returns: torch.Tensor) -> float:
        y_pred = values.detach().cpu().numpy()
        y_true = returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        return float("nan") if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)

    @torch.no_grad()
    def summarize_episode_reward(self):
        episodes = []
        for env_ind in range(self.n_envs):
            firsts = np.where(self.firsts_trajs[:, env_ind] == 1)[0]
            for i in range(len(firsts) - 1):
                start = firsts[i]
                end = firsts[i + 1] - 1
                if end >= start:
                    episodes.append((env_ind, start, end))

        if len(episodes) == 0:
            self.num_episode_finished = 0
            self.avg_episode_reward = 0.0
            self.std_episode_reward = 0.0
            self.avg_raw_episode_reward = 0.0
            self.success_rate = 0.0
            self.std_success_rate = 0.0
            self.avg_episode_length = 0.0
            return {}

        ep_rewards = []
        ep_raw_rewards = []
        ep_success = []
        ep_lengths = []
        for env_ind, start, end in episodes:
            ep_rewards.append(np.sum(self.reward_trajs[start : end + 1, env_ind]))
            ep_raw_rewards.append(np.sum(self.raw_reward_trajs[start : end + 1, env_ind]))
            ep_success.append(np.any(self.success_trajs[start : end + 1, env_ind]))
            ep_lengths.append(end - start + 1)

        ep_rewards = np.asarray(ep_rewards, dtype=np.float32)
        ep_raw_rewards = np.asarray(ep_raw_rewards, dtype=np.float32)
        ep_success = np.asarray(ep_success, dtype=np.float32)
        ep_lengths = np.asarray(ep_lengths, dtype=np.float32)

        self.num_episode_finished = len(episodes)
        self.avg_episode_reward = float(np.mean(ep_rewards))
        self.std_episode_reward = float(np.std(ep_rewards))
        self.avg_raw_episode_reward = float(np.mean(ep_raw_rewards))
        self.success_rate = float(np.mean(ep_success))
        self.std_success_rate = float(np.std(ep_success))
        self.avg_episode_length = float(np.mean(ep_lengths))
        return {
            "num_episode_finished": self.num_episode_finished,
            "avg_episode_reward": self.avg_episode_reward,
            "std_episode_reward": self.std_episode_reward,
            "avg_raw_episode_reward": self.avg_raw_episode_reward,
            "success_rate": self.success_rate,
            "std_success_rate": self.std_success_rate,
            "avg_episode_length": self.avg_episode_length,
            "rollout_reward_sum": float(np.sum(self.reward_trajs)),
            "rollout_raw_reward_sum": float(np.sum(self.raw_reward_trajs)),
            "rollout_success_any": float(np.any(self.success_trajs)),
        }
