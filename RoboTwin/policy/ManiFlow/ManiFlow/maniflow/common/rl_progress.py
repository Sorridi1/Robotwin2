"""One parent-owned terminal progress bar per online RL iteration."""

import shutil
import sys

from tqdm import tqdm


class _ProgressBar(tqdm):
    @property
    def format_dict(self):
        values = super().format_dict
        # tqdm prefixes its built-in {postfix} with a comma for the default
        # layout; our compact layout supplies its own spacing instead.
        values["metrics"] = self.postfix or ""
        return values


class RLIterationProgress:
    bar_format = "{desc} {n_fmt}/{total_fmt} |{bar:8}| {metrics}"

    def __init__(self, iteration, rollout_steps, global_step=0, mininterval=1.0, stream=None):
        self.iteration = iteration
        self.rollout_steps = rollout_steps
        self.global_step = global_step
        self.successes = 0
        self.episodes = 0
        self.last_action_steps = 0
        self.env_steps = []
        self.reward = None
        self.optimizer_steps = 0
        self.total_optimizer_steps = 0
        self.phase = "reset"
        self.bar = _ProgressBar(
            total=rollout_steps,
            desc=f"itr={iteration} reset",
            postfix=self._rollout_postfix(),
            file=sys.stdout if stream is None else stream,
            leave=True,
            dynamic_ncols=True,
            mininterval=mininterval,
            miniters=1,
            # A long description/postfix must not squeeze the actual bar away.
            # Keep succ and action counters ahead of optional timing information.
            bar_format=self.bar_format,
        )

    def _counts(self):
        return f"succ={self.successes}/{self.episodes} step={self.global_step}"

    def _rollout_postfix(self):
        text = f"{self._counts()}(+{self.last_action_steps})"
        if self.env_steps:
            low, high = min(self.env_steps), max(self.env_steps)
            text += f" env_step={low}" if low == high else f" env_step={low}..{high}"
        return text

    def _refresh(self, completed, postfix):
        # Explicit refresh avoids tqdm's adaptive miniters skipping slow simulator
        # batches, and redraws the line after worker initialization/reset output.
        columns = shutil.get_terminal_size().columns
        prefix = f"itr={self.iteration} {self.phase} {completed}/{self.bar.total} |{' ' * 8}| "
        if self.phase == "rollout" and self.reward is not None:
            reward = f" reward={self.reward:.2f}"
            if len(prefix + postfix + reward) <= columns:
                postfix += reward
        self.bar.bar_format = self.bar_format
        if len(prefix + postfix) + 18 <= columns:
            self.bar.bar_format += " [{elapsed}<{remaining}]"
        self.bar.n = completed
        self.bar.set_description_str(f"itr={self.iteration} {self.phase}", refresh=False)
        self.bar.set_postfix_str(postfix, refresh=False)
        self.bar.refresh()

    def start_rollout(self):
        self.phase = "rollout"
        self._refresh(0, self._rollout_postfix())

    def advance_rollout(self, completed, successes, terminateds, truncateds,
                        executed_steps, global_step, env_steps, reward=None):
        for success, terminated, truncated in zip(successes, terminateds, truncateds):
            if terminated or truncated:
                self.episodes += 1
                self.successes += int(bool(success))
        self.last_action_steps = executed_steps
        self.global_step = global_step
        self.env_steps = list(env_steps)
        self.reward = reward
        self._refresh(completed, self._rollout_postfix())

    def start_update(self, minibatches, optimizer_steps):
        self.phase = "update"
        self.total_optimizer_steps = optimizer_steps
        self.optimizer_steps = 0
        self.bar.set_description_str(f"itr={self.iteration} update", refresh=False)
        self.bar.set_postfix_str(f"{self._counts()} opt=0/{optimizer_steps}", refresh=False)
        self.bar.reset(total=minibatches)
        self.advance_update(0, 0)

    def advance_update(self, completed, optimizer_steps, stopped_by_kl=False):
        self.optimizer_steps = optimizer_steps
        self.phase = "update(KL-stop)" if stopped_by_kl else "update"
        self._refresh(
            completed,
            f"{self._counts()} opt={optimizer_steps}/{self.total_optimizer_steps}",
        )

    def close(self):
        self.bar.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
