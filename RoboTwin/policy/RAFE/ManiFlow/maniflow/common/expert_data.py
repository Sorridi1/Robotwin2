"""Resumable batches from ManiFlow's existing demonstration dataset.

Only loader order/augmentation RNG are checkpointed. Demonstrations stay in the
original Zarr format; neither policy labels nor online point clouds are cached.
"""

from contextlib import contextmanager
import copy
import math
from pathlib import Path
import random

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader


def capture_rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class ExpertDataIterator:
    """Deterministic DataLoader with exact epoch, batch and augmentation resume.

    num_workers=0 is intentional: prefetched worker RNG cannot be reconstructed
    from a batch offset. Dataset transforms run under their own saved RNG.
    """

    def __init__(self, cfg):
        guidance = cfg.rl.noise_guidance
        if int(guidance.get("num_workers", 0)) != 0:
            raise ValueError("RAFE exact expert-loader resume requires noise_guidance.num_workers=0")
        dataset_cfg = copy.deepcopy(cfg.task.dataset)
        override = guidance.get("dataset_path")
        if override not in (None, "", "null", "None"):
            dataset_cfg.zarr_path = str(Path(str(override)).expanduser().resolve())
        path = Path(str(dataset_cfg.zarr_path)).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"RAFE expert dataset missing: {path}. Set rl.noise_guidance.dataset_path "
                "to an existing ManiFlow RoboTwin Zarr store containing data/point_cloud, "
                "data/state, data/action and meta/episode_ends. NLL updates cannot be skipped."
            )
        self.batch_size = int(guidance.batch_size)
        if self.batch_size < 1:
            raise ValueError("noise_guidance.batch_size must be positive")
        self.seed = int(cfg.training.seed) + 9109
        dataset_cfg.seed = int(cfg.training.seed)
        outer = capture_rng_state()
        try:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            self.dataset = hydra.utils.instantiate(dataset_cfg)
            if len(self.dataset) == 0:
                raise ValueError(f"RAFE expert dataset contains no training windows: {path}")
            example = self.dataset[0]
            expected = (int(cfg.horizon), int(cfg.shape_meta.action.shape[0]))
            if tuple(example["action"].shape) != expected:
                raise ValueError(f"Expert action must be complete raw chunk {expected}; got {tuple(example['action'].shape)}")
            for key in ("point_cloud", "agent_pos"):
                actual = tuple(example["obs"][key].shape)
                shape = tuple(cfg.shape_meta.obs[key].shape)
                if actual[0] < int(cfg.n_obs_steps) or actual[1:] != shape:
                    raise ValueError(f"Expert {key} shape {actual} does not match obs window >= {cfg.n_obs_steps}, shape {shape}")
            self.rng = capture_rng_state()
        finally:
            restore_rng_state(outer)
        self.signature = {"path": str(path.resolve()), "size": len(self.dataset),
                          "batch_size": self.batch_size, "seed": self.seed,
                          "dataset": OmegaConf.to_container(dataset_cfg, resolve=True)}
        self.epoch = 0
        self.batch_offset = 0
        self._iterator = None

    @contextmanager
    def _isolated_rng(self):
        outer = capture_rng_state()
        restore_rng_state(self.rng)
        try:
            yield
        finally:
            self.rng = capture_rng_state()
            restore_rng_state(outer)

    def _new_iterator(self):
        permutation = torch.randperm(len(self.dataset), generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        batches = [permutation[i:i+self.batch_size] for i in range(0, len(permutation), self.batch_size)]
        loader = DataLoader(self.dataset, batch_sampler=batches[self.batch_offset:], num_workers=0,
                            generator=torch.Generator().manual_seed(self.seed + self.epoch + 1))
        self._iterator = iter(loader)

    def __next__(self):
        with self._isolated_rng():
            if self._iterator is None:
                self._new_iterator()
            try:
                batch = next(self._iterator)
            except StopIteration:
                self.epoch += 1
                self.batch_offset = 0
                self._new_iterator()
                batch = next(self._iterator)
            self.batch_offset += 1
        return batch

    def state_dict(self):
        return {"signature": self.signature, "epoch": self.epoch,
                "batch_offset": self.batch_offset, "rng": self.rng}

    def load_state_dict(self, state):
        if state["signature"] != self.signature:
            raise ValueError("Expert dataset/configuration differs from resume checkpoint")
        epoch, offset = int(state["epoch"]), int(state["batch_offset"])
        if epoch < 0 or not 0 <= offset <= math.ceil(len(self.dataset) / self.batch_size):
            raise ValueError("Invalid expert loader resume position")
        self.epoch, self.batch_offset = epoch, offset
        self.rng = state["rng"]
        self._iterator = None
