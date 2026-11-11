"""Small real ManiFlow model; fake simulator only for CPU integration checks."""

import copy
import numpy as np
from omegaconf import OmegaConf
import torch


def tiny_checkpoint(plan=True):
    from maniflow.policy.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy
    shape = {"obs": {"point_cloud": {"shape": [8, 6], "type": "point_cloud"},
                     "agent_pos": {"shape": [2], "type": "low_dim"}}, "action": {"shape": [2]}}
    config = {
        "_target_": "maniflow.policy.maniflow_pointcloud_policy.ManiFlowTransformerPointcloudPolicy",
        "shape_meta": shape, "horizon": 4, "n_obs_steps": 2, "n_action_steps": 3,
        "num_inference_steps": 2, "n_layer": 1, "n_head": 2, "n_emb": 16,
        "visual_cond_len": 8, "encoder_output_dim": 8,
        "diffusion_timestep_embed_dim": 8, "diffusion_target_t_embed_dim": 8,
        "language_conditioned": False, "sample_target_t_mode": "relative", "use_pc_color": True,
        "pointnet_type": "pointnet", "downsample_points": True,
        "pointcloud_encoder_cfg": {"in_channels": 6, "out_channels": 8, "use_layernorm": True,
                                  "final_norm": "layernorm", "num_points": 8, "pointwise": True},
        "action_plan": {"enabled": plan, "target_mode": "identity", "predictor_hidden_dims": [8]},
    }
    kwargs = OmegaConf.create(copy.deepcopy(config))
    del kwargs._target_
    model = ManiFlowTransformerPointcloudPolicy(**kwargs)
    generator = torch.Generator().manual_seed(99)
    model.normalizer.fit({"action": torch.randn(12, 4, 2, generator=generator),
                          "agent_pos": torch.randn(12, 2, generator=generator),
                          "point_cloud": torch.randn(12, 8, 6, generator=generator)})
    return {"cfg": OmegaConf.create({"policy": config}),
            "state_dicts": {"ema_model": copy.deepcopy(model.state_dict()), "model": copy.deepcopy(model.state_dict())}}


class FakePool:
    """Auto-reset semantics match the shared pool; no claimed physics validation."""
    def __init__(self, count=2):
        self.count, self.closed = count, False
        self.observations = [{"agent_pos": np.zeros((2, 2), dtype=np.float32),
                              "point_cloud": np.ones((2, 8, 6), dtype=np.float32)*0.1} for _ in range(count)]
        self.cursor = 700000
        self.last_reset_infos = []

    def reset(self):
        self.cursor += self.count
        self.last_reset_infos = [{"validation": {"accepted_seed": self.cursor+i}} for i in range(self.count)]
        return self.observations

    def step(self, actions):
        result = []
        for index, action in enumerate(actions):
            steps = 1 if index == 0 else len(action)
            done = index == 0
            # Mutate shared storage to check that the collector owns its old obs.
            self.observations[index]["agent_pos"][:] += 0.1
            result.append((self.observations[index], float(index+1), done, False,
                           {"executed_action_steps": steps, "success": done,
                            "validation": {"accepted_seed": self.cursor+index+100} if done else {}}))
        return result

    def get_next_seeds(self):
        return [self.cursor+i+200 for i in range(self.count)]

    def close(self):
        self.closed = True
