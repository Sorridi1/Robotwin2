"""Exercise checkpoint interruption/resume using the real CPU training loop."""

import copy
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from maniflow.dataset.base_dataset import BaseDataset
from maniflow.workspace import train_maniflow_robotwin2_workspace as module


class TinyDataset(BaseDataset):
    zarr_path = 'test-only'
    train_episodes_num = 1
    val_episodes_num = 1

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {
            'obs': {'agent_pos': torch.tensor([[1.0 + index]])},
            'action': torch.tensor([[0.3]]),
        }

    def get_validation_dataset(self):
        return self

    def get_normalizer(self):
        return None


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))

    def set_normalizer(self, normalizer):
        pass

    def compute_loss(self, batch, ema_model):
        loss = ((self.weight * batch['obs']['agent_pos'] - batch['action']) ** 2).mean()
        return loss, {'loss_flow': loss.item()}

    def predict_action(self, obs):
        return {'action_pred': self.weight * obs['agent_pos']}


class InterruptedTraining(Exception):
    pass


@pytest.fixture
def make_workspace(monkeypatch, tmp_path):
    original_instantiate = module.hydra.utils.instantiate

    def instantiate(cfg, **kwargs):
        if cfg.get('_target_') == 'test.TinyDataset':
            return TinyDataset()
        return original_instantiate(cfg, **kwargs)

    monkeypatch.setattr(module.hydra.utils, 'instantiate', instantiate)
    # No simulator, GPU, W&B account, or external service is used by these tests.
    tracker = SimpleNamespace(config={}, active=None)

    def init(**kwargs):
        return SimpleNamespace(log=lambda data, step: tracker.active.logs.append((step, dict(data))))

    tracker.init = init
    monkeypatch.setattr(module, 'wandb', tracker)

    def make(name, resume=False, use_ema=True):
        cfg = OmegaConf.create({
            'task': {'name': 'test', 'dataset': {'_target_': 'test.TinyDataset'}},
            'training': {
                'debug': False, 'resume': resume, 'num_epochs': 4,
                'device': 'cpu', 'use_ema': use_ema,
                'gradient_accumulate_every': 1,
                'lr_scheduler': 'cosine', 'lr_warmup_steps': 0,
                'checkpoint_every': 1, 'val_every': 1,
                'sample_every': 1, 'rollout_every': 1000,
                'max_train_steps': None, 'max_val_steps': None,
                'tqdm_interval_sec': 1000,
            },
            'dataloader': {'batch_size': 2, 'shuffle': False, 'num_workers': 0},
            'val_dataloader': {'batch_size': 2, 'shuffle': False, 'num_workers': 0},
            'ema': {
                '_target_': 'maniflow.model.diffusion.ema_model.EMAModel',
                'power': 0.75,
            },
            'logging': {'name': 'test', 'group': 'test'},
            'checkpoint': {
                'save_ckpt': True, 'save_last_ckpt': True,
                'save_last_snapshot': False,
                'topk': {'monitor_key': 'val_loss', 'mode': 'min', 'k': 0},
            },
        })
        workspace = module.TrainManiFlowRoboTwinWorkspace.__new__(module.TrainManiFlowRoboTwinWorkspace)
        workspace.cfg = cfg
        workspace._output_dir = str(tmp_path / name)
        (tmp_path / name).mkdir(exist_ok=True)
        workspace._saving_thread = None
        workspace.model = TinyPolicy()
        workspace.ema_model = copy.deepcopy(workspace.model) if use_ema else None
        workspace.optimizer = torch.optim.AdamW(workspace.model.parameters(), lr=0.01)
        workspace.epoch = 0
        workspace.global_step = 0
        workspace.logs = []
        tracker.active = workspace
        return workspace

    return make


@pytest.mark.parametrize('use_ema', [True, False])
def test_resume_matches_uninterrupted_weights_optimizer_and_schedule(make_workspace, use_ema):
    full = make_workspace('full', use_ema=use_ema)
    full.run()

    partial = make_workspace('split', use_ema=use_ema)
    save = partial.save_checkpoint

    def save_then_interrupt(*args, **kwargs):
        path = save(*args, **kwargs)
        if partial.epoch == 1:
            raise InterruptedTraining()
        return path

    partial.save_checkpoint = save_then_interrupt
    with pytest.raises(InterruptedTraining):
        partial.run()

    resumed = make_workspace('split', resume=True, use_ema=use_ema)
    with torch.no_grad():
        resumed.model.weight.fill_(99)  # Must actually restore saved weights.
    resumed.run()

    assert resumed.epoch == full.epoch == 4
    assert resumed.global_step == full.global_step == 8
    assert {row['epoch'] for _, row in resumed.logs} == {2, 3}
    assert [step for step, _ in resumed.logs] == [4, 5, 6, 7]
    torch.testing.assert_close(resumed.model.weight, full.model.weight, rtol=0, atol=0)
    if use_ema:
        torch.testing.assert_close(resumed.ema_model.weight, full.ema_model.weight, rtol=0, atol=0)
    state = resumed.optimizer.state[resumed.model.weight]
    expected_state = full.optimizer.state[full.model.weight]
    for key in ['step', 'exp_avg', 'exp_avg_sq']:
        torch.testing.assert_close(state[key], expected_state[key], rtol=0, atol=0)
    expected_lr = {step: row['lr'] for step, row in full.logs}
    assert all(row['lr'] == expected_lr[step] for step, row in resumed.logs)


def test_finished_checkpoint_does_not_start_more_epochs(make_workspace):
    full = make_workspace('finished')
    full.run()
    resumed = make_workspace('finished', resume=True)
    resumed.run()
    assert resumed.epoch == 4
    assert resumed.global_step == 8
    assert resumed.logs == []
    torch.testing.assert_close(resumed.model.weight, full.model.weight)


def test_loading_for_evaluation_keeps_saved_epoch_number(make_workspace):
    full = make_workspace('evaluation')
    full.run()
    evaluation = make_workspace('evaluation')
    evaluation.load_checkpoint()
    assert evaluation.epoch == 3
    assert evaluation.global_step == 7
    torch.testing.assert_close(evaluation.ema_model.weight, full.ema_model.weight)


def test_extend_finished_cosine_run_trains_only_remaining_epochs(make_workspace):
    finished = make_workspace('extended')
    finished.run()
    assert finished.optimizer.param_groups[0]['lr'] == 0
    previous_weight = finished.model.weight.detach().clone()

    extended = make_workspace('extended', resume=True)
    extended.cfg.training.num_epochs = 8
    extended.run()

    assert extended.epoch == 8
    assert extended.global_step == 16
    assert {row['epoch'] for _, row in extended.logs} == {4, 5, 6, 7}
    assert [step for step, _ in extended.logs] == list(range(8, 16))
    assert extended.logs[0][1]['lr'] > 0
    assert extended.optimizer.param_groups[0]['lr'] == 0
    assert extended.optimizer.state[extended.model.weight]['step'].item() == 16
    assert not torch.equal(extended.model.weight, previous_weight)
