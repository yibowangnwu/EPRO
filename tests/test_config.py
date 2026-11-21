import os
from pathlib import Path
import subprocess

import pytest
from omegaconf import OmegaConf

from train import compose_config, precision_overrides, validate_config


@pytest.fixture
def verl_root():
    path = Path(os.environ.get("VERL_ROOT", "third_party/verl")).resolve()
    if not (path / "verl/trainer/config/ppo_trainer.yaml").is_file():
        pytest.skip("Set VERL_ROOT to the pinned verl checkout for real Hydra configuration tests.")
    return path


@pytest.mark.parametrize("precision", ["nvfp4", "bf16"])
def test_epro_composes_real_upstream_configs(verl_root, precision):
    cfg = compose_config(verl_root, precision_overrides(precision))
    validate_config(cfg)
    # Resolve every interpolation without instantiating CUDA workers.
    resolved = OmegaConf.to_container(cfg, resolve=True)
    a = cfg.actor_rollout_ref.actor
    assert cfg.data.train_batch_size // a.ppo_mini_batch_size == 16
    assert a.ppo_epochs == 1 and a.optim.lr == 1e-6
    assert bool(cfg.actor_rollout_ref.rollout.qat.enable) == (precision == "nvfp4")
    assert cfg.algorithm.rollout_correction.rollout_is is None
    assert not cfg.algorithm.rollout_correction.bypass_mode
    assert a.policy_loss.loss_mode == "transported_sc"
    assert a.policy_loss.rollout_correction.tsc_div_delta == 0.125
    assert a.policy_loss.rollout_correction.tsc_div_scale == 0.062
    assert resolved["trainer"]["total_training_steps"] == 400


def test_double_importance_correction_rejected(verl_root):
    cfg = compose_config(verl_root, ["algorithm.rollout_correction.rollout_is=token"])
    with pytest.raises(ValueError, match="external importance"):
        validate_config(cfg)


def test_truncated_sampling_rejected(verl_root):
    cfg = compose_config(verl_root, ["actor_rollout_ref.rollout.top_p=0.9"])
    with pytest.raises(ValueError, match="full-vocabulary"):
        validate_config(cfg)


def test_shell_entry_syntax():
    root = Path(__file__).resolve().parents[1]
    for script in (root / "scripts").glob("*.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True)
