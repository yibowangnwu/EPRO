#!/usr/bin/env python3
"""Train EPRO on Qwen3-4B-Base."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def precision_overrides(precision):
    # The QAT plumbing packs rollout weights; VERL_QAT_ROLLOUT_ONLY disables fake quantization in the trainer.
    result = []
    for path in ("actor_rollout_ref.actor.fsdp_config.qat", "actor_rollout_ref.rollout.qat"):
        result += [f"++{path}.enable={'true' if precision == 'nvfp4' else 'false'}"]
        if precision == "nvfp4":
            result += [f"++{path}.mode=w4a16", f"++{path}.group_size=16",
                       f"++{path}.quantization_config_path=" + json.dumps(str(ROOT / "configs/nvfp4_w4a16.json"))]
    return result


def compose_config(verl_root, overrides):
    from hydra import compose, initialize_config_dir

    os.environ["VERL_ROOT"] = str(verl_root)
    os.environ["EPRO_ROOT"] = str(ROOT)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="qwen3_4b", overrides=overrides)


def validate_config(cfg):
    a, r = cfg.actor_rollout_ref.actor, cfg.actor_rollout_ref.rollout
    if a.strategy != "fsdp" or not cfg.trainer.use_v1 or cfg.trainer.v1.trainer_mode != "sync":
        raise ValueError("This release supports the v1 synchronous FSDP trainer.")
    if not cfg.actor_rollout_ref.model.use_remove_padding or a.use_fused_kernels or cfg.actor_rollout_ref.model.use_fused_kernels:
        raise ValueError("EPRO requires remove-padding forwards and unfused logits.")
    if a.fsdp_config.ulysses_sequence_parallel_size != 1 or a.ulysses_sequence_parallel_size != 1:
        raise ValueError("The example is validated with sequence parallelism = 1.")
    if r.top_p != 1 or r.top_k != -1 or r.temperature != 1:
        raise ValueError("Use full-vocabulary sampling at temperature 1 for this top-K/tail implementation.")
    if not r.calculate_log_probs:
        raise ValueError("Rollout action log-probabilities are required.")
    if cfg.data.train_batch_size % a.ppo_mini_batch_size:
        raise ValueError("train_batch_size must be divisible by ppo_mini_batch_size.")
    if a.policy_loss.loss_mode != "transported_sc" or not a.policy_loss.rollout_correction.get("transported_sc", False):
        raise ValueError("This entry point trains EPRO with transported score centering and SSC.")
    if r.topk_log_probs <= 0 or cfg.algorithm.rollout_correction.bypass_mode:
        raise ValueError("Transport requires a recorded head and a recomputed old trainer policy.")
    if cfg.algorithm.rollout_correction.rollout_is is not None:
        raise ValueError("Do not apply external importance sampling on top of transported weights.")
    if a.policy_loss.rollout_correction.tsc_div_scale <= 0:
        raise ValueError("SSC scale must be positive.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=["nvfp4", "bf16"], default="nvfp4")
    parser.add_argument("--train-file", default=os.getenv("TRAIN_FILE"))
    parser.add_argument("--val-files", nargs="+", default=None)
    parser.add_argument("--verl-root", type=Path, default=Path(os.getenv("VERL_ROOT", ROOT / "third_party/verl")))
    parser.add_argument("--dry-run", action="store_true", help="Compose and check real verl configs; no GPU or data read.")
    parser.add_argument("--set", action="append", default=[], metavar="HYDRA_OVERRIDE")
    args = parser.parse_args()
    verl_root = args.verl_root.resolve()
    if not (verl_root / "verl/trainer/config/ppo_trainer.yaml").is_file():
        parser.error("verl checkout missing; run bash scripts/setup.sh first.")
    overrides = precision_overrides(args.precision)
    overrides += ["trainer.experiment_name=qwen3-4b-epro",
                  "trainer.default_local_dir=" + json.dumps(str(ROOT / "checkpoints" / "qwen3-4b-epro"))]
    if args.train_file:
        overrides.append("data.train_files=" + json.dumps([str(Path(args.train_file).resolve())]))
    val_files = args.val_files
    if val_files is None and os.getenv("VAL_FILES"):
        val_files = json.loads(os.environ["VAL_FILES"])
        if not isinstance(val_files, list) or not all(isinstance(x, str) for x in val_files):
            parser.error("VAL_FILES must be a JSON list of paths.")
    if val_files:
        overrides.append("data.val_files=" + json.dumps([str(Path(p).resolve()) for p in val_files]))
    runtime = {"VERL_QAT_ROLLOUT_ONLY": "1", "VERL_QAT_SYNC_DEVICE": os.getenv("VERL_QAT_SYNC_DEVICE", "gpu"),
               "EPRO_ROOT": str(ROOT), "VERL_ROOT": str(verl_root)}
    overrides += ["++ray_kwargs.ray_init.runtime_env.env_vars." + key + "=" + json.dumps(value)
                  for key, value in runtime.items()]
    overrides += args.set
    cfg = compose_config(verl_root, overrides)
    validate_config(cfg)
    command = [sys.executable, "-m", "verl.trainer.main_ppo", "--config-path", str(ROOT / "configs"),
               "--config-name", "qwen3_4b", *overrides]
    if args.dry_run:
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(cfg, resolve=True))
        print("# Launch command:\n" + shlex.join(command))
        return
    for filename in list(cfg.data.train_files) + list(cfg.data.val_files):
        if not Path(filename).is_file():
            parser.error(f"Dataset not found: {filename}. Supply --train-file and --val-files.")
    from install_verl import install

    # Verify patch series and exact core file before training.
    install(verl_root, nvfp4=True)
    import torch

    if not torch.cuda.is_available():
        parser.error("CUDA is required for training; use --dry-run for CPU configuration validation.")
    if torch.cuda.device_count() < cfg.trainer.n_gpus_per_node:
        parser.error("Fewer visible GPUs than trainer.n_gpus_per_node.")
    env = os.environ.copy()
    env["VERL_QAT_ROLLOUT_ONLY"] = "1"
    env.setdefault("VERL_QAT_SYNC_DEVICE", "gpu")
    env["PYTHONPATH"] = str(verl_root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, cwd=verl_root, env=env, check=True)


if __name__ == "__main__":
    main()
