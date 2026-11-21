# EPRO

**Execution-Policy Robust Optimization** for *Robust Policy Optimization Through an Imperfect Execution-Policy Predictor*.

EPRO combines **Transported Score Centering (TSC)** with **Shift-Aware Soft Clipping (SSC)**. It predicts the execution policy after learner updates using a rollout-time anchor, then attenuates outward updates when **Policy Shift** becomes large.

This repository contains the algorithm, small integration patches for [verl](https://github.com/verl-project/verl), and one Qwen3-4B-Base training example.

## Installation

Use Linux, Python 3.12, and a CUDA environment compatible with the pinned verl CUDA 13 wheels. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
bash scripts/setup.sh
```

The installer checks out verl at `4b26c788d41ce932ce7706c5ee8e249ab52f4c51`, applies the patches, and uses **that revision's committed `uv.lock`** for the environment. The relevant versions are PyTorch 2.13.0, vLLM 0.29.0, Transformers 5.12.1, and compressed-tensors 0.17.0. The checkout lives in `third_party/verl`; set `VERL_ROOT` to use another location.

The integration intentionally targets this exact revision. The installer rejects other revisions or modified hook files and accepts an unchanged repeat installation. This avoids silently applying a training patch to an incompatible verl version.

## Data

Supply separate training and validation parquet files in verl's RL dataset format. The essential fields are:

```json
{
  "data_source": "math",
  "prompt": [{"role": "user", "content": "Solve the problem and put the final answer in \\boxed{}."}],
  "reward_model": {"style": "rule", "ground_truth": "42"}
}
```

For local JSONL files with `problem` and `answer` fields:

```bash
third_party/verl/.venv/bin/python scripts/prepare_data.py train.jsonl data/train.parquet
third_party/verl/.venv/bin/python scripts/prepare_data.py validation.jsonl data/val.parquet
```

The paper's 4B training used GURU math. This release accepts prepared data rather than redistributing the study's dataset preprocessing or private data paths. The example reward uses verl's bundled boxed-answer evaluator; the study's original custom evaluator is not included. To use another evaluator, override `reward.custom_reward_function.path` and `.name`. This is a portable algorithm example, rather than a complete benchmark reproduction package.

## Training

```bash
bash scripts/train_4b.sh \
  --train-file data/train.parquet \
  --val-files data/val.parquet
```

The default uses a BF16 learner and **NVFP4 W4A16 rollout weights**. The quantization patch retains rollout packing and weight reloads while disabling learner fake quantization (`VERL_QAT_ROLLOUT_ONLY=1`). This preserves the training–inference mismatch studied in the paper.

For a BF16 rollout wiring check instead:

```bash
bash scripts/train_4b.sh --precision bf16 \
  --train-file data/train.parquet --val-files data/val.parquet
```

The defaults are Qwen3-4B-Base, AdamW with learning rate `1e-6`, 256 prompts per rollout, 8 responses per prompt, minibatches of 16 prompt groups, one learner pass, and 400 rollout iterations. This gives 16 learner minibatches per rollout iteration. The prompt/response limits are 1024/4096, the importance cap is 2, and SSC uses `tau=0.125`, `sigma=0.062`. Full-parameter training is used, with no KL or entropy penalty. The default allocation is two GPUs; memory budgets can be adjusted for your hardware.

Pass additional Hydra overrides explicitly:

```bash
bash scripts/train_4b.sh --train-file data/train.parquet --val-files data/val.parquet \
  --set trainer.total_training_steps=2 \
  --set trainer.test_freq=-1 \
  --set actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
```

W&B is optional; console logging is the default. After authenticating W&B in your environment, enable it with `--set 'trainer.logger=[console,wandb]'`. No credentials are stored in this repository.

## Algorithm and integration

The main implementation is [`epro/transported_sc.py`](epro/transported_sc.py). Let `q=mu_0` be the rollout execution policy, `p0=pi_0` the old learner, and `u=pi_theta` the current learner. The rollout records a fixed top-32 head `H`. Outside `H`, the execution tail is approximated by `q_tilde(v)=rho*p0(v)`, where `rho=(1-q(H))/(1-p0(H))`. Then:

```text
Z = sum_{v in H} q(v) u(v) / p0(v) + rho (1-u(H))
mu_hat(v) = q(v) u(v) / (p0(v) Z)  on H
mu_hat(v) = rho u(v) / Z           outside H
```

The correction's gradient equals the expected learner score under this transported head/tail distribution. The importance weight is computed under the same tail proxy: `w=exp(log u(a)-log p0(a)-log Z)`. The top-K approximation is explicit; individual unrecorded tail probabilities are not reconstructed from measurements.

Policy Shift is `D=sqrt(Var_{p0}[log u-log p0])`, evaluated on the recorded head **plus one tail bucket**. It is a coarse-grained statistic. For outward updates (`A * log(u(a)/p0(a)) > 0`), SSC uses `kappa=exp(-max(D-tau,0)/sigma)`; otherwise `kappa=1`. The advantage, capped importance weight, and attenuation are detached in the surrogate.

[`patches/verl-transport.patch`](patches/verl-transport.patch) adds four small hooks: record old learner head probabilities during the existing old-policy forward, accept per-token head vectors in the FSDP engine, select the logits processor, and pass transport quantities into the policy loss. [`patches/verl-nvfp4-rollout.patch`](patches/verl-nvfp4-rollout.patch) separately contains the rollout-only quantization and vLLM weight-reload compatibility changes used by this example.

This release validates the synchronous v1 FSDP path, remove-padding logits, sequence parallelism 1, and full-vocabulary sampling at temperature 1 (`top_p=1`, `top_k=-1`). The launcher checks these assumptions before training. No extra execution-engine forward is needed during minibatch updates.

## Validation

Without a GPU, compose and check the real upstream training configuration:

```bash
bash scripts/train_4b.sh --dry-run
```

If using the separate CPU environment below, set `EPRO_PYTHON="$PWD/.venv/bin/python"` when calling the shell entry point.

For CPU tests in a separate environment:

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e '.[test]' -r requirements-test.txt
git clone https://github.com/verl-project/verl.git third_party/verl
git -C third_party/verl checkout --detach 4b26c788d41ce932ce7706c5ee8e249ab52f4c51
.venv/bin/python scripts/install_verl.py third_party/verl
VERL_ROOT="$PWD/third_party/verl" .venv/bin/python -m pytest -q
```

The tests compare transport normalizers, importance weights, and gradients against an explicit vocabulary oracle; verify SSC's direction, threshold, and stop-gradient behavior; validate the EPRO configuration in BF16 and NVFP4 modes; and exercise old-head collection and backward through **the real verl training-loss pipeline** using jagged tensors. They also check patch installation, data conversion, and the NVFP4 compressor adapter on CPU. Tests do not launch distributed workers or vLLM. GPU training and quantized weight reloads have not been re-run for this packaged release. See [VALIDATION.md](VALIDATION.md).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for upstream attribution.
