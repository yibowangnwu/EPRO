# Packaged-release validation

Validated on 2026-11-20 using Python 3.12.16 on a macOS ARM64 CPU. PyTorch 2.13.0, Transformers 5.12.1, Ray 2.54.1, TensorDict 0.11.0, compressed-tensors 0.17.0, and Hydra 1.3.8 were installed in an isolated temporary environment.

The real verl checkout was pinned to `4b26c788d41ce932ce7706c5ee8e249ab52f4c51` and patched using `scripts/install_verl.py`.

```text
VERL_ROOT=<patched-checkout> python -m pytest tests -q
30 passed, 14 warnings in 7.66s
```

The warnings are upstream PyTorch deprecations for `torch.jit.script_method`.

Coverage includes:

- Transport normalizers, sampled-action weights, and score gradients against an explicitly normalized full-vocabulary oracle, with complete and incomplete recorded heads. The incomplete-head oracle uses the declared old-trainer-proportional tail approximation.
- Anchor recovery, zero-mismatch behavior, common-logit-offset invariance, saturated head masses, and the bucketed Policy Shift formula.
- SSC attenuation, outward/inward direction, threshold behavior, stop-gradient weights, and importance truncation.
- Actual upstream Hydra composition for EPRO in both BF16 and NVFP4 configurations, plus native verl configuration validation.
- Old learner head collection and backward through the real verl training-loss pipeline with unequal-length jagged batches, for EPRO, in FP32 and BF16.
- Clean patch application and repeated installation, refusal to overwrite modified hooks, and local JSONL-to-parquet conversion.
- The compressed-tensors 0.17.0 adapter producing packed NVFP4 weights and scales on CPU.

The shell entry points pass `bash -n`, and the full training command including Ray environment overrides composes in `--dry-run` mode. Repository files were checked for credential patterns and private machine paths.

After removing the comparison-method branches, the EPRO per-token loss and gradients were also compared with the previous release on a fixed randomized batch. They matched exactly.

GPU FSDP execution, vLLM kernels, distributed Ray/TransferQueue execution, and live quantized weight synchronization were not tested for this packaged release. The NVFP4 GPU environment uses the pinned upstream `uv.lock`; the complete Linux CUDA environment installation was not performed on this macOS machine. These checks establish CPU numerical and integration correctness, not a completed GPU training run or benchmark reproduction.
