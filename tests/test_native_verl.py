"""CPU integration tests for EPRO against real, patched verl."""
import os
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

if not os.getenv("VERL_ROOT"):
    pytest.skip("Set VERL_ROOT and install the native test dependencies.", allow_module_level=True)

from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass, validate_config as native_validate
from verl.workers.utils.padding import no_padding_2_padding
from verl.trainer.ppo.transported_sc import (
    TSC_COLLECT_FLAG, TSC_OLD_HEAD_KEY, register_transport_loss,
    transported_sc_logits_processor, transported_sc_ppo_loss,
)

from train import compose_config, precision_overrides


def test_native_verl_configuration_validation():
    cfg = compose_config(Path(os.environ["VERL_ROOT"]), precision_overrides("nvfp4"))
    native_validate(cfg, use_reference_policy=False, use_critic=False)


def nested(values, lengths):
    offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
    return torch.nested.nested_tensor_from_jagged(values, offsets, min_seqlen=min(lengths), max_seqlen=max(lengths))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_old_head_collection_and_real_ppo_loss_backward(dtype):
    generator = torch.Generator().manual_seed(42)
    n, vocab, k = 8, 17, 4
    q = torch.randn(n, vocab, generator=generator).log_softmax(-1)
    p0_logits = torch.randn(n, vocab, generator=generator)
    current = (p0_logits + 0.15 * torch.randn(n, vocab, generator=generator)).to(dtype).requires_grad_()
    q_head, ids = q.topk(k, -1)
    data = TensorDict({
        "prompts": nested(torch.tensor([1, 2, 3]), [2, 1]),
        "responses": nested(torch.tensor([4, 5, 6, 7, 8]), [3, 2]),
        "response_mask": nested(torch.ones(5, dtype=torch.bool), [3, 2]),
        "advantages": nested(torch.tensor([1., -1., 0.5, -0.5, 1.5]), [3, 2]),
        "rollout_topk_ids": nested(ids, [5, 3]),
        "rollout_topk_log_probs": nested(q_head, [5, 3]),
    }, batch_size=[2])
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=5, global_batch_size=2, **{TSC_COLLECT_FLAG: True})
    collected = transported_sc_logits_processor(p0_logits[None], data)
    torch.testing.assert_close(collected[TSC_OLD_HEAD_KEY][0], p0_logits.log_softmax(-1).gather(-1, ids))
    assert not collected[TSC_OLD_HEAD_KEY].requires_grad
    data[TSC_OLD_HEAD_KEY] = nested(collected[TSC_OLD_HEAD_KEY][0], [5, 3])
    tu.assign_non_tensor(data, **{TSC_COLLECT_FLAG: False})
    p0_a = p0_logits.log_softmax(-1)[:, 0]
    q_a = q[:, 0]
    p0_response = no_padding_2_padding(p0_a, data)
    q_response = no_padding_2_padding(q_a, data)
    data["old_log_probs"] = nested(torch.cat([p0_response[0, :3], p0_response[1, :2]]), [3, 2])
    data["rollout_log_probs"] = nested(torch.cat([q_response[0, :3], q_response[1, :2]]), [3, 2])
    cfg = compose_config(Path(os.environ["VERL_ROOT"]), [])
    config = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
    outputs = transported_sc_ppo_loss(config, student_logits=current[None], data=data)
    model_output = {key: nested(value[0], [5, 3]) for key, value in outputs.items()}
    model_output["log_probs"] = nested(current.log_softmax(-1)[:, 0], [5, 3])
    register_transport_loss()
    loss, metrics = transported_sc_ppo_loss(config, model_output=model_output, data=data)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert current.grad is not None and torch.isfinite(current.grad).all()
    assert current.grad.abs().sum() > 0
    assert "actor/pg_loss" in metrics


def test_nvfp4_compressor_adapter_packs_on_cpu():
    from verl.utils.qat.quantizer import QATQuantizer

    quantizer = QATQuantizer(device=torch.device("cpu"), param_dtype=torch.bfloat16)
    key = "model.layers.0.self_attn.q_proj"
    weight = torch.randn(32, 32, generator=torch.Generator().manual_seed(42), dtype=torch.bfloat16)
    packed = dict(quantizer.quantize_with_fusion({key + ".weight": weight}, target_device=torch.device("cpu")))
    assert packed[key + ".weight_packed"].shape == (32, 16)
    assert packed[key + ".weight_packed"].dtype == torch.uint8
    assert packed[key + ".weight_scale"].shape == (32, 2)
    assert torch.isfinite(packed[key + ".weight_scale"].float()).all()
    assert torch.isfinite(packed[key + ".weight_global_scale"]).all()
