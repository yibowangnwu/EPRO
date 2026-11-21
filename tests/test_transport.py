import pytest
import torch

from epro.transported_sc import head_divergences, transported_head_terms, transported_token_loss


@pytest.mark.parametrize("seed", [7, 41, 123])
@pytest.mark.parametrize("k", [3, 11])
def test_transport_normalizer_weights_and_score_match_explicit_vocabulary(seed, k):
    """Compare the actual surrogate gradient against an explicitly normalized categorical policy.

    The incomplete-head oracle uses exactly the declared p0-proportional tail.
    Both head and tail actions are checked; full-vocabulary heads are checked too.
    """
    generator = torch.Generator().manual_seed(seed)
    p0 = torch.randn(4, 11, generator=generator).softmax(-1)
    q = torch.randn(4, 11, generator=generator).softmax(-1)
    logits = torch.randn(4, 11, generator=generator, requires_grad=True)
    log_u = logits.log_softmax(-1)
    ids = q.topk(k, dim=-1).indices
    terms = transported_head_terms(log_u.gather(-1, ids), q.log().gather(-1, ids), p0.log().gather(-1, ids), head_complete=k == 11)
    q_proxy = p0 * terms["rho"][:, None]
    q_proxy = q_proxy.scatter(-1, ids, q.gather(-1, ids))
    unnormalized = q_proxy * log_u.exp() / p0
    z = unnormalized.sum(-1)
    predicted = unnormalized / z[:, None]
    torch.testing.assert_close(terms["log_z"].exp(), z, atol=5e-7, rtol=2e-6)
    grad_center = torch.autograd.grad(terms["correction"].sum(), logits, retain_graph=True)[0]
    expected_center = torch.autograd.grad((predicted.detach() * log_u).sum(), logits, retain_graph=True)[0]
    torch.testing.assert_close(grad_center, expected_center, atol=6e-7, rtol=2e-5)
    for action in [ids[:, 0], (ids[:, 0] + 1) % 11]:
        action = action[:, None]
        surrogate = log_u.gather(-1, action).squeeze(-1) - terms["correction"]
        exact = predicted.gather(-1, action).log().sum()
        got = torch.autograd.grad(surrogate.sum(), logits, retain_graph=True)[0]
        want = torch.autograd.grad(exact, logits, retain_graph=True)[0]
        torch.testing.assert_close(got, want, atol=7e-7, rtol=3e-5)
        weight = (log_u.gather(-1, action).squeeze(-1) - p0.log().gather(-1, action).squeeze(-1) - terms["log_z"]).exp()
        oracle = (predicted / q_proxy).gather(-1, action).squeeze(-1)
        torch.testing.assert_close(weight, oracle, atol=7e-7, rtol=3e-6)


def test_anchor_transport_recovers_sampler_centering():
    p0 = torch.tensor([[0.3, 0.4, 0.2, 0.1]])
    q = torch.tensor([[0.45, 0.25, 0.2, 0.1]])
    logits = p0.log().requires_grad_()
    log_u = logits.log_softmax(-1)
    terms = transported_head_terms(log_u[:, :2], q[:, :2].log(), p0[:, :2].log())
    torch.testing.assert_close(terms["log_z"], torch.zeros(1), atol=2e-7, rtol=0)
    grad = torch.autograd.grad(terms["correction"].sum(), logits)[0]
    torch.testing.assert_close(grad, q - p0, atol=2e-7, rtol=2e-6)


def test_without_mismatch_centering_gradient_is_zero():
    p0 = torch.tensor([[0.3, 0.4, 0.2, 0.1]])
    logits = torch.tensor([[1.0, 0.3, -0.4, -1.0]], requires_grad=True)
    terms = transported_head_terms(logits.log_softmax(-1)[:, :2], p0[:, :2].log(), p0[:, :2].log())
    grad = torch.autograd.grad(terms["correction"].sum(), logits)[0]
    torch.testing.assert_close(grad, torch.zeros_like(grad), atol=1e-7, rtol=0)


def test_policy_shift_is_bucketed_weighted_standard_deviation_and_offset_invariant():
    p0 = torch.tensor([[0.30, 0.25, 0.2, 0.15, 0.1]])
    logits = torch.tensor([[1.0, 0.3, -0.4, -1.0, -2.0]])
    log_u = logits.log_softmax(-1)
    measured = head_divergences(log_u[:, :3], p0[:, :3].log())
    p_bucket = torch.cat([p0[:, :3], p0[:, 3:].sum(-1, keepdim=True)], -1)
    u_bucket = torch.cat([log_u.exp()[:, :3], log_u.exp()[:, 3:].sum(-1, keepdim=True)], -1)
    delta = (u_bucket / p_bucket).log()
    mean = (p_bucket * delta).sum(-1, keepdim=True)
    expected = (p_bucket * (delta - mean).square()).sum(-1).sqrt()
    torch.testing.assert_close(measured["lrms"], expected)
    shifted = head_divergences((logits + 42).log_softmax(-1)[:, :3], p0[:, :3].log())
    torch.testing.assert_close(shifted["lrms"], measured["lrms"], atol=2e-6, rtol=1e-5)
    zero = head_divergences(p0[:, :3].log(), p0[:, :3].log())
    torch.testing.assert_close(zero["lrms"], torch.zeros(1))


def test_ssc_retains_inward_terms_and_attenuates_outward_terms_only():
    # Four combinations: positive/out, positive/in, negative/out, negative/in.
    log_prob = torch.tensor([0.3, -0.3, -0.3, 0.3], requires_grad=True)
    advantage = torch.tensor([1.0, 1.0, -1.0, -1.0], requires_grad=True)
    div = torch.full((4,), 0.187, requires_grad=True)
    loss, metrics = transported_token_loss(log_prob, torch.zeros(4), torch.zeros(4), torch.zeros(4), advantage,
        div=div, div_delta=0.125, div_scale=0.062)
    torch.testing.assert_close(metrics["gate"], torch.tensor([(-torch.ones(())).exp(), 1.0, (-torch.ones(())).exp(), 1.0]))
    gradient = torch.autograd.grad(loss.sum(), [log_prob, advantage, div], allow_unused=True)
    torch.testing.assert_close(gradient[0], -advantage.detach() * log_prob.detach().exp() * metrics["gate"])
    assert gradient[1] is None and gradient[2] is None


@pytest.mark.parametrize("shift", [0.0, 0.124, 0.125])
def test_ssc_is_open_up_to_threshold(shift):
    _, diag = transported_token_loss(torch.tensor([0.4]), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.ones(1),
        div=torch.tensor([shift]), div_delta=0.125, div_scale=0.062)
    torch.testing.assert_close(diag["gate"], torch.ones(1))


def test_importance_cap_and_ssc_keep_separate_weights():
    log_prob = torch.tensor([1.3, 0.9, 0.7, 1.1]).log().requires_grad_()
    advantage = torch.tensor([1.0, 1.0, -1.0, -1.0])
    shift = torch.full((4,), 0.187)
    loss, diag = transported_token_loss(log_prob, torch.zeros(4), torch.zeros(4), torch.full((4,), -2.0),
                                       advantage, div=shift)
    gate = torch.tensor([(-torch.ones(())).exp(), 1.0, (-torch.ones(())).exp(), 1.0])
    torch.testing.assert_close(diag["gate"], gate)
    torch.testing.assert_close(diag["w_capped"], torch.full((4,), 2.0))
    torch.testing.assert_close(torch.autograd.grad(loss.sum(), log_prob)[0], -advantage * 2.0 * gate)


def test_tail_near_saturation_is_finite():
    log_u = torch.tensor([[0.0, -80.0]], requires_grad=True)
    terms = transported_head_terms(log_u, log_u.detach(), log_u.detach())
    assert all(torch.isfinite(x).all() for x in terms.values())
    assert torch.isfinite(torch.autograd.grad(terms["correction"].sum(), log_u)[0]).all()


def test_invalid_ssc_scale_rejected():
    with pytest.raises(ValueError, match="scale"):
        transported_token_loss(torch.ones(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.ones(1),
            div=torch.ones(1), div_scale=0)
