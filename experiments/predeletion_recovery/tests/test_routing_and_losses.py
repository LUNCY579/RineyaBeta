import torch

from pdrr.losses import causal_significance, component_recovery_loss
from pdrr.routing import build_router, router_regularization


def test_all_router_modes_normalize_responsibility():
    state = torch.randn(2, 5, 12)
    positions = torch.arange(5)
    for mode in ("uniform", "random", "static", "state", "state_position"):
        router = build_router(
            mode, state_dim=12, recipient_layers=[1, 3, 4], source_layer=2,
            seed=17, top_k=2, max_positions=16)
        assert router is not None
        route = router(state, positions)
        assert route.shape == (2, 5, 3)
        assert torch.allclose(route.sum(-1), torch.ones(2, 5), atol=1e-6)
        assert torch.all((route > 0).sum(-1) <= 2)


def test_random_router_is_seed_reproducible():
    state = torch.randn(1, 3, 8)
    first = build_router("random", state_dim=8, recipient_layers=[1, 3],
                         source_layer=2, seed=91)
    second = build_router("random", state_dim=8, recipient_layers=[1, 3],
                          source_layer=2, seed=91)
    assert torch.equal(first(state), second(state))


def test_learned_router_receives_gradient_only_through_loss_side():
    state = torch.randn(2, 4, 8, requires_grad=True)
    router = build_router("state_position", state_dim=8, recipient_layers=[1, 3],
                          source_layer=2, seed=1, max_positions=8)
    route = router(state, torch.arange(4))
    loss = (route[..., 0] * state[..., 0]).mean()
    loss.backward()
    assert state.grad is not None
    assert any(parameter.grad is not None for parameter in router.parameters())


def test_component_recovery_prefers_matching_recipient_delta():
    source = torch.randn(2, 4, 6)
    correct = [source.clone(), torch.zeros_like(source)]
    wrong = [torch.zeros_like(source), source.clone()]
    route = torch.zeros(2, 4, 2)
    route[..., 0] = 1.0
    good, _ = component_recovery_loss(source, correct, route, None)
    bad, _ = component_recovery_loss(source, wrong, route, None)
    assert good < bad


def test_causal_significance_is_nonnegative_and_normalized():
    full = torch.randn(2, 4, 7)
    ablated = full.clone()
    ablated[:, 2] += 1.5
    significance = causal_significance(full, ablated)
    assert torch.all(significance > 0)
    assert torch.allclose(significance.mean(dim=1), torch.ones(2), atol=1e-5)


def test_router_regularization_is_finite():
    route = torch.softmax(torch.randn(3, 5, 4), dim=-1)
    regularization = router_regularization(route, entropy_weight=0.01,
                                           load_balance_weight=0.1)
    assert torch.isfinite(regularization.total)
