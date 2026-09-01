import torch

from dast.config import DASTConfig, ExperimentConfig, OperatorConfig, SemanticConfig
from dast.losses import cosine_alignment_loss, depth_alignment_metrics, multi_positive_contrastive_loss
from dast.operators import LowRankOperatorBank


def test_operator_bank_shapes_and_gradients():
    bank = LowRankOperatorBank(12, 4, 3)
    states = torch.randn(2, 5, 12, requires_grad=True)
    transformed, weights = bank(states)
    assert transformed.shape == states.shape
    assert weights.shape == (2, 5, 4)
    loss = transformed.square().mean() + bank.diversity_loss() + bank.balance_loss(weights)
    loss.backward()
    assert states.grad is not None
    assert torch.isfinite(loss)


def test_operator_starts_as_identity():
    bank = LowRankOperatorBank(8, 3, 2)
    states = torch.nn.functional.normalize(torch.randn(2, 4, 8), dim=-1)
    transformed, _ = bank(states)
    assert torch.allclose(transformed, states, atol=1e-6)


def test_matching_state_loss_is_lower():
    targets = torch.nn.functional.normalize(torch.randn(3, 4, 10), dim=-1)
    matched = cosine_alignment_loss(targets, targets)
    mismatched = cosine_alignment_loss(-targets, targets)
    assert matched < mismatched


def test_contrastive_handles_duplicate_targets_as_multi_positive():
    targets = torch.nn.functional.normalize(torch.randn(3, 2, 6), dim=-1)
    targets[1] = targets[0]
    groups = torch.tensor([[1, 2], [1, 2], [3, 4]])
    loss = multi_positive_contrastive_loss(targets, targets, group_ids=groups, temperature=0.1)
    assert torch.isfinite(loss)


def test_depth_alignment_metrics_reward_correct_order():
    target = torch.nn.functional.normalize(torch.randn(5, 4, 8), dim=-1)
    correct = depth_alignment_metrics(target, target)
    reversed_metrics = depth_alignment_metrics(target.flip(1), target)
    assert correct["state/checkpoint_top1"] > reversed_metrics["state/checkpoint_top1"]
    assert correct["state/order_margin"] > reversed_metrics["state/order_margin"]


def test_operator_arm_contract():
    cfg = DASTConfig(
        experiment=ExperimentConfig(arm="operator_ordered"),
        semantic=SemanticConfig(state_weight=1.0),
        operators=OperatorConfig(enabled=True),
    )
    cfg.validate()


def test_supervised_stages_ignore_unsupervised_cosine_and_contrastive():
    torch.manual_seed(0)
    targets = torch.nn.functional.normalize(torch.randn(4, 4, 8), dim=-1)
    pred = targets.clone()
    pred[:, 1:] = torch.nn.functional.normalize(torch.randn(4, 3, 8), dim=-1)
    weights = (1.0, 0.0, 0.0, 0.0)
    matched = cosine_alignment_loss(targets, targets, stage_weights=weights)
    mixed = cosine_alignment_loss(pred, targets, stage_weights=weights)
    assert torch.allclose(matched, mixed, atol=1e-6)
    contrastive_a = multi_positive_contrastive_loss(
        pred, targets, group_ids=None, temperature=0.1, stage_weights=weights
    )
    pred2 = pred.clone()
    pred2[:, 3] = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    contrastive_b = multi_positive_contrastive_loss(
        pred2, targets, group_ids=None, temperature=0.1, stage_weights=weights
    )
    assert torch.allclose(contrastive_a, contrastive_b, atol=1e-5)


def test_target_stage_remap_puts_state1_at_deepest_checkpoint():
    from dast.losses import remap_stage_tensors

    grid = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    remapped = remap_stage_tensors(grid, (0, 1, 2, 0))
    assert torch.equal(remapped[:, 3], grid[:, 0])
    assert torch.equal(remapped[:, 0], grid[:, 0])
