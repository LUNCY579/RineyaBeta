from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

from peer_stabilizer import (  # noqa: E402
    DESCRIPTOR_DIM,
    LayerMetadata,
    LayerView,
    PeerStabilizerMatrix,
    audit_sparse_delta,
    exact_topk_mask,
    matrix_descriptor,
    matrix_property_map,
    one_step_weight,
    project_property_map,
    straight_through_topk,
    structural_profile_loss,
)


def make_layers() -> list[LayerView]:
    g = torch.Generator().manual_seed(11)
    layers = []
    for i in range(4):
        module_type = "mlp.down_proj" if i != 1 else "self_attn.q_proj"
        weight = torch.randn(7, 9, generator=g) * (1 + 0.1 * i)
        grad = torch.randn(7, 9, generator=g) if i == 3 else None
        layers.append(
            LayerView(
                weight=weight,
                gradient=grad,
                activation=torch.randn(2, 7, generator=g),
                metadata=LayerMetadata(f"layer_{i}", i, module_type),
            )
        )
    return layers


class DescriptorAndPropertyTests(unittest.TestCase):
    def test_descriptor_contract_and_finiteness(self) -> None:
        layers = make_layers()
        d = matrix_descriptor(layers[0], len(layers))
        self.assertEqual(d.shape, (DESCRIPTOR_DIM,))
        self.assertTrue(bool(torch.isfinite(d).all()))

    def test_property_projection_preserves_equal_shape(self) -> None:
        w = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        props = matrix_property_map(w)
        projected = project_property_map(props, (4, 5))
        self.assertTrue(torch.equal(props, projected))
        resized = project_property_map(props, (8, 3))
        self.assertEqual(resized.shape, (4, 8, 3))
        self.assertTrue(bool((resized >= 0).all()))

    def test_exact_topk_budget(self) -> None:
        scores = torch.arange(35, dtype=torch.float32).reshape(5, 7)
        mask = exact_topk_mask(scores, 6)
        self.assertEqual(int(mask.sum().item()), 6)
        fraction = exact_topk_mask(scores, 0.2)
        self.assertEqual(int(fraction.sum().item()), 7)

    def test_straight_through_forward_is_binary_and_backward_exists(self) -> None:
        scores = torch.randn(5, 7, requires_grad=True)
        hard, train = straight_through_topk(scores, 5)
        self.assertTrue(bool(((hard == 0) | (hard == 1)).all()))
        self.assertEqual(int(hard.sum().item()), 5)
        self.assertTrue(torch.equal(train.detach(), hard))
        (train * scores.square()).sum().backward()
        self.assertIsNotNone(scores.grad)
        self.assertGreater(float(scores.grad.abs().sum().item()), 0.0)


class PeerGeneratorTests(unittest.TestCase):
    def test_no_self_attention_and_same_module_filter(self) -> None:
        torch.manual_seed(3)
        layers = make_layers()
        model = PeerStabilizerMatrix(hidden_dim=32)
        out = model(layers, target_index=3, budget=8, straight_through=False)
        self.assertEqual(float(out.attention[3].item()), 0.0)
        self.assertEqual(float(out.attention[1].item()), 0.0)
        self.assertAlmostEqual(float(out.attention.sum().item()), 1.0, places=6)
        self.assertEqual(out.selected_count, 8)

    def test_generated_update_uses_task_gradient_direction(self) -> None:
        torch.manual_seed(4)
        layers = make_layers()
        model = PeerStabilizerMatrix(hidden_dim=32)
        grad = layers[3].gradient
        assert grad is not None
        out = model(layers, 3, 9, task_gradient=grad, straight_through=False)
        updated = one_step_weight(layers[3].weight, grad, out, 0.1, training=False)
        delta = updated - layers[3].weight
        selected = out.hard_mask.bool()
        # Selected deltas must oppose the gradient; unselected coordinates are exact zero.
        self.assertTrue(bool((delta[selected] * grad[selected] <= 0).all()))
        self.assertTrue(torch.equal(delta[~selected], torch.zeros_like(delta[~selected])))

    def test_peer_coordinates_change_generated_scores(self) -> None:
        torch.manual_seed(5)
        layers = make_layers()
        model = PeerStabilizerMatrix(hidden_dim=32)
        out_a = model(layers, 3, 8, straight_through=False)
        permuted = list(layers)
        p = layers[0].weight.reshape(-1)
        perm = torch.randperm(p.numel())
        permuted[0] = LayerView(
            p[perm].reshape_as(layers[0].weight), layers[0].metadata
        )
        out_b = model(permuted, 3, 8, straight_through=False)
        self.assertFalse(torch.allclose(out_a.scores, out_b.scores))

    def test_meta_gradient_reaches_generator(self) -> None:
        torch.manual_seed(7)
        layers = make_layers()
        model = PeerStabilizerMatrix(hidden_dim=24)
        grad = layers[3].gradient
        assert grad is not None
        out = model(layers, 3, 8, straight_through=True)
        updated = one_step_weight(layers[3].weight, grad, out, 0.2, training=True)
        loss = updated.square().mean()
        loss.backward()
        total = sum(
            float(p.grad.abs().sum().item())
            for p in model.parameters()
            if p.grad is not None
        )
        self.assertGreater(total, 0.0)

    def test_structural_regularizer_is_finite(self) -> None:
        layers = make_layers()
        model = PeerStabilizerMatrix(hidden_dim=24)
        out = model(layers, 3, 8, straight_through=False)
        loss = structural_profile_loss(layers[3].weight, out.peer_properties)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_audit_catches_outside_support_change(self) -> None:
        initial = torch.zeros(3, 3)
        mask = torch.zeros(3, 3)
        mask[0, 0] = 1
        current = initial.clone()
        current[0, 0] = 1
        clean = audit_sparse_delta(initial, current, mask)
        self.assertTrue(clean.exact)
        current[2, 2] = 0.1
        bad = audit_sparse_delta(initial, current, mask)
        self.assertFalse(bad.exact)
        self.assertEqual(bad.changed_outside, 1)

    def test_missing_compatible_peer_fails_closed(self) -> None:
        layers = make_layers()
        layers[3] = LayerView(
            layers[3].weight,
            LayerMetadata("target", 3, "unique.module"),
            gradient=layers[3].gradient,
        )
        model = PeerStabilizerMatrix(hidden_dim=24)
        with self.assertRaises(ValueError):
            model(layers, 3, 4)


if __name__ == "__main__":
    unittest.main()
