from __future__ import annotations

import unittest

import torch

from experiments.cross_architecture_graft.src.donors import (
    DonorSpec,
    LatentKVAttentionBranch,
    NoPEAttentionBranch,
    RoutedMoEBranch,
    build_donor,
)


class DonorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def _assert_causal(self, module) -> None:
        module.eval()
        first = torch.randn(2, 7, 16)
        second = first.clone()
        second[:, 4:, :] = torch.randn_like(second[:, 4:, :]) * 10
        with torch.no_grad():
            y_first = module(first)
            y_second = module(second)
        torch.testing.assert_close(y_first[:, :4], y_second[:, :4], rtol=0, atol=0)

    def test_nope_attention_is_strictly_causal(self) -> None:
        self._assert_causal(NoPEAttentionBranch(16, num_heads=4, num_kv_heads=2))

    def test_latent_kv_attention_is_strictly_causal(self) -> None:
        self._assert_causal(
            LatentKVAttentionBranch(
                16,
                num_heads=4,
                num_kv_heads=2,
                latent_rank=4,
                q_rank=4,
            )
        )

    def test_padding_mask_blocks_masked_keys(self) -> None:
        module = NoPEAttentionBranch(16, num_heads=4, num_kv_heads=2).eval()
        x = torch.randn(1, 5, 16)
        altered = x.clone()
        altered[:, 3:, :] += 1000
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        with torch.no_grad():
            first = module(x, attention_mask=mask)
            second = module(altered, attention_mask=mask)
        torch.testing.assert_close(first[:, :3], second[:, :3], rtol=0, atol=0)

    def test_routed_moe_shape_and_finiteness(self) -> None:
        module = RoutedMoEBranch(
            16,
            intermediate_size=8,
            num_experts=4,
            top_k=2,
            shared_expert=True,
        )
        output = module(torch.randn(3, 5, 16))
        self.assertEqual(tuple(output.shape), (3, 5, 16))
        self.assertTrue(torch.isfinite(output).all())

    def test_registry_builds_every_supported_mechanism(self) -> None:
        specs = [
            DonorSpec(kind="swiglu_mlp", d_model=16, intermediate_size=8),
            DonorSpec(kind="nope_attention", d_model=16, num_heads=4, num_kv_heads=2),
            DonorSpec(
                kind="latent_kv_attention",
                d_model=16,
                num_heads=4,
                num_kv_heads=2,
                latent_rank=4,
            ),
            DonorSpec(
                kind="routed_moe",
                d_model=16,
                intermediate_size=8,
                num_experts=3,
                top_k=2,
            ),
        ]
        for spec in specs:
            with self.subTest(kind=spec.kind):
                output = build_donor(spec)(torch.randn(2, 4, 16))
                self.assertEqual(tuple(output.shape), (2, 4, 16))


if __name__ == "__main__":
    unittest.main()
