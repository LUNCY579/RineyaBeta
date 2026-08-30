from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

from peer_features import LayerMetadata, LayerView  # noqa: E402
from phase_c_controls import (  # noqa: E402
    CompactPlan,
    apply_first_adamw_step,
    coordinate_shuffled_views,
    layer_shuffled_views,
    support_from_scores,
)
from phase_c_report import summarize  # noqa: E402
from phase_c_splits import index_sources, select_rows  # noqa: E402


class PhaseCSplitTests(unittest.TestCase):
    def _row(self, index: int, family: str) -> dict:
        return {
            "id": f"row-{index}",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": f"Question {index}?"},
                {"role": "assistant", "content": f"Answer {index}."},
            ],
            "metadata": {"template_family": family},
        }

    def test_grouped_split_and_selection_are_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source.jsonl"
            rows = [self._row(i, f"family-{i}") for i in range(80)]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            indexed, audit = index_sources(
                ["source.jsonl"],
                root=root,
                group_fields=["metadata.template_family"],
                split_seed=11,
                ratios={
                    "META_TRAIN": 0.50,
                    "MASK_CAL": 0.20,
                    "ONE_STEP_DEV": 0.20,
                    "SEALED": 0.10,
                },
            )
            self.assertEqual(audit["family_overlap"], 0)
            self.assertEqual(audit["normalized_prompt_overlap"], 0)
            meta = select_rows(indexed, split="META_TRAIN", count=5, seed=3)
            dev = select_rows(indexed, split="ONE_STEP_DEV", count=5, seed=3)
            self.assertTrue({row.family for row in meta}.isdisjoint({row.family for row in dev}))


class PhaseCControlTests(unittest.TestCase):
    def _views(self) -> list[LayerView]:
        views = []
        for depth in range(4):
            weight = torch.arange(24, dtype=torch.float32).reshape(4, 6) + depth * 100
            gradient = torch.flip(weight, dims=[1])
            views.append(
                LayerView(
                    weight=weight,
                    gradient=gradient,
                    metadata=LayerMetadata(
                        name=f"layer-{depth}", depth=depth, module_type="mlp.down_proj"
                    ),
                )
            )
        return views

    def test_coordinate_shuffle_preserves_peer_value_multisets_and_target(self) -> None:
        views = self._views()
        transformed, ledger = coordinate_shuffled_views(views, target_index=1, seed=19)
        self.assertTrue(torch.equal(transformed[1].weight, views[1].weight))
        self.assertEqual(ledger["kind"], "coordinate_shuffle")
        for index in (0, 2, 3):
            self.assertTrue(
                torch.equal(
                    torch.sort(transformed[index].weight.reshape(-1)).values,
                    torch.sort(views[index].weight.reshape(-1)).values,
                )
            )
            self.assertTrue(
                torch.equal(
                    torch.sort(transformed[index].gradient.reshape(-1)).values,
                    torch.sort(views[index].gradient.reshape(-1)).values,
                )
            )

    def test_layer_shuffle_is_deranged_and_target_stays_real(self) -> None:
        views = self._views()
        transformed, ledger = layer_shuffled_views(views, target_index=2, seed=7)
        self.assertTrue(torch.equal(transformed[2].weight, views[2].weight))
        mapping = {int(k): int(v) for k, v in ledger["mapping"].items()}
        self.assertTrue(all(dst != src for dst, src in mapping.items()))

    def test_indexed_first_adam_step_moves_only_selected_coordinates(self) -> None:
        class Fake(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(2, 3))

        model = Fake()
        scores = torch.tensor([[10.0, 9.0, 0.0], [0.0, 0.0, 0.0]])
        support = support_from_scores(
            parameter_name="weight",
            depth=0,
            scores=scores,
            budget=2,
            scales=1.0,
        )
        plan = CompactPlan(
            arm="test",
            budget=2,
            module_path="fake",
            supports=[support],
            transform={"kind": "test"},
        )
        gradient = {"weight": torch.ones_like(model.weight)}
        audit = apply_first_adamw_step(
            model,
            plan,
            gradient,
            lr=0.1,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        self.assertEqual(audit["changed_outside"], 0)
        self.assertEqual(int((model.weight != 0).sum().item()), 2)


class PhaseCDecisionTests(unittest.TestCase):
    def test_advancement_requires_both_causal_controls(self) -> None:
        seeds = [1, 2, 3, 4, 5]
        arms = [
            "real",
            "shuffle",
            "layer",
        ]
        trials = []
        for seed in seeds:
            for arm, post in (("real", 0.90), ("shuffle", 1.00), ("layer", 0.98)):
                trials.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "valid": True,
                        "invalid_reasons": [],
                        "pre_dev_loss": 1.10,
                        "post_dev_loss": post,
                    }
                )
        summary = summarize(
            trials,
            seeds=seeds,
            required_arms=arms,
            real_arm="real",
            shuffled_arm="shuffle",
            layer_control_arm="layer",
            min_win_fraction=0.80,
        )
        self.assertEqual(summary["status"], "ADVANCE_TO_MATCHED_SFT")


if __name__ == "__main__":
    unittest.main()
