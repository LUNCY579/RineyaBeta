from __future__ import annotations

from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

from peer_stabilizer import PeerStabilizerMatrix, audit_sparse_delta  # noqa: E402
from rineya_integration import (  # noqa: E402
    apply_sparse_plan,
    build_sparse_plan,
    load_sparse_plan,
    save_sparse_plan,
)


class FakeMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_proj = nn.Linear(6, 5, bias=False)


class FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = FakeMLP()


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FakeBlock() for _ in range(4)])


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeBackbone()
        self.unrelated = nn.Parameter(torch.ones(3))


class IntegrationTests(unittest.TestCase):
    def make_model_with_gradients(self) -> FakeModel:
        torch.manual_seed(19)
        model = FakeModel()
        for parameter in model.parameters():
            parameter.grad = torch.randn_like(parameter)
        return model

    def test_build_save_load_apply_and_train(self) -> None:
        model = self.make_model_with_gradients()
        generator = PeerStabilizerMatrix(hidden_dim=24)
        plan = build_sparse_plan(
            model,
            generator,
            module_paths=["mlp.down_proj"],
            budget=5,
        )
        self.assertEqual(len(plan.entries), 4)
        self.assertEqual(plan.selected, 20)
        self.assertEqual(len(plan.manifest()["entries"][0]["mask_sha256"]), 64)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.pt"
            save_sparse_plan(plan, path)
            loaded = load_sparse_plan(path)
        self.assertEqual(loaded.selected, plan.selected)
        self.assertTrue(torch.equal(loaded.entries[0].mask, plan.entries[0].mask))

        initial = {
            name: parameter.detach().clone() for name, parameter in model.named_parameters()
        }
        optimizer = apply_sparse_plan(model, loaded, lr=0.03, weight_decay=0.02)
        self.assertFalse(model.unrelated.requires_grad)
        for entry in loaded.entries:
            self.assertTrue(dict(model.named_parameters())[entry.parameter_name].requires_grad)

        # Simulate one SFT backward result on exactly the planned matrices.
        for entry in loaded.entries:
            parameter = dict(model.named_parameters())[entry.parameter_name]
            parameter.grad = torch.full_like(parameter, 0.2)
        optimizer.step()

        for entry in loaded.entries:
            parameter = dict(model.named_parameters())[entry.parameter_name]
            audit = audit_sparse_delta(initial[entry.parameter_name], parameter, entry.mask)
            self.assertTrue(audit.exact)
            self.assertEqual(audit.changed_inside, 5)
        self.assertTrue(torch.equal(model.unrelated.detach(), initial["unrelated"]))

    def test_missing_calibration_gradient_fails(self) -> None:
        model = FakeModel()
        generator = PeerStabilizerMatrix(hidden_dim=24)
        with self.assertRaises(RuntimeError):
            build_sparse_plan(
                model,
                generator,
                module_paths=["mlp.down_proj"],
                budget=3,
            )

    def test_package_import_path_is_supported(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import src; "
                    "from src.rineya_integration import SparsePlan; "
                    "assert src.PeerStabilizerMatrix and SparsePlan"
                ),
            ],
            cwd=HERE.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
