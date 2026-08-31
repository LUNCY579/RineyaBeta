from pathlib import Path

import pytest

from dast.smoke import run_smoke


@pytest.mark.integration
def test_end_to_end_smoke(tmp_path: Path):
    report = run_smoke(tmp_path)
    assert report["status"] == "PASS"
    assert report["training"]["finite_loss"]
    assert report["training"]["projector_updated"]
    assert report["feedback"]["feedback_logits_changed"]
    assert (tmp_path / "smoke_report.json").is_file()
