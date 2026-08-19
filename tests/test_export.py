"""ONNX export, automatic fusion, and output verification (FR-5.1 – FR-5.3).

The verification test is the CI reproducibility gate: the exported model's
output must match the PyTorch model's on a fixed input, within tolerance.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dart import DART
from dart.config import ConfigResolver
from dart.errors import ExportError, ExportVerificationError
from dart.export import (
    ExportBackend,
    ExportManager,
    VerifyReport,
    available_formats,
    register_backend,
)
from dart.models import DARTDetector

onnxruntime = pytest.importorskip("onnxruntime")


@pytest.mark.parametrize("variant", ["dart-b", "dart-xd"])
def test_export_produces_an_artifact_matching_the_pytorch_model(variant, tmp_path):
    model = DART(variant, imgsz=96)
    artifact = model.export(format="onnx", path=tmp_path / f"{variant}.onnx")

    assert artifact.exists() and artifact.suffix == ".onnx"
    report = ExportManager().verify(model.model, artifact, fmt="onnx", imgsz=96)
    assert report.passed, report.message
    assert report.max_abs_diff < 1e-4


def test_export_fuses_repconv_automatically(tmp_path):
    """FR-5.3: the user never has to call a separate fusion step."""
    model = DART("dart-b", imgsz=96)
    assert not model.model.is_fused

    session_before = _session_outputs(model.export(format="onnx", path=tmp_path / "m.onnx"), 96)
    # The exported graph is the fused one, while the caller's model stays
    # trainable — export works on a copy.
    assert not model.model.is_fused
    assert session_before[0].shape == (1, 1, 1)


def test_exported_graph_has_the_documented_input_and_output_names(tmp_path):
    model = DART("dart-xd", imgsz=96)
    artifact = model.export(format="onnx", path=tmp_path / "m.onnx")
    session = onnxruntime.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
    assert [i.name for i in session.get_inputs()] == ["images"]
    assert [o.name for o in session.get_outputs()] == ["conf", "boxes"]


def test_dynamic_batch_axis_lets_one_artifact_serve_any_batch_size(tmp_path):
    model = DART("dart-b", imgsz=96)
    artifact = model.export(format="onnx", path=tmp_path / "m.onnx")
    session = onnxruntime.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
    for batch in (1, 4):
        conf, boxes = session.run(None, {"images": torch.randn(batch, 3, 96, 96).numpy()})
        assert conf.shape == (batch, 1, 1) and boxes.shape == (batch, 1, 4)


@pytest.mark.parametrize("imgsz", [96, 320])
def test_export_honours_the_configured_resolution(imgsz, tmp_path):
    model = DART("dart-x", imgsz=imgsz)
    artifact = model.export(format="onnx", path=tmp_path / "m.onnx")
    session = onnxruntime.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape[2:] == [imgsz, imgsz]


def test_a_diverging_artifact_raises_rather_than_shipping(tmp_path):
    """A silently-wrong artifact is worse than a failed export."""

    class LyingBackend(ExportBackend):
        name = "lying"
        suffix = ".bin"

        def export(self, model, path, **opts):
            path.write_bytes(b"not a model")
            return path

        def verify(self, model, artifact, **opts):
            return VerifyReport(passed=False, max_abs_diff=1.0, message="outputs diverge")

    manager = ExportManager(backend=LyingBackend())
    model = DARTDetector(ConfigResolver().resolve("dart-b").model)
    with pytest.raises(ExportVerificationError, match="do not deploy"):
        manager.export(model, fmt="lying", path=tmp_path / "m.bin")


def test_verification_is_skipped_loudly_not_silently(tmp_path, monkeypatch):
    class NoRuntimeBackend(ExportBackend):
        name = "noruntime"
        suffix = ".bin"

        def export(self, model, path, **opts):
            path.write_bytes(b"artifact")
            return path

        def verify(self, model, artifact, **opts):
            return VerifyReport(passed=True, skipped=True, message="runtime not installed")

    manager = ExportManager(backend=NoRuntimeBackend())
    model = DARTDetector(ConfigResolver().resolve("dart-b").model)
    artifact = manager.export(model, fmt="noruntime", path=tmp_path / "m.bin")
    assert artifact.exists()


def test_a_new_backend_needs_no_change_to_the_public_signature(tmp_path):
    """FR-5.2, realised structurally."""

    class DummyBackend(ExportBackend):
        name = "dummy"
        suffix = ".dummy"

        def export(self, model, path, **opts):
            path.write_text("dummy")
            return path

        def verify(self, model, artifact, **opts):
            return VerifyReport(passed=True, message="trivially equal")

    register_backend(DummyBackend())
    assert "dummy" in available_formats()

    model = DART("dart-b")
    artifact = model.export(format="dummy", path=tmp_path / "m.dummy")
    assert artifact.read_text() == "dummy"


def test_unknown_format_lists_the_registered_ones():
    model = DART("dart-b")
    with pytest.raises(ExportError, match="Available"):
        model.export(format="nonexistent")


def _session_outputs(artifact: Path, imgsz: int):
    session = onnxruntime.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
    return session.run(None, {"images": torch.randn(1, 3, imgsz, imgsz).numpy()})
