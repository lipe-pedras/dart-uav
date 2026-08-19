"""End-to-end training, run artifacts, and reproducibility.

These correspond to the CI correctness gate (a one-epoch run on a synthetic
fixture produces the expected artifacts) and the reproducibility gate (two runs
with the same seed and config produce identical metrics).
"""

from __future__ import annotations

import csv
import json

import pytest
import yaml

from dart import DART
from dart.config import RunConfig
from dart.engine import aggregate_runs
from dart.engine.recorder import COMPARISON_FIELDS

TRAIN_KWARGS = dict(
    epochs=1,
    batch=2,
    imgsz=96,
    workers=0,
    target_policy="largest_only",
    device="cpu",
    amp=False,
    seed=0,
)


def train_once(dataset, tmp_path, name="exp", variant="dart-b", **overrides):
    model = DART(variant)
    kwargs = {**TRAIN_KWARGS, **overrides}
    return model, model.train(
        data=str(dataset), project=str(tmp_path / "runs"), name=name, **kwargs
    )


def test_one_epoch_run_produces_every_expected_artifact(yolo_dataset, tmp_path):
    _, result = train_once(yolo_dataset, tmp_path)
    run_dir = result.run_dir

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "results.json").exists()
    assert (run_dir / "weights" / "best.pt").exists()
    assert (run_dir / "weights" / "last.pt").exists()
    assert result.epochs_run == 1


def test_the_snapshotted_config_is_complete_and_reloadable(yolo_dataset, tmp_path):
    """FR-3.4: a run directory must be re-runnable from its own config alone."""
    _, result = train_once(yolo_dataset, tmp_path)
    snapshot = yaml.safe_load((result.run_dir / "config.yaml").read_text())

    assert snapshot["train"]["early_stop"]["patience"] == 40  # never typed by the user
    assert snapshot["data"]["target_policy"] == "largest_only"
    assert snapshot["seed"] == 0
    reloaded = RunConfig.from_yaml(result.run_dir / "config.yaml")
    assert reloaded.model.head.name == "gap"


def test_the_environment_record_explains_where_numbers_came_from(yolo_dataset, tmp_path):
    """FR-7.1, FR-7.2."""
    _, result = train_once(yolo_dataset, tmp_path)
    record = json.loads((result.run_dir / "environment.json").read_text())

    assert record["torch"] and record["python"] and record["platform"]
    assert record["torchmetrics"] and record["supervision"]
    assert record["fingerprint"] == result.fingerprint
    assert record["variant"] == "dart-b"


def test_metrics_csv_has_one_row_per_epoch_in_the_paper_schema(yolo_dataset, tmp_path):
    _, result = train_once(yolo_dataset, tmp_path, epochs=2)
    with (result.run_dir / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    for column in ("val_mAP50", "val_precision", "val_recall", "val_f1", "val_mean_iou"):
        assert column in rows[0]
        assert rows[0][column] != ""


def test_the_comparison_row_carries_the_efficiency_columns(yolo_dataset, tmp_path):
    """FR-4.4, FR-4.5: the columns the paper flags as missing are first-class."""
    _, result = train_once(yolo_dataset, tmp_path)
    with (result.run_dir / "comparison.csv").open(newline="") as handle:
        row = next(iter(csv.DictReader(handle)))

    assert list(row) == COMPARISON_FIELDS
    assert int(row["params"]) > 0
    assert float(row["int8_size_kb"]) > 0
    assert float(row["latency_mean_ms"]) > 0
    assert row["variant"] == "dart-b"


def test_runs_aggregate_into_the_papers_comparison_table(yolo_dataset, tmp_path):
    train_once(yolo_dataset, tmp_path, name="a", variant="dart-b")
    train_once(yolo_dataset, tmp_path, name="b", variant="dart-xd")

    out = aggregate_runs(tmp_path / "runs")
    with out.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {r["variant"] for r in rows} == {"dart-b", "dart-xd"}


def test_a_second_run_never_overwrites_the_first(yolo_dataset, tmp_path):
    _, first = train_once(yolo_dataset, tmp_path, name="exp")
    _, second = train_once(yolo_dataset, tmp_path, name="exp")
    assert first.run_dir != second.run_dir
    assert first.run_dir.exists() and second.run_dir.exists()


# ── Reproducibility (NFR-2) ─────────────────────────────────────────────


def test_same_seed_and_config_give_identical_metrics(yolo_dataset, tmp_path):
    _, first = train_once(yolo_dataset, tmp_path, name="seed_a", epochs=2)
    _, second = train_once(yolo_dataset, tmp_path, name="seed_b", epochs=2)

    assert first.fingerprint == second.fingerprint
    for key, value in first.best_metrics.items():
        assert second.best_metrics[key] == pytest.approx(value, abs=1e-6), key


def test_different_seeds_are_recorded_as_different_configs(yolo_dataset, tmp_path):
    _, first = train_once(yolo_dataset, tmp_path, name="s0", seed=0)
    _, second = train_once(yolo_dataset, tmp_path, name="s1", seed=1)
    assert first.fingerprint != second.fingerprint


# ── Resume (FR-3.3) ─────────────────────────────────────────────────────


def test_training_resumes_from_the_last_checkpoint(yolo_dataset, tmp_path):
    _, first = train_once(yolo_dataset, tmp_path, name="first", epochs=2)
    checkpoint = first.run_dir / "weights" / "last.pt"

    _, resumed = train_once(
        yolo_dataset, tmp_path, name="resumed", epochs=3, resume=str(checkpoint)
    )
    assert resumed.epochs_run == 1  # epoch 3 only
    with (resumed.run_dir / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["epoch"] == "3"


# ── Evaluation and inference ────────────────────────────────────────────


def test_benchmark_only_evaluation_needs_no_trainer(yolo_dataset, tmp_path):
    """FR-4.2."""
    _, result = train_once(yolo_dataset, tmp_path)
    model = DART(str(result.run_dir / "weights" / "best.pt"))
    metrics = model.val(
        data=str(yolo_dataset), target_policy="largest_only", batch=2, workers=0, device="cpu"
    )
    assert 0.0 <= metrics.map50 <= 1.0
    assert 0.0 <= metrics.mean_iou <= 1.0


def test_a_checkpoint_rebuilds_its_own_architecture(yolo_dataset, tmp_path):
    """FR-2.3: the resolved config travels inside the checkpoint."""
    _, result = train_once(yolo_dataset, tmp_path, variant="dart-xd", imgsz=160)
    reloaded = DART(str(result.run_dir / "weights" / "best.pt"))
    assert reloaded.cfg.variant == "dart-xd"
    assert reloaded.cfg.model.imgsz == 160
    assert reloaded.cfg.model.head.name == "decoupled_spp"


def test_predict_returns_boxes_in_source_pixel_coordinates(yolo_dataset, tmp_path):
    _, result = train_once(yolo_dataset, tmp_path)
    model = DART(str(result.run_dir / "weights" / "best.pt"))
    image = next((yolo_dataset / "images" / "val").glob("*.jpg"))

    predictions = model.predict(image, conf=0.0)
    assert len(predictions) == 1
    detection = predictions[0].top
    assert detection is not None
    x1, y1, x2, y2 = detection.box_xyxy
    assert 0 <= x1 <= 64 and 0 <= y2 <= 64 and x2 >= x1 and y2 >= y1


def test_predict_accepts_a_directory(yolo_dataset, tmp_path):
    _, result = train_once(yolo_dataset, tmp_path)
    model = DART(str(result.run_dir / "weights" / "best.pt"))
    assert len(model.predict(yolo_dataset / "images" / "val", conf=0.0)) == 6


# ── Multi-seed (FR-4.3) ─────────────────────────────────────────────────


def test_multi_seed_reports_mean_and_standard_deviation(yolo_dataset, tmp_path):
    model = DART("dart-b")
    aggregate = model.train_seeds(
        [0, 1],
        data=str(yolo_dataset),
        project=str(tmp_path / "runs"),
        name="multiseed",
        **{k: v for k, v in TRAIN_KWARGS.items() if k != "seed"},
    )
    assert aggregate.seeds == [0, 1]
    assert "map50" in aggregate.mean and "map50" in aggregate.std
    assert len(aggregate.runs) == 2
    assert len({str(r.run_dir) for r in aggregate.runs}) == 2
