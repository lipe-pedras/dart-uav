"""The command-line interface (FR-6.2): same four operations as the API."""

from __future__ import annotations

import pytest

from dart.cli import main, parse_args


def test_key_value_arguments_are_typed():
    parsed = parse_args(["epochs=300", "lr=0.001", "amp=false", "model=dart-xd", "resume=none"])
    assert parsed == {
        "epochs": 300,
        "lr": 0.001,
        "amp": False,
        "model": "dart-xd",
        "resume": None,
    }


def test_list_arguments_parse():
    assert parse_args(["seeds=[0,1,2]"])["seeds"] == [0, 1, 2]


def test_dotted_config_paths_pass_through_verbatim():
    assert parse_args(["train.early_stop.patience=25"]) == {"train.early_stop.patience": 25}


def test_malformed_arguments_are_rejected():
    with pytest.raises(ValueError, match="key=value"):
        parse_args(["--epochs", "300"])


def test_help_and_variants_commands(capsys):
    assert main([]) == 0
    assert "dart train" in capsys.readouterr().out

    assert main(["variants"]) == 0
    listed = capsys.readouterr().out
    for variant in ("dart-b", "dart-xd"):
        assert variant in listed


def test_unknown_command_exits_nonzero(capsys):
    assert main(["frobnicate"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_dart_errors_are_reported_without_a_traceback(capsys):
    assert main(["train", "model=dart-nonexistent"]) == 1
    assert "unknown DART variant" in capsys.readouterr().err


def test_train_val_export_round_trip(yolo_dataset, tmp_path, capsys):
    common = [
        f"data={yolo_dataset}",
        "epochs=1",
        "batch=2",
        "imgsz=96",
        "workers=0",
        "device=cpu",
        "amp=false",
        "target_policy=largest_only",
    ]
    assert main(["train", "model=dart-b", f"project={tmp_path / 'runs'}", "name=cli", *common]) == 0

    checkpoint = tmp_path / "runs" / "cli" / "weights" / "best.pt"
    assert checkpoint.exists()

    assert main(["val", f"model={checkpoint}", *common]) == 0
    assert "map50" in capsys.readouterr().out

    pytest.importorskip("onnxruntime")
    artifact = tmp_path / "cli.onnx"
    assert main(["export", f"model={checkpoint}", "format=onnx", f"path={artifact}"]) == 0
    assert artifact.exists()


def test_predict_command(yolo_dataset, tmp_path, capsys):
    image = next((yolo_dataset / "images" / "val").glob("*.jpg"))
    assert main(["predict", f"source={image}", "model=dart-b", "conf=0.0"]) == 0
    assert "conf=" in capsys.readouterr().out


def test_profile_command_reports_efficiency(capsys):
    assert main(["profile", "model=dart-b", "imgsz=96"]) == 0
    output = capsys.readouterr().out
    assert "params" in output and "latency" in output


def test_report_command_aggregates_runs(yolo_dataset, tmp_path, capsys):
    main(
        [
            "train",
            "model=dart-b",
            f"data={yolo_dataset}",
            f"project={tmp_path / 'runs'}",
            "name=r1",
            "epochs=1",
            "batch=2",
            "imgsz=96",
            "workers=0",
            "device=cpu",
            "amp=false",
            "target_policy=largest_only",
        ]
    )
    out = tmp_path / "model_comparison.csv"
    assert main(["report", f"runs={tmp_path / 'runs'}", f"out={out}"]) == 0
    assert "variant" in out.read_text()
