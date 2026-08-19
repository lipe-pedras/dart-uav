"""Windows/macOS portability obligations (NFR-3).

These run on every platform: the point is that the *code* satisfies the
obligations, not that the current runner happens to use ``fork``.
"""

from __future__ import annotations

import ast
import pickle
from pathlib import Path

import pytest

import dart
from dart.data import build_dataloader, check_spawn_safety
from dart.data.augment import AugmentationPipeline
from dart.errors import SpawnSafetyError

SOURCE_ROOT = Path(dart.__file__).parent
SOURCE_FILES = sorted(SOURCE_ROOT.rglob("*.py"))


def test_no_source_file_concatenates_filesystem_paths_by_hand():
    """All path handling goes through pathlib, per NFR-3."""
    offenders = []
    for path in SOURCE_FILES:
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "os.path.join" in line or "os.sep" in line:
                offenders.append(f"{path.name}:{lineno}")
            # A literal separator inside a path-looking string.
            if '"/"' in line and "split" not in line and "partition" not in line:
                offenders.append(f"{path.name}:{lineno}")
    assert not offenders, f"hand-built paths found: {offenders}"


def test_the_augmentation_pipeline_holds_no_lambdas_or_closures():
    """Everything crossing the worker boundary must pickle."""
    from dart.config import AugmentConfig

    pipeline = AugmentationPipeline.from_config(AugmentConfig())
    restored = pickle.loads(pickle.dumps(pipeline))
    assert len(restored) == len(pipeline)
    assert [type(t).__name__ for t in restored.transforms] == [
        type(t).__name__ for t in pipeline.transforms
    ]


def test_no_lambda_is_stored_as_an_attribute():
    """A lambda assigned to ``self`` is exactly what breaks spawn."""
    offenders = []
    for path in SOURCE_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Lambda):
                continue
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"lambdas stored as attributes: {offenders}"


def test_zero_workers_never_triggers_the_spawn_check(yolo_dataset):
    from dart.config import ConfigResolver
    from dart.data import DatasetLoader

    cfg = ConfigResolver().resolve("dart-b", data=str(yolo_dataset), target_policy="largest_only")
    dataset = DatasetLoader().load(cfg.data, cfg.model, verbose=False)
    loader = build_dataloader(dataset, batch_size=2, shuffle=False, workers=0)
    assert loader.num_workers == 0
    images, conf, boxes = next(iter(loader))
    assert images.shape[0] == 2 and conf.shape == (2, 1) and boxes.shape == (2, 1, 4)


def test_an_unguarded_script_is_refused_with_the_fix_named(tmp_path, monkeypatch):
    script = tmp_path / "unguarded.py"
    script.write_text("from dart import DART\nDART('dart-b').train(data='x')\n")

    class FakeMain:
        __file__ = str(script)

    monkeypatch.setitem(__import__("sys").modules, "__main__", FakeMain())
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr("multiprocessing.get_start_method", lambda allow_none=True: "spawn")

    with pytest.raises(SpawnSafetyError) as excinfo:
        check_spawn_safety(workers=4)
    message = str(excinfo.value)
    assert "__name__ == '__main__'" in message
    assert "workers=0" in message


def test_a_guarded_script_passes(tmp_path, monkeypatch):
    script = tmp_path / "guarded.py"
    script.write_text(
        "from dart import DART\n\n"
        "def main():\n    DART('dart-b').train(data='x')\n\n"
        "if __name__ == '__main__':\n    main()\n"
    )

    class FakeMain:
        __file__ = str(script)

    monkeypatch.setitem(__import__("sys").modules, "__main__", FakeMain())
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr("multiprocessing.get_start_method", lambda allow_none=True: "spawn")

    check_spawn_safety(workers=4)  # must not raise


def test_the_packages_own_entry_point_is_guarded():
    source = (SOURCE_ROOT / "cli.py").read_text(encoding="utf-8")
    assert '__name__ == "__main__"' in source
