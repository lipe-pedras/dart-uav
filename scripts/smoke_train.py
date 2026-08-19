"""One-epoch training run on a generated fixture dataset.

Used by the CI correctness gate: proves that a fresh install can train and that
the expected artifacts appear. Also serves as a copy-pasteable example of the
entry-point guard every user script needs on Windows (NFR-3).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from conftest import make_yolo_dataset  # noqa: E402

from dart import DART  # noqa: E402

EXPECTED = [
    Path("config.yaml"),
    Path("environment.json"),
    Path("metrics.csv"),
    Path("results.json"),
    Path("comparison.csv"),
    Path("weights/best.pt"),
    Path("weights/last.pt"),
]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset = make_yolo_dataset(root / "fixture")

        model = DART("dart-xd")
        result = model.train(
            data=str(dataset),
            epochs=1,
            batch=2,
            imgsz=96,
            workers=0,
            device="cpu",
            amp=False,
            seed=0,
            target_policy="largest_only",
            project=str(root / "runs"),
            name="smoke",
        )

        missing = [str(p) for p in EXPECTED if not (result.run_dir / p).exists()]
        if missing:
            print(f"FAIL: missing artifacts: {missing}", file=sys.stderr)
            return 1

        metrics = model.val(data=str(dataset), target_policy="largest_only", workers=0)
        print(f"OK: trained 1 epoch, evaluated: {metrics}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
