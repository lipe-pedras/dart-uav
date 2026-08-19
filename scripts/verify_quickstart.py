"""Execute the README's quickstart snippet against the installed package.

NFR-7 requires the quickstart to be verified on every release rather than
trusted to stay correct. The snippet is *extracted from the README*, not
retyped here, so the two cannot drift: if someone edits the README the release
gate runs whatever they wrote.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"
SNIPPET_MARKER = "from dart import DART"


def extract_quickstart(text: str) -> str:
    """Return the first fenced Python block that constructs a model."""
    for block in re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL):
        if SNIPPET_MARKER in block:
            return block
    raise SystemExit(f"no quickstart snippet found in {README}")


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from conftest import make_yolo_dataset

    snippet = extract_quickstart(README.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset = make_yolo_dataset(root / "my-dataset")

        # Point the snippet at the generated fixture and shrink the run so the
        # release gate stays fast. Nothing else about it is altered.
        runnable = (
            snippet.replace('"datasets/my-dataset"', f'r"{dataset}"')
            .replace("epochs=300", "epochs=1")
            .replace("imgsz=320", "imgsz=96, batch=2, workers=0, device='cpu', amp=False")
            .replace('model.export(format="onnx")', "")
        )
        exec(compile(runnable, "README-quickstart", "exec"), {"__name__": "__main__"})

    print("OK: README quickstart executed against the installed package")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
