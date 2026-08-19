# Reproducing the paper

The paper reports a grid of 7 variants × 4 resolutions across 5 datasets. This
page shows how to regenerate that grid, and its comparison table, from a clean
install.

## What is and is not reproducible from this package

**Reproducible:** the architecture of all seven variants, the training
procedure (schedule, augmentation, compound early stopping), the metric
definitions, the efficiency measurements, and the comparison-table schema.

**Not shipped here:** the five datasets themselves, and the trained weights.
Weights live on the Hugging Face Hub and are resolved by
`DART("repo_id:file.pt")`. Datasets are the authors' or third parties' to
distribute.

**Environment-dependent:** absolute latency and FPS. Everything else should
match within floating-point tolerance on the same hardware and software; the
`environment.json` written into every run directory is what makes a
cross-environment discrepancy explainable.

## Training from scratch is the v1 default

The paper's Table I comes from randomly-initialised models. Nothing in the
default configuration loads pretrained weights, so a plain `train()` call
reproduces the published setup. Loading an externally-supplied backbone is
supported (`pretrained=`) but is not what the paper measured.

## One cell of the grid

```python
from dart import DART

model = DART("dart-xd", imgsz=320)
model.train(
    data="datasets/medicalkit-drone-dataset",
    target_class=1,               # 0 = landing base, 1 = package
    target_policy="largest_only",
    epochs=350,
    batch=32,
    lr=1e-3,
    patience=40,
    seed=0,
    project="runs",
    name="dart-xd-320-medicalkit",
)
```

Note the two explicit keys. `medicalkit-drone-dataset` annotates two classes,
so DART refuses to guess which one is the target; and images can carry more
than one annotation, so it refuses to guess which one to keep. Both refusals
are deliberate — they are the difference between a documented experimental
choice and an undocumented one.

## The whole grid

```python
from pathlib import Path
from dart import DART

VARIANTS = ["dart-b", "dart-b-ciou", "dart-s", "dart-d", "dart-f", "dart-x", "dart-xd"]
RESOLUTIONS = [96, 160, 224, 320]
DATASETS = {
    "medicalkit-drone-dataset": dict(target_class=1),
    "targetcross-dataset": {},
    "imav-2025-gate-dataset": {},
    "imav-2025-platform-dataset": {},
    "sae-2026-manometer": {},
}


def main() -> None:
    for dataset, extra in DATASETS.items():
        for variant in VARIANTS:
            for imgsz in RESOLUTIONS:
                DART(variant, imgsz=imgsz).train(
                    data=f"datasets/{dataset}",
                    target_policy="largest_only",
                    epochs=350,
                    batch=32,
                    patience=40,
                    seed=0,
                    project=f"runs/{dataset}",
                    name=f"{variant}-{imgsz}",
                    **extra,
                )


if __name__ == "__main__":     # required for DataLoader workers on Windows
    main()
```

That is 140 runs. Each writes its own directory; none overwrites another.

## The comparison table

Every run drops a `comparison.csv` holding its single row. Concatenate them:

```bash
dart report runs=runs out=model_comparison.csv
```

```python
from dart.engine import aggregate_runs
aggregate_runs("runs", "model_comparison.csv")
```

Columns:

```
variant, dataset, seed, imgsz, best_epoch, total_epochs,
val_mAP50, val_mAP50_95, val_precision, val_recall, val_f1, val_mean_iou,
val_loss, val_best_thresh, val_best_f1,
params, int8_size_kb, latency_mean_ms, latency_p95_ms, fps, device,
fingerprint
```

The first block matches the paper's own `model_comparison.csv` schema, so
existing analysis scripts keep working. The efficiency block is what the paper
flags as pending. `fingerprint` ties each row back to the exact configuration
that produced it — two rows with the same fingerprint came from the same
scientific configuration, regardless of which folder they were written to.

The representative epoch of a run is the one with the highest `val_mAP50`,
breaking ties toward the lower `val_loss` — the same rule the research
codebase's `compare_models.py` used.

## Variance across seeds

The paper reports single-seed numbers. To report mean ± standard deviation
instead:

```python
result = DART("dart-xd", imgsz=320).train_seeds(
    [0, 1, 2],
    data="datasets/targetcross-dataset",
    target_policy="largest_only",
    epochs=350,
)
print(result.format())
```

Each seed gets its own run directory (`<name>_seed<k>`), so the individual runs
stay inspectable and the aggregate is derived, not primary.

## Benchmark-only re-evaluation

To re-measure a published checkpoint without retraining — the situation that
produced the paper's YOLOv11n re-evaluation disclosure:

```bash
dart val model=runs/dart-xd-320/weights/best.pt \
         data=datasets/targetcross-dataset target_policy=largest_only
```

The checkpoint carries its own resolved config, so the architecture, the
resolution, and the preprocessing are reconstructed from the file rather than
retyped — which is what stops a re-evaluation from silently measuring a
different model than the one that was trained.

## Deployment measurements

`dart profile` reports parameters, INT8 size estimate, and latency on the
machine it runs on. For the paper's embedded targets, export first and measure
with ONNX Runtime on the device:

```bash
dart export model=runs/dart-xd-320/weights/best.pt format=onnx path=dart-xd-320.onnx
dart profile model=runs/dart-xd-320/weights/best.pt imgsz=320    # host-side reference
```

Export fuses RepConv automatically and verifies the artifact against the
PyTorch model before returning it, so what you benchmark on the device is
numerically the model you trained.
