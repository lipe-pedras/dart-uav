# DART

**A family of lightweight CNNs for real-time detection in resource-constrained
UAV navigation.**

DART packages the model family, training pipeline, and evaluation methodology
of the DART paper (SBR 2026) into an installable Python package. It is what a
reviewer uses to reproduce the paper's numbers from a clean install, and what a
practitioner uses to train a DART variant on their own single-class detection
task and deploy it to a Raspberry Pi or a Jetson.

```bash
pip install dart-uav
```

The distribution is `dart-uav`; the import namespace is `dart`.

---

## Quickstart

Nine lines, from nothing to a trained detector:

```python
from dart import DART

model = DART("dart-xd")
model.train(
    data="datasets/my-dataset",   # YOLO, COCO, VOC, or CreateML
    epochs=300,
    imgsz=320,
    target_policy="largest_only", # required: how many boxes per image to keep
    seed=0,
)
metrics = model.val(data="datasets/my-dataset")
model.export(format="onnx")
```

Same thing from the shell:

```bash
dart train data=datasets/my-dataset model=dart-xd epochs=300 imgsz=320 \
           target_policy=largest_only seed=0
dart val    data=datasets/my-dataset model=runs/exp/weights/best.pt
dart export model=runs/exp/weights/best.pt format=onnx
```

Every run writes a self-contained directory:

```
runs/exp/
    config.yaml        the fully-resolved config, defaults included
    environment.json   package/dependency versions, hardware, fingerprint
    metrics.csv        one row per epoch
    results.json       best epoch, stop reason, efficiency profile
    comparison.csv     this run's row in the paper's comparison schema
    weights/best.pt    best checkpoint
    weights/last.pt    resume point
```

Re-running `config.yaml` reproduces the run without knowing which package
version produced it — that is the point of snapshotting the *resolved* config
rather than the command line.

---

## The seven variants

All seven share one backbone sequence (RepConv → InvertedResidual →
InvertedResidual → Ghost) and differ only along the axes below. Each is a YAML
file in `dart/config/presets/`, not a class — an eighth variant is a file.

| Variant | Attention | Activation | Neck | Head | Loss |
|---|---|---|---|---|---|
| `dart-b` | SE | ReLU6 | — | GAP | BCE + Smooth L1 |
| `dart-b-ciou` | SE | ReLU6 | — | GAP | Focal + CIoU |
| `dart-s` | SE | ReLU6 | — | SPP-Lite | Focal + CIoU |
| `dart-d` | SE | ReLU6 | — | Decoupled SPP | Focal + CIoU |
| `dart-f` | SE | ReLU6 | FPN-Lite | GAP | Focal + CIoU |
| `dart-x` | Coord | SiLU | FPN-Lite | SPP-Lite | Focal + CIoU |
| `dart-xd` | Coord | SiLU | FPN-Lite | Decoupled SPP | Focal + CIoU |

`dart-b` and `dart-b-ciou` differ by exactly two lines, which is what makes the
paper's loss-only ablation self-evident from the configs themselves.

Supported input resolutions: 96, 160, 224, 320 (`imgsz=`). The backbone width
multiplier (`width=`) scales every block's channels — 1.0 gives the paper's
16 → 32 → 48 → 64.

```bash
dart variants          # list the presets
dart profile model=dart-xd imgsz=320
```

---

## Datasets: explicit or nothing

DART refuses to guess what your annotations mean. Two configuration keys have
**no default**, and a dataset that needs one and does not have it is an error,
not a silent adaptation:

`data.target_policy`
: What to do with an image carrying more than one annotation.
  `largest_only` (the paper's convention) or `keep_all`.

`data.target_class`
: Which class is the detection target when the dataset annotates several.
  Everything else becomes background.

```
ConfigurationError: dataset at datasets/medicalkit/images/train annotates 2
classes (0 (landing-base), 1 (package)) but the model is configured
single-class. DART will not guess which class is the target: set
'data.target_class' to the class id you want to detect. Every other class then
becomes background.
```

Validation runs on the raw annotations *before* any policy is applied and
before the first epoch, so a misconfigured dataset costs seconds, not an epoch
of compute.

Supported layouts, auto-detected unless you set `format=`:

| Format | Annotations |
|---|---|
| `yolo` | `labels/<split>/*.txt` plus a `data.yaml` |
| `coco` | one `*.json` with `images` and `annotations` |
| `voc` | per-image `*.xml` |
| `createml` | one `*.json` array of records |

Directory layouts understood: `<root>/images/<split>` + `<root>/labels/<split>`,
`<root>/<split>/images` + `<root>/<split>/labels`, a single `<root>/<split>`,
or an Ultralytics-style dataset YAML naming `path` / `train` / `val`.

---

## Training

```python
model.train(
    data="datasets/my-dataset",
    epochs=300, batch=32, lr=1e-3, imgsz=320,
    target_policy="largest_only",
    patience=40,
    seed=0,
    project="runs", name="dart-xd-320",
)
```

**Early stopping** is fully configurable. The primary metric drives patience;
an optional secondary metric can veto a stop, but only once the primary has
saturated — otherwise a wandering secondary would keep a dead run alive. The
paper's criterion is the default:

```yaml
train:
  early_stop:
    primary_metric: map50
    direction: max
    patience: 40
    min_delta: 0.001
    secondary_metric: mean_iou     # set to null for plain patience stopping
    secondary_rel_gain: 0.03
    secondary_window: 3
```

**Multiple seeds** (mean ± std per metric):

```python
result = model.train_seeds([0, 1, 2], data="datasets/my-dataset",
                           target_policy="largest_only")
print(result.format())
```

**Resume** from the last epoch boundary:

```bash
dart train data=datasets/my-dataset resume=runs/exp/weights/last.pt
```

---

## Metrics

Computed with [`torchmetrics`](https://lightning.ai/docs/torchmetrics/), not by
bespoke code. That is a scientific-integrity decision: numbers in a
peer-reviewed paper should come from code the community has already validated.

| Metric | Meaning |
|---|---|
| `map50` | mAP at IoU 0.5 |
| `map50_95` | COCO-style mAP, IoU 0.5:0.05:0.95 |
| `precision` | Of the announcements, how many had a correct detection available |
| `recall` | Of the objects present, how many were announced *and* well-localised |
| `f1` | Harmonic mean of the two |
| `mean_iou` | Mean IoU over ground-truth-positive images |

A confident but badly-localised box is a false positive, and poor localisation
cannot inflate recall — the denominator is the total ground-truth positive
count.

Efficiency metrics (parameters, INT8 size, latency, FPS) are part of the same
schema, measured on the **fused** model, because the multi-branch training
graph is not what deploys.

Regenerate the paper's comparison table from a set of runs:

```bash
dart report runs=runs out=model_comparison.csv
```

---

## Export

```python
model.export(format="onnx", path="dart-xd.onnx")
```

RepConv reparameterization (training-time three branches → inference-time
single 3×3 conv) happens automatically before export; there is no separate
fusion step to forget. The artifact is then verified against the PyTorch model
on a fixed input, and export **fails** rather than shipping something
numerically different.

ONNX is the only v1 backend, because ONNX Runtime covers all three
paper-validated targets (Raspberry Pi 4, Raspberry Pi 5, Jetson Orin Nano
Super) and ONNX is what the other runtimes consume. Adding a backend:

```python
from dart.export import ExportBackend, register_backend

class TFLiteBackend(ExportBackend):
    name = "tflite"
    suffix = ".tflite"
    def export(self, model, path, **opts): ...
    def verify(self, model, artifact, **opts): ...

register_backend(TFLiteBackend())
```

No change to `ExportManager`, to `DART.export()`, or to any model code.

---

## Defining your own variant

Every architectural axis — including the backbone block sequence — is
configuration:

```yaml
# my-variant.yaml
variant: my-variant
model:
  imgsz: 320
  act: silu
  backbone:
    width_mult: 0.75
    tap_indices: [1]
    blocks:
      - {type: repconv,      out_ch: 16, stride: 2, attention: none,  act: silu}
      - {type: dw_sep,       out_ch: 32, stride: 2, attention: coord, act: silu}
      - {type: inverted_res, out_ch: 48, stride: 2, attention: coord, act: silu, expand_ratio: 6}
      - {type: ghost,        out_ch: 96, stride: 2, attention: se,    act: silu}
  neck: {name: fpn_lite}
  head: {name: decoupled_spp, num_targets: 1}
train:
  loss: {conf: focal, bbox: ciou}
```

```python
model = DART("my-variant.yaml")
```

Available block types: `repconv`, `inverted_res`, `ghost`, `dw_sep`.
Attention: `none`, `se`, `coord`. Necks: `identity`, `fpn_lite`.
Heads: `gap`, `spp`, `decoupled_spp`, `softargmax`.

New implementations register themselves:

```python
from dart.models import register_block, register_component, BackboneBlock

@register_block("my_block")
class MyBlock(BackboneBlock):
    def _forward_impl(self, x): ...
```

Attention is applied by the base class, so a block never mentions it.

---

## Pretrained backbones

DART v1 trains from scratch, which is what reproduces the paper's Table I.
Loading an externally-supplied backbone is nonetheless supported end to end:

```python
model = DART("dart-xd", pretrained="path/to/backbone.pt")
report = model.model.load_backbone_weights("org/repo:backbone.pt")
print(report.summary())   # loaded / skipped-on-shape / missing / unexpected
```

Only shape-compatible tensors load; the rest are reported, never silently
dropped. The trainer keeps backbone and head in separate parameter groups
(`backbone.pretrained_lr_mult`), so a pretrained backbone can take a smaller
step than a freshly-initialised head. Publishing DART's own pretrained
checkpoint is future work; nothing in the design forecloses it.

---

## Reproducibility

- `seed=` seeds Python, NumPy, PyTorch (CPU and CUDA), the DataLoader
  shuffling generator, and cuDNN algorithm selection.
- Not controlled: non-deterministic CUDA kernels without a deterministic
  implementation, and third-party op atomics. Worker completion order does not
  affect results, since batches are assembled by index.
- Every run records package and dependency versions, hardware, and a config
  fingerprint, so a metric discrepancy across environments is explainable.

Two runs with the same seed and config produce identical metrics on the same
hardware and software.

---

## Platform notes

Linux, macOS, and Windows; CPU or a single CUDA GPU. Distributed training is
deliberately out of scope.

On Windows and macOS, DataLoader workers start with `spawn`, which re-imports
your script in every worker. Guard your entry point:

```python
if __name__ == "__main__":
    main()
```

DART detects an unguarded script and raises `SpawnSafetyError` naming the fix
rather than letting processes spawn without bound. `workers=0` is always a
correct fallback.

---

## Scope

**In:** single-class single-object detection, the seven validated variants and
their ablations, YOLO/COCO/VOC/CreateML ingestion, ONNX export, CPU and
single-GPU training.

**Out, deliberately:** multi-task support (segmentation, pose, tracking), a
hosted experiment platform, speculative architecture variants, multi-GPU
training.

**Deferred:** additional export backends, a documentation site, and DART's own
pretrained backbone checkpoint.

---

## Documentation

- [docs/configuration.md](docs/configuration.md) — every config field, its
  default, and what it does
- [docs/architecture.md](docs/architecture.md) — module map, the output
  contract, and how to add a block / head / neck / export backend
- [docs/reproducing-the-paper.md](docs/reproducing-the-paper.md) — the full
  grid and the comparison table

## License

Apache-2.0. See [LICENSE](LICENSE).

## Citation

```bibtex
@inproceedings{dart2026,
  title     = {DART: A Family of Lightweight CNNs for Real-Time Detection in
               Resource-Constrained UAV Navigation},
  booktitle = {Latin American Robotics Symposium (SBR)},
  year      = {2026}
}
```
