# Configuration reference

Every DART run is described by one `RunConfig`. `ConfigResolver` is the only
place defaults are applied: a preset name, a YAML file, and keyword arguments
all go in, and a fully-populated `RunConfig` comes out. That resolved object is
what gets written to `config.yaml` in the run directory, which is why a run
directory is re-runnable on its own.

Precedence, lowest to highest:

1. dataclass defaults (in `dart/config/schema.py`)
2. the preset named by `extends:`/`preset:` in a YAML config
3. the YAML config or preset passed as `model=`
4. keyword arguments at the call site / `key=value` on the CLI

```python
from dart.config import ConfigResolver

cfg = ConfigResolver().resolve("dart-xd", imgsz=320, epochs=300)
cfg.to_yaml("resolved.yaml")
print(cfg.fingerprint())
```

Any field can be set three ways — a nested YAML mapping, a dotted keyword, or
one of the shorthands:

```python
model.train(imgsz=320)                          # shorthand
model.train(**{"model.imgsz": 320})             # dotted path
model.train(model__imgsz=320)                   # dotted path, keyword-safe
```

```bash
dart train imgsz=320 train.early_stop.patience=25
```

## Shorthands

| Shorthand | Full path |
|---|---|
| `data` | `data.path` |
| `format` | `data.format` |
| `target_policy` | `data.target_policy` |
| `target_class` | `data.target_class` |
| `grayscale` | `data.grayscale` |
| `workers` | `data.workers` |
| `epochs` | `train.epochs` |
| `batch`, `batch_size` | `train.batch_size` |
| `lr` | `train.lr` |
| `weight_decay` | `train.weight_decay` |
| `device` | `train.device` |
| `amp` | `train.amp` |
| `patience` | `train.early_stop.patience` |
| `resume` | `train.resume` |
| `project`, `name` | `train.project`, `train.name` |
| `conf` | `train.conf_threshold` |
| `imgsz` | `model.imgsz` |
| `width`, `width_mult` | `model.backbone.width_mult` |
| `pretrained` | `model.backbone.pretrained` |
| `seed` | `seed` |

---

## `model`

| Field | Default | Meaning |
|---|---|---|
| `model.imgsz` | `96` | Square input resolution. The paper validates 96, 160, 224, 320; any positive multiple of 32 is accepted. |
| `model.act` | `relu6` | Default activation for blocks and heads. `relu6`, `relu`, `silu` (alias `swish`), `hardswish`. |
| `model.num_classes` | `1` | v1 is single-class; any other value errors. The field exists so multi-class is an added head, not a rewrite. |

### `model.backbone`

| Field | Default | Meaning |
|---|---|---|
| `width_mult` | `1.0` | Scales every block's `out_ch`. At 1.0 the paper's widths (16→32→48→64); at 0.5, (8→16→24→32). Floored at 8 channels. |
| `blocks` | preset-defined | Ordered list of `BlockSpec`. Replacing this list *is* how a new backbone is defined. |
| `tap_indices` | `[]` | Zero-based block indices whose outputs the neck receives. Negatives count from the end. The last block is always tapped. |
| `in_channels` | `3` | 1 when `data.grayscale` is set; derived automatically unless you set it explicitly. |
| `pretrained` | `null` | A `state_dict` path or `repo_id:file.pt` Hub reference loaded into the backbone at construction. |
| `pretrained_lr_mult` | `1.0` | Learning-rate multiplier for backbone parameters relative to head parameters. |

### `BlockSpec`

| Field | Default | Meaning |
|---|---|---|
| `type` | `inverted_res` | `repconv`, `inverted_res`, `ghost`, `dw_sep`, or any registered block. |
| `out_ch` | `32` | Output channels *before* `width_mult`. |
| `stride` | `1` | Spatial stride. |
| `attention` | `none` | `none`, `se`, `coord`, or any registered attention. Applies to any block type. |
| `act` | `relu6` | Activation inside the block. |
| *anything else* | — | Passed to the block constructor, e.g. `expand_ratio: 4` (inverted residual), `dw_kernel: 5`, `ratio: 2` (ghost). |

### `model.neck` and `model.head`

| Field | Default | Meaning |
|---|---|---|
| `neck.name` | `identity` | `identity` (null object) or `fpn_lite`. |
| `neck.kwargs` | `{}` | e.g. `out_channels`, `pool_size` for `fpn_lite`. |
| `head.name` | `gap` | `gap`, `spp`, `decoupled_spp`, `softargmax`. |
| `head.num_targets` | `1` | Boxes emitted per image (`N` in the output contract). |
| `head.kwargs` | `{}` | e.g. `feat_dim`, `hidden`, `temperature`. |

---

## `data`

| Field | Default | Meaning |
|---|---|---|
| `path` | `null` | Dataset root or dataset YAML. Required to train or evaluate. |
| `format` | `auto` | `yolo`, `coco`, `voc`, `createml`, or `auto` (inferred from the files present). |
| `train`, `val` | `train`, `val` | Split names or sub-paths. |
| **`target_policy`** | **none — explicit** | `largest_only` or `keep_all`. A dataset with more annotations per image than the model has targets and no policy set is an **error**. |
| **`target_class`** | **none — explicit** | Class id kept as the target; everything else becomes background. A multi-class dataset with no selection is an **error**. |
| `class_map` | `{}` | Explicit `old_id: new_id` remapping applied at ingestion; `new_id: null` drops a class to background. Runs *before* `target_class`, so that key names an id in the remapped space. Never inferred. |
| `class_names` | `{}` | `id: name`, used to make error messages readable. |
| `grayscale` | `false` | Load images as one channel. |
| `batch_size` | `32` | Kept in step with `train.batch_size` when you use the `batch` shorthand. |
| `workers` | `4` | DataLoader workers. `0` is always correct and never spawns. |

### `data.augment`

Defaults reproduce the paper's pipeline exactly. `enabled: false` disables all
of it; validation splits never augment regardless.

| Field | Default | Meaning |
|---|---|---|
| `hflip`, `vflip` | `0.5` | Probability of each mirror. |
| `brightness` / `brightness_range` | `0.7` / `(0.6, 1.4)` | Intensity scaling. |
| `contrast` / `contrast_range` | `0.7` / `(0.7, 1.3)` | Deviation-from-mean scaling. |
| `noise` / `noise_sigma` | `0.6` / `(2, 12)` | Additive Gaussian sensor noise. |
| `crop_resize` / `crop_scale` | `0.6` / `(0.75, 0.98)` | Crop-and-resize; simulates altitude change. Skipped on negatives. |
| `rotate` / `rotate_degrees` | `0.5` / `15.0` | Rotation about the centre; simulates camera roll. Skipped on negatives. |
| `blur` / `blur_kernels` | `0.4` / `(3, 5)` | Motion blur / defocus. |

Replace the pipeline wholesale in Python:

```python
from dart.data import AugmentationPipeline
from dart.data.augment import HorizontalFlip, GaussianBlur

dataset.augment = AugmentationPipeline([HorizontalFlip(0.5), GaussianBlur(0.2)])
```

---

## `train`

| Field | Default | Meaning |
|---|---|---|
| `epochs` | `300` | Maximum epochs; early stopping may end the run sooner. |
| `batch_size` | `32` | |
| `lr` | `1e-3` | Base learning rate (head group; backbone scales by `pretrained_lr_mult`). |
| `weight_decay` | `1e-4` | |
| `optimizer` | `adamw` | `adamw`, `adam`, `sgd`. |
| `scheduler` | `cosine` | `cosine`, `step`, `none`. |
| `min_lr` | `1e-6` | Cosine floor. |
| `grad_clip` | `10.0` | Max gradient norm; `0` disables. |
| `amp` | `true` | Mixed precision, CUDA only. The confidence loss always runs in fp32 regardless. |
| `conf_threshold` | `0.5` | Confidence at which precision/recall/F1 are reported. |
| `device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:1`, `mps`. |
| `project`, `name` | `runs`, `exp` | Output directory; suffixed rather than overwritten if it exists. |
| `resume` | `null` | Path to a `last.pt` to resume from (epoch-boundary granularity). |
| `save_period` | `0` | Also save `epoch<N>.pt` every N epochs. |

### `train.loss`

| Field | Default | Meaning |
|---|---|---|
| `conf` | `bce` | `bce` or `focal`. |
| `bbox` | `smooth_l1` | `smooth_l1` or `ciou`. |
| `alpha`, `beta` | `1.0`, `2.0` | `total = alpha * conf + beta * bbox`. |
| `focal_alpha`, `focal_gamma` | `0.25`, `2.0` | Focal loss parameters. |

The paper's two configurations are `bce + smooth_l1` (DART-B) and
`focal + ciou` (everything else).

### `train.early_stop`

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `true` | `false` runs the full epoch budget but still tracks the best. |
| `primary_metric` | `map50` | Any key the evaluator reports. |
| `direction` | `max` | `max` or `min`. |
| `patience` | `40` | Epochs without improvement before stopping. |
| `min_delta` | `1e-3` | Improvement smaller than this does not count. |
| `secondary_metric` | `mean_iou` | `null` reduces this to plain patience stopping. |
| `secondary_rel_gain` | `0.03` | Relative gain the secondary must show to veto a stop. |
| `secondary_window` | `3` | Epochs over which that gain is measured. |
| `primary_saturated` | `0.999` | The secondary may only veto once the primary has reached this. |

---

## Top level

| Field | Default | Meaning |
|---|---|---|
| `seed` | `0` | Seeds Python, NumPy, torch (CPU and CUDA), the shuffling generator, cuDNN, **and model initialisation**. |
| `variant` | `custom` | The name recorded in metrics rows; presets set their own. |
