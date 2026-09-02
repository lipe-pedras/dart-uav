# Architecture

## Module map

```
dart/
    api.py           DART facade — train / val / predict / export
    cli.py           the same four operations as key=value commands
    config/          ConfigResolver, the dataclass schema, the seven presets
    models/          registry, blocks, attention, backbone, necks, heads, detector
    data/            adapters, validator, policies, augmentation, dataset, loader
    losses/          loss primitives and the composite loss
    metrics/         torchmetrics-backed MetricSet, EfficiencyProfiler
    engine/          Trainer, Evaluator, EarlyStopper, RunRecorder, MultiSeedRunner
    export/          ExportManager and its pluggable backends
```

Dependencies point one way and there are no cycles. In particular **`dart.models`
knows nothing about `dart.data` or `dart.engine`** — a model can be built, run,
and exported with no training machinery present, which is what makes
benchmark-only evaluation and export cheap.

## Three design principles

**1. Configuration over code for variants.** The seven variants differ along
orthogonal axes (block sequence, attention, head, neck, activation, loss).
Modelling them as seven subclasses would make an eighth variant a new class.
Instead every axis is resolved by name through a registry, and a variant is a
YAML file listing which implementations to assemble. The test this must pass:
*changing any architectural choice, including the backbone block sequence,
requires editing configuration only, never source.*

**2. Explicit failure over silent adaptation.** The data layer validates the
dataset against the model configuration and raises rather than guessing. No
component silently drops annotations, remaps classes, or picks a policy.

**3. Delegate the commodity, own the contribution.** Dataset parsing goes to
`supervision`; metric computation goes to `torchmetrics`. DART owns the
architecture, the training engine, the config system, and the glue.

## The output contract

Every head returns:

```
conf  : (B, N, 1)   objectness in [0, 1]
boxes : (B, N, 4)   [cx, cy, w, h] in [0, 1]
```

`N` is `model.head.num_targets`, which is 1 today. The count dimension is
explicit rather than squeezed away so that multi-target support becomes a head
emitting `N > 1` plus an assignment policy — not a change to the head
signature, the loss interface, the metric interface, or the batch contract.

Batches match it:

```
images : (B, C, H, W)   float32, normalised
conf   : (B, N)         1.0 where target n is present
boxes  : (B, N, 4)      zeros where absent
```

## The model

```
input → Backbone (list of blocks) → taps → Neck → Head → (conf, boxes)
```

`BackboneBlock` owns the attention slot and applies it through a template
method: `forward()` calls the subclass's `_forward_impl()` and then the
attention module. Subclasses implement convolution logic only and never
mention attention. Consequences: any block type can carry any attention
mechanism; "no attention" is `IdentityAttention`, a null object, so there is no
branch and no `Optional` to guard; a third attention mechanism is a class plus
a registration.

`fuse()` is defined on `BackboneBlock` as a no-op and overridden only by
`RepConvBlock`. `Backbone.fuse()` iterates all blocks unconditionally, so a
second fusible block type needs no change to the traversal, and
`ExportManager` and `EfficiencyProfiler` both call it automatically.

`FPNLiteNeck` takes the tapped feature maps as a list and fuses the first with
the last. Which blocks are tapped comes from `backbone.tap_indices`, so a
six-block backbone can feed the same neck from positions 3 and 6.

## Adding things

**A block type**

```python
from dart.models import BackboneBlock, register_block

@register_block("my_block")
class MyBlock(BackboneBlock):
    def __init__(self, in_ch, out_ch, stride=1, attention="none", act="relu6", **extra):
        super().__init__(in_ch, out_ch, stride, attention, act)
        ...

    def _forward_impl(self, x):        # attention is applied by the base class
        ...

    def fuse(self):                    # only if training != inference structure
        ...
```

Then use it: `- {type: my_block, out_ch: 48, stride: 2, attention: coord}`.

**An attention mechanism, neck, or head**

```python
from dart.models import register_component

@register_component("attention", "my_attention")
class MyAttention(nn.Module):
    def __init__(self, channels, **kwargs): ...
    def forward(self, x): ...
```

Heads must honour the output contract; `Head._shape()` does the reshaping.

**An export backend**

```python
from dart.export import ExportBackend, register_backend

class TFLiteBackend(ExportBackend):
    name = "tflite"
    suffix = ".tflite"
    def export(self, model, path, **opts) -> Path: ...
    def verify(self, model, artifact, **opts) -> VerifyReport: ...

register_backend(TFLiteBackend())
```

`verify()` is part of the interface, not an optional extra: an export that
silently produces a numerically different model is worse than a failed export.

**A dataset format**

Implement `FormatAdapter.read()` returning `RawAnnotations` (absolute-pixel
`xyxy` boxes plus class ids) and add it to `dart.data.adapters.ADAPTERS`.
Nothing downstream changes — normalisation, policy application, and validation
are all format-agnostic.

## The data path

```
path → locate split → adapter.read() → RawAnnotations
                                          ↓
                              DatasetValidator.validate()      ← raises here
                                          ↓
                    ClassSelector → TargetPolicy → Augmentation → tensors
```

Validation runs on the **raw** annotations. If it ran after the policy, a
`largest_only` policy would already have discarded the extra annotations and
the multi-target violation would be invisible — the system would have silently
adapted. It also runs before `Trainer.fit()` is entered, so a misconfigured
dataset fails in seconds rather than after an epoch of wasted compute.

## The training loop

```
Trainer.fit
  ├─ RunRecorder.on_run_start   config.yaml + environment.json
  └─ per epoch
       ├─ train_epoch
       ├─ Evaluator.evaluate → MetricResult
       ├─ EarlyStopper.step  → StopDecision
       ├─ RunRecorder.on_epoch_end / save_checkpoint
       └─ break if decision.stop
  └─ RunRecorder.on_run_end     results.json + efficiency + comparison.csv
```

`EarlyStopper` is a pure decision function — a metrics dict in, a decision out,
no reference to the model, the trainer, or the filesystem. `RunRecorder` owns
all persistence; the trainer never writes a file itself. Each is therefore
testable without the other, and without training anything.

## Metrics framing

`torchmetrics` supplies the arithmetic. What DART owns is how the paper's
single-target task maps onto standard detection metrics:

- **precision** — of the images where a detection was announced
  (`conf >= threshold`), the fraction where a correct detection was available
  (object present *and* IoU ≥ 0.5). A confident but badly-localised box is a
  false positive.
- **recall** — of the images containing an object, the fraction where a
  detection was both announced and well-localised. The denominator is the total
  ground-truth positive count, so poor localisation cannot inflate it.
- **mean IoU** — averaged over ground-truth-positive images only.

`EfficiencyProfiler` always measures a fused deep copy: parameters and latency
taken from the multi-branch training graph would overstate both, and the
caller's model must stay trainable.

## Docstring style

Docstrings are the source for the generated API documentation, so they are
reStructuredText, in NumPy style, and must parse under `numpydoc`.

```python
def ciou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute the Complete-IoU regression loss between paired boxes.

    Extended prose goes here, explaining why the thing exists.

    Parameters
    ----------
    pred : torch.Tensor, shape (n, 4)
        Predicted boxes as ``[cx, cy, w, h]`` in ``[0, 1]``.
    eps : float, default: 1e-7
        Numerical floor applied to every denominator.

    Returns
    -------
    torch.Tensor
        Scalar mean of ``1 - CIoU`` over the ``n`` pairs.
    """
```

The rules, in the order they are most often got wrong:

- **Opening.** Multi-line docstrings put `"""` alone on its line and the summary
  on the next. One-liners stay on a single line. Summaries are imperative for
  functions (`Compute…`, `Build…`), noun phrases for classes.
- **Literals use double backticks.** There is no `conf.py` yet, so
  `default_role` is `title-reference` and a single backtick silently renders as
  italics instead of linking. Cross-references always name an explicit role:
  ``:class:`~dart.engine.trainer.RunResult` ``, `:func:`, `:meth:`, `:exc:`.
- **`See Also` will not accept a bare `~name`.** Write either a plain dotted
  name (`FPNLiteNeck : …`) or the full role form
  (``:class:`~dart.losses.composite.CompositeLoss` : …``). A description that
  wraps must indent its continuation lines by four spaces. Both of these are
  parse errors in `numpydoc`, not style preferences.
- **Types are spelled out**, even though the signature already has them:
  `x : int, default: 8`, `x : str or None, default: None`,
  `x : {"largest_only", "keep_all"}` for registry names, and
  `x : torch.Tensor, shape (B, C, H, W)` for tensors. Shape symbols are `B`
  batch, `C` channels, `H`/`W` spatial, `N` targets per image, `n` boxes in a
  flat set.
- **Constructor parameters live on the class docstring**, not on `__init__` —
  `autoclass` renders only the class docstring by default. `__init__` still
  carries a one-line summary.
- **Dataclasses get `Attributes`, not `Parameters`.** They are records read by
  attribute, and autodoc documents their fields as attributes.
- **Private helpers** always get a summary line; they get `Parameters`/`Returns`
  only when the body is non-obvious.
- **Never write a quoted forward slash, `os.sep`, or `os.path.join` in a
  docstring under `src/dart`.** `tests/test_portability.py` greps raw source
  text and will fail with a confusing "hand-built paths found" message. Say
  "path separator", or use ``` ``a / b`` ```.

Verify with `ruff check`, and with `numpydoc.docscrape.NumpyDocString` over every
docstring — ruff does not catch malformed `See Also` entries.
