# DART Framework — Project Description & Requirements Specification

**Status:** Draft for approval
**Stage:** 1 of 3 (Requirements → UML → Implementation)
**Companion to:** *DART: A Family of Lightweight CNNs for Real-Time Detection in
Resource-Constrained UAV Navigation* (SBR 2026)

---

## 1. Introduction

### 1.1 Purpose

This document specifies the requirements for the DART framework: an open-source
Python package that packages the model family, training pipeline, and evaluation
methodology described in the DART paper into a reusable, installable tool. It is
the artifact a reviewer or independent researcher would use to reproduce the
paper's results, and the artifact a practitioner would use to train or deploy a
DART variant on their own single-class detection task.

This document is the first of three project stages. Implementation does not
begin until this specification is approved; UML design does not begin until the
scope defined here is fixed.

### 1.2 Purpose of the Framework (Problem Statement)

The DART paper produced a working but research-script-style codebase: standalone
scripts (`train.py`, `grid.py`, `benchmark_datasets.py`, `driver.py`, …) coupled
to the exact five datasets and hardware used in the paper. This is sufficient
for producing the paper's results but is not something a third party can
`pip install` and point at their own data. The framework's purpose is to close
that gap: same underlying training/evaluation logic, wrapped behind a small,
stable, documented public API.

### 1.3 Relationship to Prior Art

The target developer experience is explicitly modeled on Ultralytics' YOLO API
(`from ultralytics import YOLO; model = YOLO('yolov8n.pt')`), because that
ergonomic pattern is what makes a detector easy to adopt. The framework
reproduces that *ergonomic pattern*, not Ultralytics' *engineering scope*
(multi-task support, hosted platform, broad export matrix, dedicated
maintenance team). Section 3.2 makes this boundary explicit.

### 1.4 Definitions

| Term | Meaning |
|---|---|
| Variant | One of the seven architectural/loss configurations validated in the paper (DART-B, DART-B-CIoU, DART-S, DART-D, DART-F, DART-X, DART-XD) |
| Run | One training or evaluation execution of one variant on one dataset at one resolution |
| Backbone | The four-block feature extractor shared by all variants |
| Head | The detection head (GAP / SPP-Lite / Decoupled SPP-Lite) attached to the backbone |
| Neck | The optional FPN-Lite feature-fusion stage between Block 2 and Block 4 |

### 1.5 References

- DART paper (SBR 2026 submission), Sections III (Architecture), V (Experimental
  Setup), VI (Results)
- `model.py`, `train.py`, `dataset.py`, `grid.py`, `benchmark_datasets.py`,
  `driver.py`, `eval_only.py`, `baselines.py`, `yolo_baseline.py`,
  `train_yolo.py`, `mobilenet_baseline.py`, `fomo_torch.py`,
  `prepare_hf_datasets.py` (existing research codebase, migration source)

---

## 2. Overall Description

### 2.1 Product Perspective

A standalone, pip-installable Python package. It depends on PyTorch as the
training substrate and on established third-party libraries for dataset
ingestion and metrics (Section 2.5), rather than reimplementing those
subsystems from scratch. The package's own code is limited to: the DART
architecture (backbone, heads, necks, losses), the training engine
(scheduling, early stopping, checkpointing), the config system that exposes
the seven variants as named presets, and the export/CLI glue.

### 2.2 Product Functions (summary)

1. Load a detection dataset in any of several common annotation formats.
2. Instantiate any of the seven paper-validated DART variants, or a
   user-defined variant, from a short name or a config file.
3. Train a variant on a user's dataset with sensible, overridable defaults,
   including the paper's compound early-stopping criterion.
4. Evaluate a trained checkpoint and report the same metric set used in the
   paper (mAP@0.5, precision, recall, F1, mean IoU), computed via a
   third-party metrics library rather than bespoke code.
5. Run the same configuration across multiple random seeds and report
   mean ± standard deviation per metric.
6. Export a trained checkpoint to ONNX for deployment on the three embedded
   targets used in the paper (Raspberry Pi 4/5, Jetson Orin Nano Super class
   hardware).
7. Drive all of the above from a Python API or a command-line interface.
8. Persist a full, reloadable record of every run (config, weights, metrics,
   environment) sufficient to regenerate the paper's comparison tables.

### 2.3 User Classes and Characteristics

| User class | Need | Priority |
|---|---|---|
| Paper reviewers / reproducers | Re-run the exact experiments in the paper from a clean install | High |
| UAV / embedded practitioners | Train DART on their own single-class dataset and deploy it | High |
| Extending researchers | Add a new head/neck/loss variant without forking the training loop | Medium |
| The author (ongoing research) | Reuse this exact codebase for future papers/datasets | High |

### 2.4 Operating Environment

- Python 3.9+
- Training: Linux, macOS, and Windows; CPU or CUDA GPU. See NFR-3 for the
  specific portability obligations Windows support imposes.
- Inference targets validated by the paper: Raspberry Pi 4, Raspberry Pi 5
  (CPU-only), NVIDIA Jetson Orin Nano Super (GPU-accelerated).

### 2.5 Design and Implementation Constraints

- **Must** use PyTorch as the tensor/autograd substrate (already the case).
- **Must** use an established third-party library for dataset ingestion
  (`supervision`) rather than hand-written per-format parsers.
- **Must** use an established third-party library for metric computation
  (`torchmetrics`) rather than the bespoke `iou_batch` /
  `compute_pr_metrics` implementations in the current research codebase.
  This is a scientific-integrity requirement, not just an engineering
  preference: metrics reported in a peer-reviewed paper should be
  computed by code the community has already validated, not by
  project-specific code a reviewer cannot easily audit.
- **Should** keep the on-disk data format YOLO-compatible by default (the
  paper's datasets are already in this format), while accepting other
  formats through the same ingestion layer.
- **Should not** introduce a hosted service, account system, or telemetry.

### 2.6 Assumptions and Dependencies

- Model weights and full per-epoch artifacts continue to be hosted on the
  Hugging Face Hub (already decided); the framework's checkpoint loader is
  responsible for resolving a short name to a Hub download when the file is
  not present locally.
- The SBR 2026 review process is finished; the framework is published under
  the author's own GitHub account/organization, with no anonymity
  requirement.
- The final PyPI package name is not "dart" (unavailable) and is decided
  separately from this document (open item, Section 8).

---

## 3. Scope

### 3.1 In Scope

- Single-class, single-object detection (the task class the paper defines
  and validates).
- The seven DART variants and their documented ablations (head type, neck
  presence, attention type, activation, loss configuration).
- Training from scratch remains the v1 default and is what reproduces the
  paper's published results (Table I). Separately, generic support for
  initializing a DART backbone from externally-supplied pretrained weights
  (a user-provided `state_dict` or Hub reference) is in scope as
  infrastructure — see FR-2.6. Producing and officially publishing DART's
  own pretrained backbone checkpoint is a larger, separate effort and is
  tracked in Section 3.3 (Deferred), not v1.
- Dataset ingestion for YOLO, COCO, Pascal VOC, and CreateML formats (the
  formats `supervision` already provides adapters for).
- Export to ONNX. ONNX Runtime covers all three paper-validated embedded
  targets, and ONNX is the interchange format the other runtimes
  (TensorRT, and TFLite via conversion) consume, so it is the highest-value
  single target. Additional export backends are deferred (Section 3.3).
- CPU and single-GPU training. No distributed/multi-GPU training requirement.

### 3.2 Explicitly Out of Scope

These are deliberate exclusions, not oversights, per the scope discipline
established during framework planning:

- Multi-task support (segmentation, pose, classification, tracking). DART is
  a detector; it should not claim to be more.
- A hosted experiment-tracking platform or web dashboard equivalent to
  Ultralytics HUB.
- Speculative new architectural variants beyond the seven validated in the
  paper. New variants may be added later, but each addition is a separate,
  deliberate decision with its own validation, not a "framework feature."
- Multi-GPU / distributed training.

### 3.3 Deferred (Not v1, Not Rejected)

- Additional export backends beyond ONNX (TFLite, NCNN, direct TensorRT).
  Deferred because ONNX Runtime already covers the three paper-validated
  embedded targets and ONNX is the interchange format the others consume.
  FR-5.2 requires the export interface be pluggable so these can be added
  without touching the public API or the model code.
- A hosted documentation site. Confirmed as future work; v1 ships a README
  plus in-repo Markdown docs.
- Producing and officially publishing a DART-native pretrained backbone
  checkpoint (ImageNet, COCO, or self-supervised on unlabeled drone
  imagery). Confirmed as future work. Candidate approaches — partial weight
  transfer from a pretrained MobileNetV2 into the compatible
  inverted-residual blocks, full from-scratch pretraining of the DART
  backbone, or self-supervised pretraining — have different cost/risk
  profiles and warrant their own validation before becoming a default.
  Motivation: three of the paper's five datasets have under 1,000 training
  images, and pretraining tends to help most exactly in that regime. Note
  that FR-2.6 requires the *design* to accommodate this from day one.

---

## 4. Functional Requirements

Each requirement has an ID for traceability into the UML stage. Priority
follows MoSCoW: **M**ust, **S**hould, **C**ould.

### FR-1 — Dataset Ingestion

| ID | Requirement | Priority |
|---|---|---|
| FR-1.1 | The system shall load a detection dataset given a path and an explicit or auto-detected format among {YOLO, COCO, Pascal VOC, CreateML}. | M |
| FR-1.2 | The system shall normalize any supported input format to a single internal representation before it reaches the training loop, so the training engine is format-agnostic. | M |
| FR-1.3 | The system shall treat "how many annotations per image are kept" as an explicit, configurable target-selection policy with **no silent default**. The paper's convention (`largest_only`) and `keep_all` shall both be selectable. If a dataset violates the model's target configuration and no policy is set, the system shall error rather than choose one (FR-1.7). The interface shall be designed so that future multi-target and multi-class support is an added policy and an added head, not a rewrite of the ingestion or training path — see NFR-8. | M |
| FR-1.4 | The system shall support per-dataset class selection and remapping at ingestion time, and shall do so **only when explicitly configured**. The system shall never infer, guess, or silently apply a class mapping. (Concrete case from the paper: `medicalkit-drone-dataset` ships class 0 = landing base and class 1 = package; selecting the package as the target and mapping the base to background must be a written configuration choice, not a default behavior.) | M |
| FR-1.7 | The system shall validate the dataset against the selected model configuration before training begins, and shall fail with an explicit, actionable error rather than silently adapting. Specifically: (a) if the model is configured single-class and the dataset contains more than one annotated class, the system shall error and require an explicit class-selection configuration (FR-1.4); (b) if the model is configured single-target and any image contains more than one annotation, the system shall error and require an explicit target-selection policy (FR-1.3). Errors shall name the offending dataset, the offending class or image, and the configuration key that resolves it. | M |
| FR-1.5 | The system shall report basic dataset statistics (image count, positive/negative rate) on load. | S |
| FR-1.6 | The system shall support the paper's augmentation set (flip, brightness/contrast, Gaussian noise, crop-resize, rotation) as a configurable, swappable pipeline stage. | M |

### FR-2 — Model / Configuration

| ID | Requirement | Priority |
|---|---|---|
| FR-2.1 | The system shall instantiate any of the seven paper-validated variants from a short name (e.g. `"dart-xd"`). | M |
| FR-2.2 | The system shall instantiate a variant from an explicit YAML config exposing the same architectural axes as `grid.py` (attention type, head type, activation, neck on/off, loss configuration), so users can define variants outside the validated seven. | M |
| FR-2.3 | The system shall support loading a trained checkpoint by short name or local path, auto-resolving to a Hugging Face Hub download when the file is not present locally. | M |
| FR-2.4 | The system shall support all four input resolutions used in the paper (96, 160, 224, 320) as a config parameter, not a hardcoded constant per variant. | M |
| FR-2.5 | The system shall expose the backbone width multiplier as a config parameter. This scales every backbone block's channel count by a constant factor (at 1.0: 16→32→48→64; at 0.5: 8→16→24→32), trading capacity for size and speed. It is already implemented in the research codebase and the paper reports only width = 1.0; exposing it costs one config field and opens a size/accuracy axis orthogonal to input resolution. | S |
| FR-2.6 | The system shall be designed on the assumption that pretrained backbone weights will exist and be used. Concretely: backbone construction shall be separable from head/neck construction so a backbone can be built and loaded independently; loading shall accept a `state_dict` from a local path or Hub reference, load only shape-compatible layers, and warn (not fail) on the rest; and the training engine shall support a distinct learning rate for pretrained backbone parameters versus randomly-initialized head parameters. Shipping DART's own official pretrained checkpoint remains deferred (Section 3.3), but no design decision may foreclose it. | M |

### FR-3 — Training Engine

| ID | Requirement | Priority |
|---|---|---|
| FR-3.1 | The system shall provide a `train()` operation accepting, at minimum: dataset, variant/config, epochs, batch size, resolution, and random seed. | M |
| FR-3.2 | The system shall provide configurable early stopping in which the monitored metric, the improvement threshold, the patience, and the optimization direction (maximize/minimize) are all user-specified. A compound criterion shall be optionally composable on top: a secondary metric that, while still improving by a configurable relative gain over a configurable window, prevents stopping even when the primary metric has plateaued. The paper's criterion (primary mAP@0.5, secondary mean IoU at 3% over 3 epochs, patience 40) shall be reproducible as one configuration of this mechanism, not hardcoded as the only behavior. | M |
| FR-3.3 | The system shall support resuming an interrupted training run from its last checkpoint. | S |
| FR-3.4 | The system shall write, into each run's output directory, the *fully-resolved* configuration actually used — every default merged with every user override and CLI argument, not just what the user typed. Rationale: a run launched as `dart train data=x.yaml model=dart-xd` depends on dozens of unstated defaults (LR schedule, augmentation probabilities, patience, seed). If those defaults change in a later version, the original run becomes unreproducible from the command alone. Snapshotting the resolved config means any run directory is self-contained: re-running it needs only that file, with no knowledge of which package version produced it. | M |
| FR-3.5 | The system shall accept an explicit random seed and document which sources of nondeterminism it does and does not control. | M |
| FR-3.6 | The system shall support the two loss configurations used in the paper (BCE + Smooth L1; Focal + CIoU) as named, swappable loss presets. | M |

### FR-4 — Evaluation & Metrics

| ID | Requirement | Priority |
|---|---|---|
| FR-4.1 | The system shall compute mAP@0.5, mAP@0.5:0.95, precision, recall, F1, and mean IoU using `torchmetrics`' detection metrics, not bespoke implementations. mAP@0.5:0.95 is added beyond the paper's reported set because it is the standard COCO-style primary metric and its absence would make DART harder to compare against externally-published detectors. | M |
| FR-4.2 | The system shall support a benchmark-only evaluation mode (load a checkpoint, run validation, no training), mirroring `eval_only.py`. | M |
| FR-4.3 | The system shall support running one configuration across multiple seeds and reporting mean ± standard deviation per metric. | S |
| FR-4.4 | The system shall export per-run metrics in a schema compatible with the paper's `model_comparison.csv`, so the framework can regenerate the paper's own tables and the benchmark site's `data.json` from a fresh run. | M |
| FR-4.5 | The system shall record, per run, the efficiency metrics currently missing from `model_comparison.csv` (parameter count, quantized size, CPU/GPU latency) as a first-class part of the metrics schema, not an afterthought. | M |

### FR-5 — Export

| ID | Requirement | Priority |
|---|---|---|
| FR-5.1 | The system shall export a trained model to ONNX. | M |
| FR-5.2 | The export interface shall be backend-pluggable: adding a new export target shall require registering a new backend, not modifying the public `export()` signature or the model code. | S |
| FR-5.3 | The system shall preserve the RepConv reparameterization fusion (training-time multi-branch → inference-time single conv) automatically before export, without requiring the user to call a separate fusion step. | M |

### FR-6 — Interfaces

| ID | Requirement | Priority |
|---|---|---|
| FR-6.1 | The system shall expose a Python API sufficient to train, evaluate, predict, and export without touching a config file (all overridable by keyword arguments). | M |
| FR-6.2 | The system shall expose an equivalent command-line interface for the same four operations. | S |
| FR-6.3 | The system shall accept YAML as the on-disk config format. | M |

### FR-7 — Reproducibility & Provenance

| ID | Requirement | Priority |
|---|---|---|
| FR-7.1 | The system shall record, per run, the package version, key dependency versions, and hardware description sufficient to explain a metric discrepancy across environments. | S |
| FR-7.2 | The system shall make the origin of every reported baseline number traceable: which script/version produced it, matching the paper's own disclosure norms (e.g. the YOLOv11n re-evaluation after the incomplete-training-run discrepancy found during paper preparation). | M |

---

## 5. Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-1 | Performance | Training throughput shall not regress more than ~10% versus the current raw-script pipeline for an equivalent configuration; the abstraction layer must not become the bottleneck. |
| NFR-2 | Reproducibility | Two runs with the same seed and config shall produce metrics within floating-point tolerance on the same hardware/software environment. |
| NFR-3 | Portability | Training shall run unmodified on Linux, macOS, and Windows. This imposes three concrete obligations: all filesystem paths shall use `pathlib` rather than string concatenation or hardcoded separators; all `DataLoader` worker usage shall be safe under the `spawn` start method Windows requires — concretely, every object crossing the worker boundary (dataset, transforms, collate function) shall be picklable, meaning no lambdas stored as attributes, no local closures, and no open file or capture handles held on the dataset object; the package's own CLI entry points shall be guarded with `if __name__ == "__main__"`; and the documentation shall state the same requirement for user scripts. Where a user script violates it, the system shall detect the condition and raise a clear error naming the fix, rather than allowing runaway process spawning. `num_workers=0` shall remain a always-correct fallback. Exported artifacts shall run on the three paper-validated embedded targets. |
| NFR-4 | Usability | A new user shall be able to train the recommended variant on a YOLO-format dataset in under 10 lines of Python, matching the Ultralytics quickstart experience in brevity. |
| NFR-5 | Maintainability | Dataset ingestion and metric computation shall be delegated to third-party libraries (Section 2.5) specifically to keep the project's own maintenance surface small. |
| NFR-6 | Licensing | The package shall use a permissive license (MIT or Apache-2.0), decided in Section 8. |
| NFR-7 | Documentation | Every public API method and every config field shall have a docstring; the README shall include a working quickstart snippet, verified against the actual package on every release. |
| NFR-8 | Extensibility | The architecture shall be designed so that the currently out-of-scope extensions — multi-target detection, multi-class detection, and pretrained backbone weights — can be added by introducing a new policy, head, or loading path, without restructuring the dataset ingestion layer, the training engine, or the public API. Where a v1 simplification would foreclose one of these, the simplification shall be rejected. |
| NFR-9 | Continuous integration | The repository shall run automated checks on every pull request, at the tiers defined in Section 5.1. |

### 5.1 Continuous Integration Tiers

CI is free for public GitHub repositories, so the constraint is maintenance
effort and runtime, not cost.

**Trigger policy (decided):** Tiers 1 and 2 both run on every pull request;
Tier 3 runs on tags only. Tier 2 is a superset of Tier 1 in practice — the
reproducibility and export checks presuppose that the correctness checks
passed — so "Tier 2 on PRs" means the full Tier 1 + Tier 2 suite gates every
merge.

**Tier 1 — Correctness gate.** *Trigger: every pull request.*
Runs on Linux + Windows, across the minimum and latest supported Python
versions.
- Package imports cleanly.
- A one-epoch training run on a tiny synthetic fixture dataset completes and
  produces the expected artifacts (weights, resolved config, metrics file).
- Unit tests for the pieces where a silent bug would corrupt published
  numbers: metric computation against hand-checked fixtures, the RepConv
  fusion equivalence check (fused output must match unfused within
  floating-point tolerance), and dataset ingestion for each supported format.
- Validation tests for FR-1.7: a multi-class dataset against a single-class
  model, and a multi-annotation image against a single-target model, must
  each raise the documented error rather than silently proceeding.
- Lint and format check.

The RepConv fusion test deserves emphasis: it is the one place where a
regression would silently change every exported model's behavior while
leaving training metrics untouched.

**Tier 2 — Reproducibility gate.** *Trigger: every pull request.*
- Two runs with the same seed and config produce identical metrics
  (enforces NFR-2 rather than merely asserting it in prose).
- Export to ONNX succeeds and the exported model's output matches the
  PyTorch model's output on a fixed input, within tolerance.

**Tier 3 — Release gate.** *Trigger: tags only.*
- Build wheel and sdist, verify install into a clean environment.
- Verify the README quickstart snippet executes against the built package
  (enforces NFR-7).
- Publish to PyPI.

Deliberately excluded: full training runs on the real datasets (too slow and
too expensive for CI), GPU runners, and macOS in Tier 1 (Linux plus Windows
covers the portability risks that NFR-3 identifies; macOS can be added to
Tier 2 if a platform-specific bug ever appears).

---

## 6. External Interface Requirements

### 6.1 Python API (illustrative, not final — finalized at UML stage)

```python
from dart import DART

model = DART("dart-xd")                 # architecture, random init
model = DART("dart-xd.pt")               # checkpoint, resolved locally or via HF Hub

model.train(data="my_dataset.yaml", epochs=300, imgsz=320, seed=0)
metrics = model.val(data="my_dataset.yaml")     # -> mAP50, precision, recall, f1, mean_iou
results = model.predict("image.jpg")
model.export(format="onnx")
```

### 6.2 CLI (illustrative)

```
dart train  data=my_dataset.yaml model=dart-xd epochs=300 imgsz=320
dart val    data=my_dataset.yaml model=dart-xd.pt
dart export model=dart-xd.pt format=onnx
```

### 6.3 Config File (illustrative)

```yaml
# dart-xd.yaml
head: decoupled_spp
neck: fpn_lite
attention: coord
activation: silu
loss:
  conf: focal
  bbox: ciou
```

### 6.4 Data Formats

Input: YOLO, COCO, Pascal VOC, CreateML (via `supervision`).
Output (export): ONNX.
Metrics: CSV/JSON matching the paper's `model_comparison.csv` schema.

---

## 7. Traceability to the Paper

| Paper artifact | Framework requirement |
|---|---|
| Table of 7 variants (DART-B … DART-XD) | FR-2.1, FR-2.2 |
| Compound early-stopping criterion (Section V) | FR-3.2 |
| Two loss configurations (Section III-D) | FR-3.6 |
| `model_comparison.csv` (135 runs) | FR-4.4 |
| Missing efficiency columns (params/size/FPS) flagged as pending in the paper | FR-4.5 |
| Three embedded inference targets (RPi4, RPi5, Orin Nano Super) | FR-5.1, NFR-3 |
| RepConv structural reparameterization | FR-5.3 |

This table is the acceptance check for this document: every load-bearing claim
in the paper should have a corresponding requirement above. If a reviewer or
co-author can name a paper claim not covered here, that is a gap to close
before UML begins, not after.

---

## 8. Decisions

| # | Item | Decision |
|---|---|---|
| 8.1 | PyPI package name | **`dart-uav`**. Python namespace remains `dart` (`from dart import DART`), mirroring how `pip install ultralytics` and the import name are decoupled. Verify availability on PyPI immediately before first publish. |
| 8.2 | License | **Apache-2.0**. Permissive like MIT, with an explicit patent grant and contributor terms — the better default for a package intended to be adopted by industry and academia alike. |
| 8.3 | FOMO point-in-box metric | **Removed from scope.** FOMO is a baseline the paper compared against, not a DART model; the framework has no reason to implement a foreign model's evaluation criterion. Former FR-4.3 deleted; FR-4.4–4.6 renumbered to FR-4.3–4.5. |
| 8.4 | Minimum PyTorch version | **Intentionally left open.** To be determined empirically and pinned during implementation; does not block UML. See note below. |
| 8.5 | Width multiplier (FR-2.5) | **Approved and kept in scope**, priority raised to **S**. Already implemented in the research codebase; costs one config field and provides a size/accuracy axis orthogonal to input resolution. |
| 8.6 | Export backends | **ONNX only for v1** (FR-5.1). NCNN removed entirely; TFLite and other backends deferred (Section 3.3) behind a pluggable backend interface (FR-5.2). |
| 8.7 | Windows support | **In scope** (NFR-3), with `pathlib` throughout and explicit `spawn`-safety obligations handled by the framework itself. |
| 8.8 | CI trigger policy | **Tiers 1 and 2 on every pull request; Tier 3 on tags only** (Section 5.1). |

### 8.4 — Minimum PyTorch version (still open)

This is the one item that cannot be settled from the requirements alone,
because it is driven by the export dependencies rather than by DART itself:

- The training and model code use no exotic APIs and would run on a wide
  range of PyTorch versions.
- With export scope reduced to ONNX (FR-5.1), the constraint is much looser
  than it would have been with TFLite/NCNN backends: `torch.onnx.export` has
  been stable for many releases.
- Pinning too low risks subtle ONNX opset differences; pinning too high
  excludes users on older CUDA/driver stacks, which is common on the
  embedded and lab machines this project targets.

**Recommended resolution:** pin the floor to whatever `ai-edge-torch`'s
current release requires, verified empirically in a clean environment rather
than taken from documentation, and record the tested combination in
`pyproject.toml` and the README. This verification is a concrete task for the
implementation stage; it does not block UML design, since no class structure
depends on it.

---

## 9. Approval

This document gates Stage 2 (UML). Section 3 (scope), Sections 4–5
(requirements), and Section 8 (decisions) are resolved apart from item 8.4,
which is explicitly deferred to implementation and does not block design.
UML design (class diagram for the model/config system, sequence diagram for
the train→evaluate→export flow) may begin.
