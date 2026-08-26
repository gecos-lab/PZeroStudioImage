# DOMStudio 2

DOMStudio is a research-oriented desktop workspace for detecting and tracing
geological fractures in raster imagery. Version 2 replaces the former
filter-tab application with one explicit pipeline:

```text
native raster → probability + orientation + uncertainty → topology-aware vectorisation → fracture graph
```

The redesign does not expose Canny, Sobel, Laplacian, Roberts, HED, shearlet,
or skeletonisation tabs. Those generic edge operators produced visually
different rasters but did not form a coherent, measurable fracture-extraction
method.

DOMStudio is a research prototype, not a validated geological interpretation
system. This repository bundles no learned weights and makes no accuracy,
generalisation, calibration, or runtime-performance claim.

## What changed

- Source rasters remain at native resolution and retain CRS, affine transform,
  dtype, and nodata metadata.
- Detection runs in a worker thread instead of blocking the Qt event loop.
- The built-in analytical detector estimates multiscale ridge evidence, axial
  tangent orientation, and cross-scale disagreement. These fields are useful
  for tracing but are not calibrated probabilities.
- Optional learned inference uses strict, checksummed, local-only ONNX model
  packs. The application never downloads or silently substitutes weights.
- Automatic vectorisation uses strong/weak evidence hysteresis to retain faint
  fracture continuations, direction-gated gap repair, centreline thinning, and
  skeleton-to-graph conversion. Short spurs and small closed blobs are rejected
  before vectors are exposed to the user.
- A* state includes the incoming direction, so curvature is a real transition
  cost. Search is bounded to an anchor-defined corridor and accelerated by an
  exact scalar cost-to-go lower bound, curvature-aware lookahead, and feasible
  incumbent pruning without changing the optimum.
- Automatically extracted and manually corrected paths form one vector network
  with stable nodes, edges, connected components, degrees, cycles, lengths, and
  axial orientations. It exports in GeoPackage, GeoJSON, Shapefile, or vertex
  CSV form. Pixel centres are transformed with the source affine exactly once.
- A reusable project-state service uses versioned JSON rather than executable
  pickle files.
- Geometry-first evaluation reports tolerance-aware precision, recall, F1,
  symmetric distance, Hausdorff distance, endpoint error, and length error.

## Run it

Python 3.10–3.12 is supported.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

For editable installation and the `domstudio` command:

```powershell
python -m pip install -e .
domstudio
```

The standard application has no PyTorch dependency. For learned-model
development or ONNX inference, install only the relevant extra:

```powershell
python -m pip install -e ".[learned]"
python -m pip install -e ".[inference]"
```

## Workflow

1. Open PNG, JPEG, BMP, GeoTIFF, or JPEG 2000 imagery.
2. Use the default **Fast preview** for quick parameter exploration, choose
   **Analytical detector** for a full-resolution reproducible pass, or select
   **Local model pack** for a verified ONNX model.
3. Run detection. DOMStudio follows supported fracture evidence through weak
   sections, repairs short direction-consistent gaps, and immediately displays
   the resulting blue vector network over the teal/amber evidence and
   error-proxy overlay.
4. Tune continuation, bridge-gap, minimum-length, and small-loop rejection
   controls when the image scale or fracture style requires it, then rerun.
5. Use Trace with two anchors to add or correct difficult fractures that need
   expert guidance.
6. Review, undo, or clear vector traces, then export the network.

The **Search half-width** control is the main tracing speed/coverage trade-off:
smaller corridors reduce latency and memory, while wider corridors accommodate
strongly curved fractures. Record it with any reported result.

The analytical profile is a transparent domain-specific fallback, not a claim
of state-of-the-art learned performance. A publication should compare it and
the learned model on geographically held-out sites using a frozen protocol;
see [the research protocol](docs/RESEARCH_PROTOCOL.md).

## Architecture

```text
domstudio/
  application.py          Qt controller and background jobs
  core/                   raster/evidence models, detector, vectoriser, A*
  learning/               compact GeoTraceNet, losses, tiled inference
  evaluation/             geometry-first reproducibility metrics
  services/               raster I/O, safe sessions, vector export
  ui/                     canvas, workflow window, theme, controls
```

The scientific core has no Qt dependency. UI, file I/O, inference, and geometry
evaluation can therefore be tested independently.

## Strict local model packs

A v2 model pack is a self-contained directory containing `model.json` and its
referenced ONNX file. It is an auditable research artefact, not just a
checkpoint wrapper. The manifest must declare:

- identity: `name`, `model_version`, `architecture`, a relative `.onnx` path,
  and a non-empty `license` identifier or notice;
- integrity: the ONNX `model_sha256`, a canonical `pack_sha256` binding the
  manifest to those model bytes, and the source training-checkpoint SHA-256 in
  provenance;
- provenance: `code_revision`, `training_checkpoint_sha256`, `dataset_id`, and
  the geographic `split_id` used to produce the model;
- inference contract: NCHW float32 RGB input shape and name, percentile
  preprocessing and invalid-pixel policy, output names, sigmoid score ranges,
  axial `(cos 2θ, sin 2θ)` orientation encoding, tile size, and overlap;
- at least one explicit limitation.

`model-pack.example.json` is a field template only; its placeholder digests are
intentionally invalid. Use `python -m domstudio.learning.export_onnx --help` to
create a checked pack from a trusted GeoTraceNet state dictionary.

The loader recomputes the model and pack hashes, computes a canonical manifest
hash for run records, rejects path traversal and external ONNX tensor data,
runs `onnx.checker.check_model`, and verifies the ONNX Runtime tensor contract
before inference.

Exporting a v2 pack must also run the ONNX checker and a deterministic numerical
parity test: the frozen PyTorch model and CPU ONNX Runtime model receive the
same seeded inputs, and export fails if any declared output is non-finite,
mis-shaped, or outside the recorded tolerances. Archive the parity inputs,
tolerances, library versions, and result with any reported model.

The compact `GeoTraceNet` implementation predicts three fields:

- a sigmoid fracture-centreline score;
- axial orientation as `(cos 2θ, sin 2θ)`;
- a per-pixel learned error proxy.

The learned uncertainty output is **uncalibrated by default**. It may rank
locations where the model expects greater error, but it is not a probability of
error, confidence interval, or decomposition of epistemic and aleatoric
uncertainty. Give it probabilistic meaning only after a documented post-hoc
calibrator has been fitted on validation sites, frozen, hashed, and evaluated
on untouched geographic test sites.

No third-party checkpoint is bundled. This is intentional: a publishable model
must identify its training data, use reproducible geographic splits, establish
redistribution rights, and bind its exact contract and provenance to reported
results. The repository therefore contains architecture and tooling only, not
weights or benchmark numbers.

## Verification

The repository uses the standard library test runner, so tests work without an
additional test framework:

```powershell
python -m unittest discover -s tests -v
```

The suite covers full-resolution/nodata invariants, affine round trips,
deterministic evidence, direction-aware curved tracing, geometry metrics, safe
JSON sessions, and headless UI construction. Optional learned dependencies are
required for ONNX export/runtime checks.

A seeded synthetic benchmark records detector and vectorisation throughput,
network diagnostics, path latency, A* expansions, and runtime versions as JSON:

```powershell
python -m benchmarks.benchmark_pipeline --size 1024 --repeats 3
```

The synthetic benchmark is an engineering regression tool, not scientific
evidence. Publication results require a reproducible held-out geographic split,
expert agreement, frozen ablations, probability/error-proxy calibration,
end-to-end runtime and peak-memory measurements, and site-level uncertainty
intervals as specified in [the research protocol](docs/RESEARCH_PROTOCOL.md).

## Licence

DOMStudio is licensed under the GNU Affero General Public License v3.0; see
`LICENSE.txt`.
