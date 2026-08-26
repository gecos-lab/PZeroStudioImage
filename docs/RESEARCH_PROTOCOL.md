# DOMStudio 2 research protocol

This protocol separates a usable application from a defensible scientific
claim. DOMStudio is an experimental instrument. The repository currently
bundles no learned weights and reports no empirical accuracy, generalisation,
calibration, or runtime result. Add numerical claims only after completing and
archiving the applicable protocol below.

## Claim boundaries

Dense fracture detection, human-guided path tracing, graph construction, and
operator efficiency are different tasks. Evaluate and report them separately.
Do not attribute a mask comparison to A*, infer geological validity from a
synthetic benchmark, or describe a sigmoid output as calibrated merely because
it lies in `[0, 1]`.

In particular, the learned uncertainty head is an **uncalibrated error proxy**.
Without post-hoc calibration it may be evaluated as a ranking of likely errors,
but it is not a probability of error, a confidence interval, or a quantified
epistemic/aleatoric uncertainty. Any calibrated interpretation must name its
target event, calibration method, fitting data, and frozen calibration artefact.

## Preregistered questions

1. Does adding axial orientation improve centreline localisation and guided
   path accuracy relative to a probability-only model?
2. Does the raw uncertainty proxy improve error ranking or selective prediction,
   and does a validation-fitted calibrator generalise to unseen sites?
3. Does direction-state A* reduce tracing error and corrective interaction
   relative to pixel-state shortest-path search?
4. Does an anchor-defined corridor reduce latency and peak memory without
   reducing trace completion rate?

Define the primary endpoint and smallest meaningful effect for each question
before training the final model.

## Geographic data split

- Group all images, crops, tiles, repeated surveys, and reconstructions from the
  same outcrop or scene before splitting. Neighbouring tiles must never cross a
  split boundary.
- Assign distinct geological sites to training, model-selection/calibration,
  and final test sets. Keep the test sites inaccessible until the method,
  thresholds, tracer settings, and statistical plan are frozen.
- If the number of sites is limited, preregister grouped leave-one-site-out or
  nested site-level cross-validation. Never substitute random tile folds.
- Record sensor, acquisition date, ground sampling distance, lithology,
  illumination, weathering, scale, CRS, preprocessing, and annotation
  provenance for every scene.
- Freeze machine-readable scene and tile manifests. Archive their SHA-256
  digests, exclusion decisions, and the script that generated the split.
- Evaluate at least one genuinely external geography when making a broad
  generalisation claim.

Double-annotate a representative, site-stratified subset. Report annotator
training, blinding, centreline convention, adjudication rules, and both
inter-annotator agreement and model-to-each-annotator results. Thin geological
traces have real positional and interpretive ambiguity.

## Strict model-pack v2 release gate

Each evaluated learned model must be frozen as the exact local v2 model pack
used for inference. Its manifest must contain:

- model identity: name, semantic model version, architecture, relative ONNX
  filename, and a licence identifier or complete licence notice;
- integrity: lowercase SHA-256 of the ONNX bytes and a canonical pack SHA-256
  binding the complete manifest to those bytes;
- provenance: exact `code_revision`, source `training_checkpoint_sha256`,
  `dataset_id`, and geographic `split_id`;
- a fixed inference contract: NCHW float32 RGB input name and shape,
  preprocessing percentiles and invalid-pixel fill, output names and
  activations, axial `cos(2θ), sin(2θ)` encoding, tile size, and overlap;
- a non-empty limitations list.

Archive the loader-computed canonical manifest SHA-256 alongside the required
model and pack hashes. The research record must additionally include hashes for
the dataset manifest, split manifest, training configuration, and any post-hoc
calibrator. A changed model, contract, provenance field, or limitation requires
a new model version and pack digest.

Pack export is successful only if all of these checks pass:

1. `onnx.checker.check_model` accepts the self-contained graph; external tensor
   data, path escape, and undeclared tensors are rejected.
2. ONNX Runtime exposes the declared float32 input and all required output
   names, shapes, ranges, and finite values.
3. The frozen PyTorch model and CPU ONNX Runtime model receive the same seeded
   edge-case and random inputs. Each probability, orientation, and uncertainty
   output must satisfy preregistered absolute and relative parity tolerances.
4. The parity inputs, tolerances, maximum observed differences, PyTorch/ONNX/
   ONNX Runtime versions, opset, and export command are written to the release
   record.

Do not download weights at runtime or silently fall back to another model.
Release weights only when the training data, checkpoint, code, and third-party
licences permit redistribution. Otherwise publish the manifest and evaluation
provenance without the restricted bytes and state how authorised reviewers can
reproduce the result.

## Comparisons and ablations

The desktop application intentionally contains no legacy filter tabs. If a
paper needs Canny, Sobel, or another historical comparison, run a frozen
baseline in a separate benchmark environment and record every parameter.

At minimum, freeze and evaluate:

- probability head only;
- probability plus orientation;
- probability plus orientation plus the raw uncertainty proxy;
- raw proxy versus the same output with a validation-fitted calibrator;
- uncertainty path cost disabled versus enabled;
- pixel-state shortest path versus direction-state A*;
- zero versus fitted curvature penalty;
- preregistered corridor radii at fixed endpoint-distance fractions;
- single-threshold vectorisation versus strong/weak hysteresis;
- gap repair disabled versus direction-and-evidence-gated gap repair;
- spur/fragment pruning and small-loop rejection disabled versus enabled;
- analytical detector versus the exact learned v2 model pack;
- tile overlap and test-time tiling choices used by the learned model.

Train comparable variants with the same sites, augmentation budget, stopping
rule, and model-selection endpoint. Select parameters on validation sites only;
run the frozen final comparison on test sites once.

## Outcomes and calibration

For dense centreline detection, report tolerance-aware precision, recall, and
F1 at multiple geologically meaningful tolerances, plus symmetric distance and
Hausdorff distance. State whether empty masks and nodata regions are included.

For the sigmoid fracture score, report reliability diagrams, Brier score, and a
predeclared calibration error measure by site. Class imbalance and spatial
correlation make a pooled pixel-only calibration number insufficient.

For the raw uncertainty proxy, first define the error it is intended to rank,
such as centreline miss distance or incorrect pixel classification. Report
risk-coverage curves and ranking metrics on validation and untouched test sites.
If applying temperature scaling, isotonic regression, or another post-hoc
mapping, fit it only on calibration sites, freeze and hash it, then report both
raw and calibrated results. Do not reuse test labels to choose bins, thresholds,
or the calibration family.

For vector traces, report symmetric distance, Hausdorff distance, endpoint
error, relative length error, and completion rate. Before claiming improved
topology, add explicit precision/recall for endpoints, junctions, intersections,
connected components, and edge connectivity rather than relying on a single
mask IoU or junction count. Freeze the strong and weak evidence thresholds,
maximum bridge distance and angle, minimum vector/spur lengths, loop cutoff,
and simplification tolerance on validation sites. Report false joins separately
from missed joins: they have different geological consequences.

For interaction, record time from the second anchor to acceptance, rejected
paths, corrective clicks, manual edits, and total interpretation time. Use a
counterbalanced task order, record operator experience, and report results per
operator and site.

## Runtime protocol

Report cold and warm runs separately and disclose image dimensions, tile size,
CPU, RAM, GPU, driver, execution provider, thread count, and software versions.
Measure:

- raster decoding and display conversion;
- detector inference and overlay preparation;
- tracing latency, searchable states, and A* expansions by endpoint distance;
- vector/network construction and export;
- peak resident CPU memory and, where applicable, peak accelerator memory.

Report median, dispersion, and tail latency rather than a best run. State which
steps are excluded from any throughput figure. The seeded synthetic benchmark
is suitable for regression detection only and cannot support a real-world speed
or accuracy claim.

## Statistical analysis

Treat geological site—not tile—as the primary independent sampling unit.
Retain per-scene and per-site results, aggregate according to the preregistered
hierarchy, and report site-level confidence intervals, for example by grouped
bootstrap. Include failures and timeouts in completion-rate and runtime
summaries. Document multiplicity handling, missing data, and every deviation
from the frozen protocol.

## Reproducibility record

Archive for every reported run:

- source revision, environment lock, random seeds, and deterministic settings;
- raw-data inventory plus dataset, scene, tile, and split manifest hashes;
- the complete v2 pack, model/checkpoint/manifest/pack hashes, licence,
  provenance, contract, limitations, and export parity report;
- calibration target, fitting split, method, parameters, and artefact hash;
- detector, tiling, threshold, tracing, and vectorisation configuration;
- unrounded per-image, per-trace, per-site, and per-operator outcomes;
- hardware, provider, runtime, wall-time protocol, and peak-memory records;
- exported vectors in the original CRS and the script used to compute tables
  and figures.

Until those records and held-out results exist, describe DOMStudio as a
research prototype and report no state-of-the-art, clinical-style confidence,
or deployment-readiness claim.
