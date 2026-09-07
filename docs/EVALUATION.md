# Detector evaluation

The `v0.2.0` detector is the control for this evaluation tooling. The evaluator
calls the unchanged `flake_searcher.detector.test_grid_batched` implementation
and does not alter class handling, thresholds, connected components, models, or
application behavior. The frozen source and model identities are recorded in
`evaluation/control-v0.2.0.json`.

## Read-only file evaluation

Supply every input and the output directory explicitly:

```text
python tools/evaluate_detector.py \
  --model "models/detector/WSe2_EVE Microscope_20x_ALPHA.h5" \
  --input "/absolute/path/to/reference-image.png" \
  --output-dir "/absolute/path/to/new-empty-results" \
  --ratio 5 --batch-size 23000 --radius 2 --repeats 10 \
  --save-masks --save-overlays
```

An input may be an image file or directory. Directory traversal is recursive.
An individually supplied file protects its parent directory as the input data
root. The output must be a separate, new or empty directory with no ancestor or
descendant relationship to any input root. The command refuses unsafe paths.
It also categorically refuses output below the repository's `flakes/`,
`datapoints/`, or `deploy example_zmeter-deploy-main/` directories.

Inputs are hashed before and after evaluation. Reports, CSV timing records,
masks, and overlays are written only below the output directory. Masks and
overlays are optional derived products; originals are never copied or rewritten.

`report.json` records the Git commit, platform, Python and relevant package
versions, selected model path and hash, detector parameters, TensorFlow import
and model-load times, decode time, detector and model-prediction latency, grid
size, raw class counts, retained components, output hashes, throughput, and
repeatability. `per_image.csv` summarizes images and `repeats.csv` retains every
timed run.

The current detector's internal time ends before circle rendering. The evaluator
therefore reports both that value and complete detector-call wall time. First-run
and warmed behavior remain visible in the repeat rows rather than being merged.

## Scientific interpretation

Without complete human-reviewed ground truth, the report sets accuracy,
precision, recall, false-positive rate, and miss rate to JSON `null` and explains
that they are unavailable. Raw class counts, component counts, and folder names
are descriptive only. Existing clicked RGB points and training folders must not
be promoted to detector ground truth.

The evaluator observes all three raw model classes. It deliberately preserves a
known control behavior: the released binary connected-component step treats any
nonzero class as foreground, so class 2 can survive as a filtered detection even
though `raw` counts class 1 only. A later detector milestone may evaluate that
behavior after the baseline is established.

## UI responsiveness diagnostic

The optional probe exercises only file input:

```text
python tools/probe_ui_responsiveness.py \
  --model "/absolute/path/to/model.h5" \
  --input "/absolute/path/to/reference-image.png" \
  --output-dir "/absolute/path/to/new-empty-ui-results"
```

It measures synchronous model loading on a Qt event loop and heartbeat latency
while the existing A-Eye inference worker runs. It does not open the production
window, render results, capture the screen, start Auto Scan, use serial hardware,
or instrument production UI code.

For Mac/Windows comparison, use the same commit, model hash, input hashes, and
parameters. Run each command in the pinned runtime environment under comparable
power conditions. Compare output hashes first, then class/component counts and
latency distributions. File-input results isolate inference; live microscope and
display-color behavior remain outside this diagnostic.

## Future annotation protocol

`evaluation/annotation.schema.json` defines the storage format but contains no
annotations. The minimum future task is:

1. Select images not used to train the evaluated model and preserve their hashes.
2. Write the scientific inclusion/exclusion definition before reviewing images.
3. Have a named domain reviewer mark every valid, excluded, and uncertain region
   with an image-coordinate polygon and mark each image as completely reviewed.
4. Include confirmed negative images and resolve uncertain regions before using
   the set for false-positive or miss-rate claims.
5. Define and review the detection-to-region matching rule before calculating
   quality metrics.

Draft or incomplete annotations are not ground truth. This milestone does not
implement quality metrics from the schema and makes no scientific performance
claim.
