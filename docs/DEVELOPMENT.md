# Development

## Layout

Runtime imports belong under `flake_searcher/`. UI files are resolved through
`flake_searcher.paths` and must not depend on the shell working directory.
Current detector code lives in `flake_searcher/detector.py`; training-only code
lives under `flake_searcher/training/`. Historical and experimental scripts are
preserved under `research/` and are not runtime dependencies.

Do not commit generated `.flake-searcher/`, checkpoints, Python caches, local
image collections, or microscope captures.

## Tests

With the full Conda environment:

```text
conda run --name flake-searcher python -m unittest discover -s tests -v
```

Tests use fake serial and stage objects and do not require microscope hardware.
The model asset test is skipped only when TensorFlow is absent.

Before committing a dependency update:

```text
regenerate and review requirements/runtime.lock.txt and requirements/full.lock.txt
python deploy_flake_searcher.py --setup full --preview
```

The requirements locks contain hashes for all resolved packages. Update the
embedded SAM2 source checksum only when intentionally changing its pinned
commit. Update `assets/manifest.json` whenever a tracked model or checkpoint
identity intentionally changes.

## Protected local material

`flakes/`, ignored AI image collections, local SAM2 source checkouts,
checkpoints, root `datapoints/`, and local deployment examples are user material.
Repository maintenance must not move, delete, or rewrite them without explicit
backup and review.
