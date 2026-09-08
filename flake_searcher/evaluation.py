"""Read-only evaluation tooling for the released detector implementation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from .detector import test_grid_batched


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
UNAVAILABLE_QUALITY_METRICS = {
    "accuracy": None,
    "precision": None,
    "recall": None,
    "false_positive_rate": None,
    "miss_rate": None,
}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROTECTED_DATA_ROOTS = (
    PROJECT_ROOT / "flakes",
    PROJECT_ROOT / "datapoints",
    PROJECT_ROOT / "deploy example_zmeter-deploy-main",
)


class EvaluationError(RuntimeError):
    """Raised when an evaluation cannot run without risking input data."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_source_sha256(path: Path) -> str:
    """Hash UTF-8 source text after normalizing CRLF line endings to LF."""
    source = path.read_bytes().decode("utf-8")
    canonical_source = source.replace("\r\n", "\n")
    return hashlib.sha256(canonical_source.encode("utf-8")).hexdigest()


def hash_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _resolved_input_roots(input_paths: Iterable[str | Path]) -> tuple[list[Path], list[Path]]:
    roots = []
    images = []
    for supplied in input_paths:
        path = Path(supplied).expanduser().resolve()
        if not path.exists():
            raise EvaluationError(f"Input does not exist: {path}")
        if path.is_dir():
            roots.append(path)
            candidates = path.rglob("*")
        elif path.is_file():
            roots.append(path.parent)
            candidates = (path,)
        else:
            raise EvaluationError(f"Input is not a regular file or directory: {path}")
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(candidate.resolve())

    unique_images = sorted(set(images), key=lambda item: os.fspath(item).casefold())
    if not unique_images:
        raise EvaluationError("No supported image files were found in the supplied inputs.")
    return sorted(set(roots)), unique_images


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def prepare_evaluation_paths(
    input_paths: Iterable[str | Path], output_dir: str | Path
) -> tuple[list[Path], Path]:
    roots, images = _resolved_input_roots(input_paths)
    output = Path(output_dir).expanduser().resolve()
    for protected_root in PROTECTED_DATA_ROOTS:
        protected = protected_root.resolve()
        if output == protected or protected in output.parents:
            raise EvaluationError(f"Output directory is inside protected data: {protected}")
    for root in roots:
        if _overlaps(root, output):
            raise EvaluationError(
                f"Output directory must be separate from every input dataset: {output} overlaps {root}"
            )
    if output.exists():
        if not output.is_dir():
            raise EvaluationError(f"Output path is not a directory: {output}")
        if any(output.iterdir()):
            raise EvaluationError(f"Output directory must be empty: {output}")
    else:
        output.mkdir(parents=True)
    return images, output


class PredictionRecorder:
    """Observe model output without changing the detector's call contract."""

    def __init__(self, model):
        self.model = model
        self.last_predictions = None
        self.last_elapsed_ms = None

    def reset(self) -> None:
        self.last_predictions = None
        self.last_elapsed_ms = None

    def predict(self, *args, **kwargs):
        started = time.perf_counter_ns()
        predictions = self.model.predict(*args, **kwargs)
        self.last_elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        self.last_predictions = np.asarray(predictions)
        return predictions


def _percentiles(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p95": float(np.percentile(values, 95)),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def _component_summary(mask: np.ndarray) -> dict:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask != 0).astype(np.uint8), connectivity=8
    )
    areas = [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)]
    return {"count": count - 1, "areas_grid_points": areas}


def _safe_stem(index: int, path: Path, file_hash: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in path.stem)
    return f"{index:04d}_{cleaned[:80]}_{file_hash[:12]}"


def evaluate_image(
    image_path: Path,
    recorder: PredictionRecorder,
    *,
    ratio: int,
    batch_size: int,
    radius: int,
    repeats: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    decode_started = time.perf_counter_ns()
    image = cv2.imread(os.fspath(image_path))
    decode_ms = (time.perf_counter_ns() - decode_started) / 1_000_000
    if image is None:
        raise EvaluationError(f"OpenCV could not decode input image: {image_path}")

    repeat_records = []
    last_mask = None
    last_overlay = None
    for repeat_index in range(repeats):
        recorder.reset()
        started = time.perf_counter_ns()
        mask, overlay, stats = test_grid_batched(
            image,
            recorder,
            ratio=ratio,
            batch_size=batch_size,
            radius=radius,
            thickness=-1,
        )
        detector_ms = (time.perf_counter_ns() - started) / 1_000_000
        if recorder.last_predictions is None or recorder.last_elapsed_ms is None:
            raise EvaluationError("The detector completed without observable model predictions.")

        raw_classes = np.argmax(recorder.last_predictions, axis=1)
        class_count = recorder.last_predictions.shape[1]
        class_counts = {
            str(index): int(np.sum(raw_classes == index)) for index in range(class_count)
        }
        mask_hash = hash_array(mask)
        overlay_hash = hash_array(overlay)
        raw_class_hash = hash_array(raw_classes)
        combined = hashlib.sha256(
            f"{mask_hash}:{overlay_hash}:{raw_class_hash}".encode("ascii")
        ).hexdigest()
        repeat_records.append(
            {
                "repeat": repeat_index + 1,
                "detector_wall_ms": detector_ms,
                "detector_internal_ms": float(stats["elapsed"] * 1000),
                "model_predict_ms": recorder.last_elapsed_ms,
                "raw_class_counts": class_counts,
                "raw_class_hash": raw_class_hash,
                "mask_hash": mask_hash,
                "overlay_hash": overlay_hash,
                "output_hash": combined,
                "raw_class_1_points": int(stats["raw"]),
                "filtered_points": int(stats["filtered"]),
                "components": _component_summary(mask),
            }
        )
        last_mask = mask
        last_overlay = overlay

    output_hashes = {record["output_hash"] for record in repeat_records}
    detector_times = [record["detector_wall_ms"] for record in repeat_records]
    predict_times = [record["model_predict_ms"] for record in repeat_records]
    summary = {
        "path": os.fspath(image_path),
        "sha256": sha256_file(image_path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "decode_ms": decode_ms,
        "grid_rows": int(last_mask.shape[0]),
        "grid_cols": int(last_mask.shape[1]),
        "grid_points": int(last_mask.size),
        "repeatable": len(output_hashes) == 1,
        "unique_output_hashes": sorted(output_hashes),
        "detector_wall_ms": _percentiles(detector_times),
        "model_predict_ms": _percentiles(predict_times),
        "repeats": repeat_records,
    }
    return summary, last_mask, last_overlay


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for distribution in ("numpy", "opencv-python", "tensorflow", "keras"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _load_model(model_path: Path):
    import_started = time.perf_counter_ns()
    from tensorflow import keras
    import_ms = (time.perf_counter_ns() - import_started) / 1_000_000

    load_started = time.perf_counter_ns()
    model = keras.models.load_model(os.fspath(model_path), compile=False)
    load_ms = (time.perf_counter_ns() - load_started) / 1_000_000
    return model, import_ms, load_ms


def _write_csvs(output: Path, image_records: list[dict]) -> None:
    with (output / "per_image.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "path",
                "sha256",
                "width",
                "height",
                "grid_points",
                "decode_ms",
                "repeatable",
                "detector_median_ms",
                "detector_p95_ms",
                "predict_median_ms",
                "component_count",
                "filtered_points",
                "class_0",
                "class_1",
                "class_2",
            ),
        )
        writer.writeheader()
        for image in image_records:
            first = image["repeats"][0]
            counts = first["raw_class_counts"]
            writer.writerow(
                {
                    "path": image["path"],
                    "sha256": image["sha256"],
                    "width": image["width"],
                    "height": image["height"],
                    "grid_points": image["grid_points"],
                    "decode_ms": image["decode_ms"],
                    "repeatable": image["repeatable"],
                    "detector_median_ms": image["detector_wall_ms"]["median"],
                    "detector_p95_ms": image["detector_wall_ms"]["p95"],
                    "predict_median_ms": image["model_predict_ms"]["median"],
                    "component_count": first["components"]["count"],
                    "filtered_points": first["filtered_points"],
                    "class_0": counts.get("0", 0),
                    "class_1": counts.get("1", 0),
                    "class_2": counts.get("2", 0),
                }
            )

    with (output / "repeats.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "path",
            "repeat",
            "detector_wall_ms",
            "detector_internal_ms",
            "model_predict_ms",
            "raw_class_1_points",
            "filtered_points",
            "component_count",
            "raw_class_hash",
            "mask_hash",
            "overlay_hash",
            "output_hash",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for image in image_records:
            for repeat in image["repeats"]:
                writer.writerow(
                    {
                        "path": image["path"],
                        "repeat": repeat["repeat"],
                        "detector_wall_ms": repeat["detector_wall_ms"],
                        "detector_internal_ms": repeat["detector_internal_ms"],
                        "model_predict_ms": repeat["model_predict_ms"],
                        "raw_class_1_points": repeat["raw_class_1_points"],
                        "filtered_points": repeat["filtered_points"],
                        "component_count": repeat["components"]["count"],
                        "raw_class_hash": repeat["raw_class_hash"],
                        "mask_hash": repeat["mask_hash"],
                        "overlay_hash": repeat["overlay_hash"],
                        "output_hash": repeat["output_hash"],
                    }
                )


def run_evaluation(
    *,
    model_path: str | Path,
    input_paths: Iterable[str | Path],
    output_dir: str | Path,
    ratio: int = 5,
    batch_size: int = 23000,
    radius: int = 2,
    repeats: int = 3,
    save_masks: bool = False,
    save_overlays: bool = False,
    model_loader: Callable[[Path], object] | None = None,
) -> dict:
    if ratio <= 0 or batch_size <= 0 or radius <= 0 or repeats <= 0:
        raise EvaluationError("ratio, batch size, radius, and repeats must all be positive.")
    model_file = Path(model_path).expanduser().resolve()
    if not model_file.is_file():
        raise EvaluationError(f"Model file does not exist: {model_file}")

    images, output = prepare_evaluation_paths(input_paths, output_dir)
    before_hashes = {os.fspath(path): sha256_file(path) for path in images}
    model_hash = sha256_file(model_file)

    if model_loader is None:
        model, import_ms, load_ms = _load_model(model_file)
    else:
        load_started = time.perf_counter_ns()
        model = model_loader(model_file)
        import_ms = 0.0
        load_ms = (time.perf_counter_ns() - load_started) / 1_000_000
    recorder = PredictionRecorder(model)

    image_records = []
    run_started = time.perf_counter_ns()
    for index, image_path in enumerate(images, start=1):
        record, mask, overlay = evaluate_image(
            image_path,
            recorder,
            ratio=ratio,
            batch_size=batch_size,
            radius=radius,
            repeats=repeats,
        )
        stem = _safe_stem(index, image_path, before_hashes[os.fspath(image_path)])
        if save_masks:
            mask_dir = output / "masks"
            mask_dir.mkdir(exist_ok=True)
            mask_path = mask_dir / f"{stem}.png"
            if not cv2.imwrite(os.fspath(mask_path), (mask * 255).astype(np.uint8)):
                raise EvaluationError(f"Could not write mask: {mask_path}")
            record["mask_output"] = os.fspath(mask_path)
        if save_overlays:
            overlay_dir = output / "overlays"
            overlay_dir.mkdir(exist_ok=True)
            overlay_path = overlay_dir / f"{stem}.png"
            if not cv2.imwrite(os.fspath(overlay_path), overlay):
                raise EvaluationError(f"Could not write overlay: {overlay_path}")
            record["overlay_output"] = os.fspath(overlay_path)
        image_records.append(record)
    run_ms = (time.perf_counter_ns() - run_started) / 1_000_000

    after_hashes = {os.fspath(path): sha256_file(path) for path in images}
    if before_hashes != after_hashes:
        changed = sorted(path for path in before_hashes if before_hashes[path] != after_hashes[path])
        raise EvaluationError(f"Input integrity check failed; changed files: {changed}")

    total_detector_ms = sum(
        repeat["detector_wall_ms"]
        for image in image_records
        for repeat in image["repeats"]
    )
    total_grid_points = sum(image["grid_points"] * repeats for image in image_records)
    report = {
        "schema_version": "1.0",
        "control": "v0.2.0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "git_commit": _git_commit(),
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "packages": _package_versions(),
            "model_path": os.fspath(model_file),
            "model_sha256": model_hash,
        },
        "parameters": {
            "ratio": ratio,
            "batch_size": batch_size,
            "radius": radius,
            "thickness": -1,
            "repeats": repeats,
        },
        "initialization": {
            "tensorflow_import_ms": import_ms,
            "model_load_ms": load_ms,
        },
        "input_integrity": {
            "verified": True,
            "before": before_hashes,
            "after": after_hashes,
        },
        "quality_metrics": {
            "available": False,
            "reason": "No complete human-reviewed ground-truth annotations were supplied.",
            **UNAVAILABLE_QUALITY_METRICS,
        },
        "repeatability": {
            "all_images_repeatable": all(image["repeatable"] for image in image_records),
            "nonrepeatable_images": [
                image["path"] for image in image_records if not image["repeatable"]
            ],
        },
        "throughput": {
            "evaluation_wall_ms": run_ms,
            "detector_images_per_second": (
                len(image_records) * repeats * 1000 / total_detector_ms
            ),
            "detector_grid_points_per_second": total_grid_points * 1000 / total_detector_ms,
        },
        "images": image_records,
    }
    _write_csvs(output, image_records)
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Explicit detector .h5 path")
    parser.add_argument(
        "--input",
        required=True,
        action="append",
        nargs="+",
        help="Explicit image file or directory; may be supplied more than once",
    )
    parser.add_argument("--output-dir", required=True, help="New or empty separate directory")
    parser.add_argument("--ratio", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=23000)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--save-overlays", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = [path for group in args.input for path in group]
    try:
        run_evaluation(
            model_path=args.model,
            input_paths=inputs,
            output_dir=args.output_dir,
            ratio=args.ratio,
            batch_size=args.batch_size,
            radius=args.radius,
            repeats=args.repeats,
            save_masks=args.save_masks,
            save_overlays=args.save_overlays,
        )
    except EvaluationError as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2
    print(f"Evaluation report: {Path(args.output_dir).expanduser().resolve() / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
