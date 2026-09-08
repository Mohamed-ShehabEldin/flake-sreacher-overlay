"""Standalone UI responsiveness diagnostics; not imported by the application."""

from __future__ import annotations

import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .evaluation import EvaluationError, prepare_evaluation_paths, sha256_file


def configure_console_output(*streams) -> None:
    """Escape characters unsupported by a diagnostic console's encoding."""
    for stream in streams:
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(errors="backslashreplace")


def summarize_heartbeats(samples: list[dict], target_interval_ms: float) -> dict:
    by_phase = defaultdict(list)
    for sample in samples:
        by_phase[sample["phase"]].append(float(sample["interval_ms"]))

    def summarize(values):
        return {
            "samples": len(values),
            "median_ms": statistics.median(values),
            "p95_ms": float(np.percentile(values, 95)),
            "max_ms": max(values),
            "max_excess_ms": max(values) - target_interval_ms,
        }

    return {
        "target_interval_ms": target_interval_ms,
        "overall": summarize([float(sample["interval_ms"]) for sample in samples]),
        "by_phase": {phase: summarize(values) for phase, values in sorted(by_phase.items())},
    }


def run_ui_probe(
    *,
    model_path: str | Path,
    input_path: str | Path,
    output_dir: str | Path,
    ratio: int = 5,
    batch_size: int = 23000,
    radius: int = 2,
    heartbeat_interval_ms: int = 10,
    baseline_ms: int = 250,
    timeout_ms: int = 300000,
) -> dict:
    if heartbeat_interval_ms <= 0 or baseline_ms <= 0 or timeout_ms <= 0:
        raise EvaluationError("Heartbeat, baseline, and timeout intervals must be positive.")
    images, output = prepare_evaluation_paths([input_path], output_dir)
    if len(images) != 1:
        raise EvaluationError("The UI probe requires exactly one explicit image file.")
    image_path = images[0]
    model_file = Path(model_path).expanduser().resolve()
    if not model_file.is_file():
        raise EvaluationError(f"Model file does not exist: {model_file}")
    input_hash_before = sha256_file(image_path)
    model_hash_before = sha256_file(model_file)

    from PyQt5.QtCore import QCoreApplication, QTimer

    from .a_eye_tab import InferenceWorker
    from .pipeline import AutoScanPipeline

    app = QCoreApplication.instance() or QCoreApplication([])
    heartbeat_samples = []
    state = {
        "phase": "baseline",
        "last_tick_ns": time.perf_counter_ns(),
        "model_load_ms": None,
        "inference_ms": None,
        "inference_stats": None,
        "error": None,
        "worker": None,
        "timed_out": False,
    }
    timer = QTimer()
    timer.setInterval(heartbeat_interval_ms)

    def heartbeat():
        now = time.perf_counter_ns()
        heartbeat_samples.append(
            {
                "phase": state["phase"],
                "interval_ms": (now - state["last_tick_ns"]) / 1_000_000,
            }
        )
        state["last_tick_ns"] = now

    timer.timeout.connect(heartbeat)
    timer.start()
    pipeline = AutoScanPipeline()

    def finish():
        timer.stop()
        app.quit()

    def inference_done(_mask, _overlay, stats):
        state["inference_ms"] = (time.perf_counter_ns() - inference_started[0]) / 1_000_000
        state["inference_stats"] = stats
        state["phase"] = "post_inference"
        QTimer.singleShot(baseline_ms, finish)

    def inference_error(message):
        state["error"] = message
        finish()

    inference_started = [0]

    def start_inference():
        state["phase"] = "background_inference"
        worker = InferenceWorker(pipeline, os.fspath(image_path), ratio, batch_size, radius)
        state["worker"] = worker
        worker.done.connect(inference_done)
        worker.error.connect(inference_error)
        inference_started[0] = time.perf_counter_ns()
        worker.start()

    def load_model_on_ui_thread():
        state["phase"] = "ui_thread_model_load"
        started = time.perf_counter_ns()
        try:
            pipeline.load_model_from_path(model_file)
        except Exception as error:
            state["error"] = str(error)
            finish()
            return
        state["model_load_ms"] = (time.perf_counter_ns() - started) / 1_000_000
        QTimer.singleShot(baseline_ms, start_inference)

    def timeout():
        state["timed_out"] = True
        state["error"] = f"UI diagnostic exceeded {timeout_ms} ms"
        finish()

    QTimer.singleShot(baseline_ms, load_model_on_ui_thread)
    QTimer.singleShot(timeout_ms, timeout)
    app.exec_()
    worker = state["worker"]
    if worker is not None and worker.isRunning():
        worker.wait()
    if state["error"]:
        raise EvaluationError(state["error"])
    if not heartbeat_samples:
        raise EvaluationError("The UI diagnostic recorded no event-loop heartbeats.")

    input_hash_after = sha256_file(image_path)
    model_hash_after = sha256_file(model_file)
    if input_hash_after != input_hash_before or model_hash_after != model_hash_before:
        raise EvaluationError("Input or model integrity changed during the UI diagnostic.")

    report = {
        "schema_version": "1.0",
        "diagnostic_only": True,
        "production_ui_modified": False,
        "live_capture_used": False,
        "hardware_used": False,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": {
            "path": os.fspath(image_path),
            "sha256_before": input_hash_before,
            "sha256_after": input_hash_after,
        },
        "model": {
            "path": os.fspath(model_file),
            "sha256_before": model_hash_before,
            "sha256_after": model_hash_after,
        },
        "parameters": {
            "ratio": ratio,
            "batch_size": batch_size,
            "radius": radius,
            "heartbeat_interval_ms": heartbeat_interval_ms,
        },
        "model_load_ui_thread_ms": state["model_load_ms"],
        "background_inference_wall_ms": state["inference_ms"],
        "detector_internal_ms": float(state["inference_stats"]["elapsed"] * 1000),
        "heartbeats": summarize_heartbeats(heartbeat_samples, heartbeat_interval_ms),
    }
    (output / "ui_responsiveness.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
