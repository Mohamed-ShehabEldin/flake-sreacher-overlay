"""Measure UI-thread model loading and background file-inference responsiveness."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flake_searcher.evaluation import EvaluationError
from flake_searcher.ui_diagnostics import run_ui_probe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="Exactly one explicit image file")
    parser.add_argument("--output-dir", required=True, help="New or empty separate directory")
    parser.add_argument("--ratio", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=23000)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--heartbeat-ms", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        run_ui_probe(
            model_path=args.model,
            input_path=args.input,
            output_dir=args.output_dir,
            ratio=args.ratio,
            batch_size=args.batch_size,
            radius=args.radius,
            heartbeat_interval_ms=args.heartbeat_ms,
        )
    except EvaluationError as error:
        print(f"UI diagnostic failed: {error}", file=sys.stderr)
        return 2
    print(f"UI diagnostic: {Path(args.output_dir).expanduser().resolve() / 'ui_responsiveness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
