"""Run the full pipeline. Usage: uv run python scripts/run_pipeline.py [stage ...]

Stages: warehouse forecasting customers insights (default: all).
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecom.config import raw_dir  # noqa: E402
from ecom.pipeline import run  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.info("data source: %s", raw_dir())
    run(sys.argv[1:] or None)
