from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent.parent / "training"

RESULTS_DIR: Path = Path(__file__).parent.parent / "generated_artifacts"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
