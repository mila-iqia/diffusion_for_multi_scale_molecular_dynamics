from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"

RESULTS_DIR: Path = Path(__file__).parent / "generated_artifacts"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
