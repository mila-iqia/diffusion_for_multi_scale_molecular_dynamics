from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "generated_artifacts"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
