from pathlib import Path

PLOTS_OUTPUT_DIRECTORY = Path(__file__).parent.parent / "images"
PLOTS_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
