from pathlib import Path

ROOT_DIR = Path(__file__).parent
TOP_DIR = ROOT_DIR.parent
ANALYSIS_RESULTS_DIR = TOP_DIR.joinpath("analysis_results/")
ANALYSIS_RESULTS_DIR.mkdir(exist_ok=True)
