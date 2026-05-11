import os
from pathlib import Path

# mace-torch pins e3nn==0.4.4 which is too old to use weights_only=False.
# This is necessary in torch.load() for torch>=2.6.0
# We workaround using this environment variable.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "true")

ROOT_DIR = Path(__file__).parent.parent
TOP_DIR = ROOT_DIR.parent
DATA_DIR = TOP_DIR.joinpath("data/")
