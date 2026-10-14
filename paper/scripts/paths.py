"""Where the paper scripts read inputs and write outputs. Override with environment variables.

MYOJI_HDEMG     folder with the HD-EMG recordings Sub{n}_emg.pkl / Sub{n}_kinematics.pkl
                (available from the authors on reasonable request)
MYOJI_EMG2POSE  folder with the emg2pose dataset (or use the scripts' range-request options)
MYOJI_ASSETS    trained-artifact bundle: learned/tokenizer.pt, learned/tokenizer-full.pt, spar/...
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # paper/
REPO = ROOT.parent
DATA = ROOT / "data"  # shipped inputs
REFERENCE = DATA / "reference"  # frozen window selections and protocols the campaigns build on
SUPPLEMENTARY = DATA / "supplementary"  # Supplementary Data 1-9
RESULTS = ROOT / "results"  # run roots written by the campaigns
OUTPUT = ROOT / "output"  # figures and tables
HDEMG = Path(os.environ.get("MYOJI_HDEMG", REPO / "data/hdemg"))
EMG2POSE = Path(os.environ.get("MYOJI_EMG2POSE", REPO / "data/emg2pose"))
ASSETS = Path(os.environ.get("MYOJI_ASSETS", REPO.parent / "myoji-assets"))
LEARNED_TOKENIZER = ASSETS / "learned/tokenizer.pt"  # frozen release format (Tokenizer)
LEARNED_TOKENIZER_FULL = ASSETS / "learned/tokenizer-full.pt"  # with the spectral head (load_universal_checkpoint)
