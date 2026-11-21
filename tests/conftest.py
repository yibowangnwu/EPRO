import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if os.getenv("VERL_ROOT"):
    sys.path.insert(0, os.environ["VERL_ROOT"])
