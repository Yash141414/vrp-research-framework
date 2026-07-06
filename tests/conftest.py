import sys
from pathlib import Path

# Insert the repo root so all modules are importable without the personal
# absolute path prefix (Apology.Proj.Nifty_momentum_system.nifty_data_layer.*).
sys.path.insert(0, str(Path(__file__).parent.parent))
