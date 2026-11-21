"""Run random_forest; model definition: src/models/random_forest.py."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"
from .train_ml import main

if __name__ == "__main__":
    main(default_model="random_forest")
