"""Run svm; model definition: src/models/svm.py."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"
from .train_ml import main

if __name__ == "__main__":
    main(default_model="svm")
