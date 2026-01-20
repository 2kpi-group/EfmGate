# Réexport des fonctions dataset
from .dataset import load_dataset, split_data, r2_score

from .models.efm_gate import EfmLSTM, EfmLSTMPredictor


# -----------------------------
# Définition de ce qui sera accessible via 'from efmgate import *'
# -----------------------------
__all__ = [
    "load_dataset",
    "split_data",
    "r2_score",
    "EfmLSTM",
    "EfmLSTMPredictor",
    # "some_data_function"  # décommenter si tu ajoutes des fonctions dans data
]

