# Réexporter les fonctions du dataset

from .data.dataset import generate_ou_signal, split_data, r2_score

# Réexporter les classes du modèle
from .models.efm_gate import EfmLSTM, EfmLSTMPredictor

# Définir ce qui sera importé avec 'from efmgate import *'

__all__ = [
    "generate_ou_signal",
    "split_data",
    "r2_score",
    "EfmLSTM",
    "EfmLSTMPredictor"
]

