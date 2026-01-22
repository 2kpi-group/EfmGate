# Nouveau contenu :
from .data import generate_ou_signal, build_dataset, split_data, r2_score
from .models.efm_gate import EfmLSTM, EfmLSTMPredictor

# -----------------------------
# Définition de ce qui sera accessible via 'from efmgate import *'
# -----------------------------
__all__ = [
    'generate_ou_signal', 
    'build_dataset', 
    'split_data', 
    'r2_score',
    'EfmLSTM', 
    'EfmLSTMPredictor'
]
