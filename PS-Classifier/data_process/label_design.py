import sys
import os
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
print(PROJECT_ROOT)

All_PS_smiles_PATH = os.path.join(PROJECT_ROOT, "data", "PS_smiles", "all_PS_smiles.txt")

# If there is a red underline below, don't worry, it will not affect the code running
from test_and_infer.infer_SMILES_Classifier import normalize_smiles

def extract_labels(smiles_list, all_drugs_smiles):
    labels = []
    for smiles in smiles_list:
        normalized_smiles = normalize_smiles(smiles)
        if normalized_smiles not in all_drugs_smiles:
            labels.append(0)
        else:
            labels.append(1)
    return torch.tensor(labels)