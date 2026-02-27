# This is an example file using NPS retrieval against the existing library. 
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
print(PROJECT_ROOT)

import torch
import pandas as pd
import numpy as np
import jcamp
from tqdm import tqdm
import torch.nn.functional as F
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader

# If there is a red underline below, don't worry, it will not affect the code running
from model.IR_encoder import IRModel
from model.SMILES_encoder import SmilesModel

from test_and_infer.test_and_infer_functions import ModelInference
from test_and_infer.test_and_infer_functions import get_feature_from_smiles
from test_and_infer.test_and_infer_functions import get_topK_result
from test_and_infer.test_and_infer_functions import normalize_smiles

TOKENIZER_PATH = os.path.join(PROJECT_ROOT, 'model', "tokenizer-smiles-roberta-1e_new")


class IRSmilesDataset(Dataset):
    def __init__(self, ir_spectra, smiles):
        self.ir_spectra = ir_spectra
        self.smiles = smiles

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.ir_spectra[idx], self.smiles[idx]


def load_data(smiles_path, ir_path):
    with open(smiles_path, 'r') as f:
        smiles_list = f.read().splitlines()
    ir_data = torch.load(ir_path)
    return smiles_list, ir_data


def load_smiles(smiles_path):
    with open(smiles_path, 'r') as f:
        smiles_list = f.read().splitlines()
    return smiles_list


'--------------------------------------------------------------------------------------------'

Interference_library_path = os.path.join(PROJECT_ROOT, 'data', 'processed_library', 'PS', 'smiles_Existing_PS.txt')
Interference_library = load_smiles(Interference_library_path)

NPS_smiles_path = os.path.join(PROJECT_ROOT, 'data', 'test_data', 'NPS', 'filtered_final_NPS_smiles.txt')
NPS_ir_path = os.path.join(PROJECT_ROOT, 'data', 'test_data', 'NPS', 'filtered_final_NPS_ir.pt')

NPS_smiles, NPS_ir = load_data(NPS_smiles_path, NPS_ir_path)

NPS_dataset = IRSmilesDataset(NPS_ir, NPS_smiles)
batch_size = 208
NPS_loader = DataLoader(NPS_dataset, batch_size=batch_size, shuffle=False)

'--------------------------------------------------------------------------------------------'
IR_model = IRModel()
SmilesModel = SmilesModel(roberta_model_path=None,
                          roberta_tokenizer_path=TOKENIZER_PATH,
                          smiles_maxlen=300,
                          max_position_embeddings=505,
                          vocab_size=181,
                          feature_dim=768,
                          )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SmilesModel.to(device)
IR_model.to(device)
'--------------------------------------------------------------------------------------------'

model_pairs = [
    (os.path.join(PROJECT_ROOT, 'check_points', 'Multi-stage_training_Stage_III_EXP',
                  'best_smiles_model_0.9230379746835443.pth'),
     os.path.join(PROJECT_ROOT, 'check_points', 'Multi-stage_training_Stage_III_EXP',
                  'best_ir_model_0.9230379746835443.pth')),
]


def evaluate_loader(loader, combined_features, smiles_list, loader_name):
    top_1_matches = 0
    top_5_matches = 0
    top_10_matches = 0
    total_samples = 0

    results = []

    normalized_library_smiles = [normalize_smiles(s) for s in smiles_list]

    test_loader_tqdm = tqdm(loader, unit="batch")
    for ir_spectra_batch, smiles_batch in test_loader_tqdm:
        ir_spectra_tensor = ir_spectra_batch.to(device)

        for idx, ir_spectra in enumerate(ir_spectra_tensor):
            query_smiles = normalize_smiles(smiles_batch[idx])
            ir_feature = ModelInferenc.ir_encode(ir_spectra)

            indices, scores = get_topK_result(ir_feature, combined_features, 10)
            top_indices = indices[0] if len(indices.shape) > 1 else indices

            found_at_rank = -1
            current_top_smiles = []

            for rank, i in enumerate(top_indices):
                i = int(i)
                if i < len(normalized_library_smiles):
                    candidate_smiles = normalized_library_smiles[i]
                    current_top_smiles.append(smiles_list[i])

                    if found_at_rank == -1 and query_smiles == candidate_smiles:
                        found_at_rank = rank

            if found_at_rank != -1:
                if found_at_rank < 1:
                    top_1_matches += 1
                if found_at_rank < 5:
                    top_5_matches += 1
                if found_at_rank < 10:
                    top_10_matches += 1

            results.append((current_top_smiles, scores.tolist()))
            total_samples += 1

    top_1_ratio = top_1_matches / total_samples if total_samples > 0 else 0
    top_5_ratio = top_5_matches / total_samples if total_samples > 0 else 0
    top_10_ratio = top_10_matches / total_samples if total_samples > 0 else 0

    print(f"\nResults for {loader_name}:")
    print(f"Total Samples: {total_samples}")
    print(f"Recall@1  : {top_1_ratio:.4f}")
    print(f"Recall@5  : {top_5_ratio:.4f}")
    print(f"Recall@10 : {top_10_ratio:.4f}")

    with open(f'{loader_name}_results.txt', 'w') as file:
        for top_smiles, scores in results:
            file.write(f'Top SMILES: {top_smiles}\n')
            file.write(f'Scores: {scores}\n\n')


for smiles_model_path, ir_model_path in model_pairs:
    print(f"Processing models: {smiles_model_path} and {ir_model_path}")

    ModelInferenc = ModelInference(
        SmilesModel,
        IR_model,
        pretrain_model_path_sm=smiles_model_path,
        pretrain_model_path_ir=ir_model_path,
        device=None
    )

    NPS_smiles_features = get_feature_from_smiles(NPS_smiles, ModelInferenc)
    Existing_PS_library = get_feature_from_smiles(Interference_library, ModelInferenc)
    library_Existing_PS = torch.cat((NPS_smiles_features, Existing_PS_library), dim=0)
    smiles_Existing_PS = list(NPS_smiles) + list(Interference_library)

    evaluate_loader(NPS_loader, library_Existing_PS, smiles_Existing_PS, 'NPS against Existing library')




