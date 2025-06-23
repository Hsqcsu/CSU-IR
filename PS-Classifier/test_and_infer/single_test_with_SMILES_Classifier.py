import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# If there is a red underline below, don't worry, it will not affect the code running
from model.SMILES_encoder import SmilesModel
from model.Classifier_model import classifymodel
from test_and_infer.infer_SMILES_Classifier import normalize_smiles

TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "model", "tokenizer-smiles-roberta-1e_new")
Smiles_Model = SmilesModel(roberta_model_path=None,
    roberta_tokenizer_path= TOKENIZER_PATH,
    smiles_maxlen=300,
    max_position_embeddings=505,
    vocab_size=181,
    feature_dim=768,
)
Classifier_model = classifymodel(dim=1024,num_classes=2)

PRETRAIN_SMILES_MODEL_PATH = os.path.join(PROJECT_ROOT, "check_points", "best_smiles_model_650-4000.pth")
PRETRAIN_SMILES_Classifier_MODEL_PATH = os.path.join(PROJECT_ROOT, "check_points", "best_SMILES_Classifier_model.pth")

Smiles_Model.load_weights(PRETRAIN_SMILES_MODEL_PATH )
Classifier_model.load_weights(PRETRAIN_SMILES_Classifier_MODEL_PATH)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Smiles_Model.to(device)
Classifier_model.to(device)

def predict_smiles(smiles):
    smiles = normalize_smiles(smiles)
    tokenizer = Smiles_Model.smiles_tokenizer
    encoded_smiles = tokenizer.encode_plus(
        text=smiles,
        max_length=300,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoded_smiles['input_ids'].to(device)
    attention_mask = encoded_smiles['attention_mask'].to(device)
    lengths = attention_mask.sum(dim=1)
    smiles_features = Smiles_Model.encode((input_ids, attention_mask), lengths)
    logits = Classifier_model (smiles_features)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    return "psychotropic substances" if pred == 1 else "non-psychotic substance"

smiles = 'CO[Si](CCCS)(OC)OC  '
print(predict_smiles(smiles))
'''
CO[Si](CCCS)(OC)OC, whose name is (3-Mercaptopropyl)trimethoxysilane, is an organosilicon compound commonly used for industrial purposes.
'''


smiles = 'NC(=O)c1cccc(-c2cccc(OC(=O)NC3CCCCC3)c2)c1'
print(predict_smiles(smiles))
'''
NC(=O)c1cccc(-c2cccc(OC(=O)NC3CCCCC3)c2)c1, whose name is N-desethyletonitazene, was added to the DEA's infrared drug library in 2024.
'''