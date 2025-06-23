import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# If there is a red underline below, don't worry, it will not affect the code running
from model.SMILES_encoder import SmilesModel
from model.Classifier_model import classifymodel
from test_and_infer.infer_SMILES_Classifier import infer_internal_batch

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
TEST_SMILES_PATH = os.path.join(PROJECT_ROOT, "data", "test_data", "SMILES_Classifier", "test_smiles.txt")
TEST_LABEL_PATH = os.path.join(PROJECT_ROOT, "data", "test_data", "SMILES_Classifier", "test_labels.pt")


Smiles_Model.load_weights(PRETRAIN_SMILES_MODEL_PATH )
Classifier_model.load_weights(PRETRAIN_SMILES_Classifier_MODEL_PATH)

with open(TEST_SMILES_PATH , 'r', encoding='utf-8') as f:
    test_smiles_list = [line.strip() for line in f if line.strip()]
test_label = torch.load(TEST_LABEL_PATH)


class SmilesDataset(Dataset):
    def __init__(self, smiles_list, labels_tensor):
        self.smiles = smiles_list
        self.labels = labels_tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.smiles[idx], self.labels[idx]

test_dataset = SmilesDataset(test_smiles_list, test_label)
batch_size = 208
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Smiles_Model.to(device)
Classifier_model.to(device)


predictions, labels = infer_internal_batch(Smiles_Model, Classifier_model, test_loader, device)


label_0_pred_0 = sum(1 for pred, label in zip(predictions, labels) if label == 0 and pred == 0)
label_0_pred_1 = sum(1 for pred, label in zip(predictions, labels) if label == 0 and pred == 1)
label_1_pred_1 = sum(1 for pred, label in zip(predictions, labels) if label == 1 and pred == 1)
label_1_pred_0 = sum(1 for pred, label in zip(predictions, labels) if label == 1 and pred == 0)

print(f"Number of non-PS and predicted non-PS: {label_0_pred_0}")
print(f"Number of non-PS and predicted PS: {label_0_pred_1}")
print(f"Number of PS and predicted PS: {label_1_pred_1}")
print(f"Number of PS and predicted non-PS: {label_1_pred_0}")
