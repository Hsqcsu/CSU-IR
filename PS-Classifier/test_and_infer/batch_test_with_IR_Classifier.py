import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# If there is a red underline below, don't worry, it will not affect the code running
from model.IR_encoder import IRModel
from model.Classifier_model import classifymodel
from test_and_infer.infer_IR_Classifier import infer_internal_batch

IR_model = IRModel()
Classifier_model = classifymodel(dim=1024,num_classes=2)


PRETRAIN_IR_MODEL_PATH = os.path.join(PROJECT_ROOT, "check_points", "best_ir_model_650-4000.pth")
PRETRAIN_IR_Classifier_MODEL_PATH = os.path.join(PROJECT_ROOT, "check_points", "best_IR_Classifier_model.pth")
TEST_IR_PATH = os.path.join(PROJECT_ROOT, "data", "test_data", "IR_Classifier", "test_ir.pt")
TEST_LABEL_PATH = os.path.join(PROJECT_ROOT, "data", "test_data", "IR_Classifier", "test_labels.pt")


IR_model.load_weights(PRETRAIN_IR_MODEL_PATH)
Classifier_model.load_weights(PRETRAIN_IR_Classifier_MODEL_PATH)

test_ir = torch.load(TEST_IR_PATH)
test_label = torch.load(TEST_LABEL_PATH)


class IRDataset(Dataset):
    def __init__(self, ir_spectra, labels):
        self.ir_spectra = ir_spectra
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ir_spectra[idx], self.labels[idx]

test_dataset = IRDataset(test_ir, test_label)
batch_size = 208
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IR_model.to(device)
Classifier_model.to(device)


predictions, labels = infer_internal_batch(IR_model, Classifier_model, test_loader, device)


label_0_pred_0 = sum(1 for pred, label in zip(predictions, labels) if label == 0 and pred == 0)
label_0_pred_1 = sum(1 for pred, label in zip(predictions, labels) if label == 0 and pred == 1)
label_1_pred_1 = sum(1 for pred, label in zip(predictions, labels) if label == 1 and pred == 1)
label_1_pred_0 = sum(1 for pred, label in zip(predictions, labels) if label == 1 and pred == 0)

print(f"Number of non-PS and predicted non-PS: {label_0_pred_0}")
print(f"Number of non-PS and predicted PS: {label_0_pred_1}")
print(f"Number of PS and predicted PS: {label_1_pred_1}")
print(f"Number of PS and predicted non-PS: {label_1_pred_0}")
