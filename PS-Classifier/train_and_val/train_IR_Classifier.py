'''
Due to copyright issues, our training data is not open to the public for the time being, but users can perform training according to the following code, which is consistent with our training logic.
'''

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import os
from tqdm import tqdm
import json
from torch.cuda.amp import autocast, GradScaler
import torch
from torch.utils.data import Dataset, DataLoader

# If there is a red underline below, don't worry, it will not affect the code running
from model.IR_encoder import IRModel
from model.Classifier_model import classifymodel
from test_and_infer.infer_IR_Classifier import normalize_smiles


IR_model = IRModel()
for param in IR_model.parameters():
    param.requires_grad = False


IR_Classifier_model = classifymodel(dim=1024,num_classes=2)


PRETRAIN_IR_MODEL_PATH = os.path.join(PROJECT_ROOT, "check_points", "best_ir_model_650-4000.pth")
IR_model.load_weights(PRETRAIN_IR_MODEL_PATH)

def load_data(ir_path, ir_label_path):
    ir_data = torch.load(ir_path)
    ir_label_data = torch.load(ir_label_path)
    return ir_data, ir_label_data

class IRDataset(Dataset):
    def __init__(self, ir_spectra, labels):
        self.ir_spectra = ir_spectra
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ir_spectra[idx], self.labels[idx]

'''
Prepare the ir and label data files (saved in pt format), and use the two functions above to prepare the Dataloader.

train_ir, train_labels = load_data(train_ir_path, train_ir_label_path)
val_ir, val_labels = load_data(val_ir_path, val_ir_label_path)
test_ir, test_labels = load_data(test_ir_path, test_ir_label_path)

train_dataset = IRDataset(train_ir, train_labels)
val_dataset = IRDataset(val_ir, val_labels)
test_dataset = IRDataset(test_ir, test_labels)

batch_size = 208
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
'''


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IR_model.to(device)
IR_Classifier_model.to(device)


criterion = torch.nn.CrossEntropyLoss()
criterion.to(device)

optimizer = torch.optim.AdamW(IR_Classifier_model.parameters(), lr=0.00005, weight_decay=0.0001)
scaler = GradScaler()

def train_model(ir_model, mlp_model,train_loader, val_loader, optimizer, num_epochs=100, device='cuda'):
    ir_model.to(device)
    mlp_model.to(device)
    best_losses = []
    model_save_paths_ir = []

    training_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        ir_model.train()
        mlp_model.train()
        running_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for ir_spectra_batch,ir_label_batch in train_loader_tqdm:
            ir_spectra_tensor = ir_spectra_batch.to(device)
            ir_spectra_tensor = ir_spectra_tensor[:, 150:]
            ir_label = ir_label_batch.to(device)



            optimizer.zero_grad()

            with autocast():
                ir_features = ir_model(ir_spectra_tensor)
                logits = mlp_model(ir_features)
                loss = criterion(logits, ir_label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * ir_spectra_tensor.size(0)

            train_loader_tqdm.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        training_losses.append(epoch_loss)

        val_loss = validate_model(ir_model, mlp_model, val_loader, device)
        validation_losses.append(val_loss)

        print(f'Epoch {epoch}/{num_epochs - 1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

        if len(best_losses) < 5:
            model_save_path_ir = f'best_mlp_model_epoch_{epoch + 1}_ratio_{val_loss:.4f}.pth'
            torch.save( mlp_model.state_dict(), model_save_path_ir)

            best_losses.append(val_loss)
            model_save_paths_ir.append(model_save_path_ir)
        elif len(best_losses) == 5:
            if val_loss < max(best_losses):
                worst_index = best_losses.index(max(best_losses))
                os.remove(model_save_paths_ir[worst_index])
                best_losses.pop(worst_index)
                model_save_paths_ir.pop(worst_index)


                model_save_path_ir = f'best_ mlp_model_epoch_{epoch + 1}_ratio_{val_loss:.4f}.pth'
                torch.save(mlp_model.state_dict(), model_save_path_ir)

                best_losses.append(val_loss)
                model_save_paths_ir.append(model_save_path_ir)

        if epoch == num_epochs - 1:
            final_model_save_path = f'final_mlp_model_epoch_{epoch + 1}.pth'
            torch.save(mlp_model.state_dict(), final_model_save_path)

    loss_data = {
        'training_losses': training_losses,
        'validation_losses': validation_losses
    }
    with open('loss_data.json', 'w') as f:
        json.dump(loss_data, f)

def validate_model(ir_model, mlp_model, val_loader, device='cuda'):
    ir_model.eval()
    mlp_model.eval()
    running_loss = 0.0

    val_loader_tqdm = tqdm(val_loader, desc="Validating", unit="batch")
    with torch.no_grad():
        for ir_spectra_batch, ir_label_batch in val_loader_tqdm:
            ir_spectra_tensor = ir_spectra_batch.to(device)
            ir_spectra_tensor = ir_spectra_tensor[:, 150:]
            ir_label = ir_label_batch.to(device)

            with autocast():
                ir_features = ir_model(ir_spectra_tensor)
                logits = mlp_model(ir_features)
                loss = criterion(logits, ir_label)

            running_loss += loss.item() * ir_spectra_tensor.size(0)

            # 更新 tqdm 描述
            val_loader_tqdm.set_postfix(loss=loss.item())

    val_loss = running_loss / len(val_loader.dataset)
    return val_loss

train_model(IR_model,IR_Classifier_model, train_loader, val_loader, optimizer, num_epochs=100, device=device)



