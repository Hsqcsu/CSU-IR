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
from model.SMILES_encoder import SmilesModel
from model.Classifier_model import classifymodel
from test_and_infer.infer_SMILES_Classifier import normalize_smiles


Smiles_Model = SmilesModel(roberta_model_path=None,
    roberta_tokenizer_path= "D:\\Spectrum\\models\\models\\tokenizer-smiles-roberta-1e_new",
    smiles_maxlen=300,
    max_position_embeddings=505,
    vocab_size=181,
    feature_dim=768,
)
for param in Smiles_Model.parameters():
    param.requires_grad = False


SMILES_Classifier_model = classifymodel(dim=1024,num_classes=2)


PRETRAIN_SMILES_MODEL_PATH = os.path.join(PROJECT_ROOT, "check_points", "best_smiles_model_650-4000.pth")
Smiles_Model.load_weights(PRETRAIN_SMILES_MODEL_PATH)


class SmilesDataset(Dataset):
    def __init__(self, smiles_list, labels_tensor):
        self.smiles = smiles_list
        self.labels = labels_tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.smiles[idx], self.labels[idx]

def load_smiles_labels(smiles_path, labels_path):
    with open(smiles_path, 'r', encoding='utf-8') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    labels_tensor = torch.load(labels_path)
    return smiles_list, labels_tensor

train_smiles_path = r'D:\Spectrum\github\PS-Classifier\data\pretrain_data\SMILES_Classifier\train_smiles.txt'
train_labels_path = r'D:\Spectrum\github\PS-Classifier\data\pretrain_data\SMILES_Classifier\train_labels.pt'
val_smiles_path = r'D:\Spectrum\github\PS-Classifier\data\pretrain_data\SMILES_Classifier\val_smiles.txt'
val_labels_path = r'D:\Spectrum\github\PS-Classifier\data\pretrain_data\SMILES_Classifier\val_labels.pt'
test_smiles_path = r'D:\Spectrum\github\PS-Classifier\data\pretrain_data\SMILES_Classifier\test_smiles.txt'
test_labels_path = r'D:\Spectrum\github\PS-Classifier\data\pretrain_data\SMILES_Classifier\test_labels.pt'


combined_train_smiles, combined_train_labels = load_smiles_labels(train_smiles_path, train_labels_path)
combined_val_smiles, combined_val_labels = load_smiles_labels(val_smiles_path, val_labels_path)
combined_test_smiles, combined_test_labels = load_smiles_labels(test_smiles_path, test_labels_path)


train_dataset = SmilesDataset(combined_train_smiles, combined_train_labels)
val_dataset = SmilesDataset(combined_val_smiles, combined_val_labels)
test_dataset = SmilesDataset(combined_test_smiles, combined_test_labels)


batch_size = 208
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Smiles_Model.to(device)
SMILES_Classifier_model.to(device)

criterion = torch.nn.CrossEntropyLoss()
criterion.to(device)


optimizer = torch.optim.AdamW(SMILES_Classifier_model.parameters(), lr=0.00005, weight_decay=0.0001)
scaler = GradScaler()

def train_model(smiles_model, mlp_model,train_loader, val_loader, optimizer, num_epochs=100, device='cuda'):
    smiles_model.to(device)
    mlp_model.to(device)
    best_losses = []
    model_save_paths_ir = []


    training_losses = []
    validation_losses = []


    for epoch in range(num_epochs):
        smiles_model.train()
        mlp_model.train()
        running_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for smiles_batch,ir_label_batch in train_loader_tqdm:
            ir_label = ir_label_batch.to(device)

            tokenizer = smiles_model.smiles_tokenizer
            encoded_smiles = [tokenizer.encode_plus(
                text=smiles,
                max_length=300,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ) for smiles in smiles_batch]

            input_ids = torch.cat([item['input_ids'] for item in encoded_smiles], dim=0).to(device)
            attention_mask = torch.cat([item['attention_mask'] for item in encoded_smiles], dim=0).to(device)
            lengths = attention_mask.sum(dim=1)


            optimizer.zero_grad()

            with autocast():
                smiles_features = smiles_model.encode((input_ids, attention_mask), lengths)
                logits = mlp_model(smiles_features)

                loss = criterion(logits, ir_label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * ir_label.size(0)

            train_loader_tqdm.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        training_losses.append(epoch_loss)

        val_loss = validate_model(smiles_model, mlp_model, val_loader, device)
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


def validate_model(smiles_model, mlp_model, val_loader, device='cuda'):
    smiles_model.eval()
    mlp_model.eval()
    running_loss = 0.0
    val_loader_tqdm = tqdm(val_loader, desc="Validating", unit="batch")

    with torch.no_grad():
        for smiles_batch,ir_label_batch in val_loader_tqdm:
            ir_label = ir_label_batch.to(device)

            tokenizer = smiles_model.smiles_tokenizer
            encoded_smiles = [tokenizer.encode_plus(
                text=smiles,
                max_length=300,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ) for smiles in smiles_batch]

            input_ids = torch.cat([item['input_ids'] for item in encoded_smiles], dim=0).to(device)
            attention_mask = torch.cat([item['attention_mask'] for item in encoded_smiles], dim=0).to(device)
            lengths = attention_mask.sum(dim=1)

            with autocast():
                smiles_features = smiles_model.encode((input_ids, attention_mask), lengths)
                logits = mlp_model(smiles_features)
                loss = criterion(logits, ir_label)

            running_loss += loss.item() * ir_label.size(0)
            val_loader_tqdm.set_postfix(loss=loss.item())

    val_loss = running_loss / len(val_loader.dataset)
    return val_loss

# 训练模型
train_model(Smiles_Model,SMILES_Classifier_model, train_loader, val_loader, optimizer, num_epochs=100, device=device)



