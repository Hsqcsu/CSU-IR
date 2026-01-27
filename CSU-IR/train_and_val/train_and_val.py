import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# If there is a red underline below, don't worry, it will not affect the code running
from model.IR_encoder import IRModel
from model.SMILES_encoder import SmilesModel
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR 
from torch.cuda.amp import autocast, GradScaler
import json



def load_smiles_ir(smiles_path, ir_path):
    with open(smiles_path, 'r', encoding='utf-8') as f:
        smiles = f.read().splitlines()
    ir = torch.load(ir_path)
    return smiles, ir


def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    return 1.0

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "model", "tokenizer-smiles-roberta-1e_new")

IR_model = IRModel()
SmilesModel = SmilesModel(roberta_model_path=None,
    roberta_tokenizer_path= TOKENIZER_PATH,
    smiles_maxlen=300,
    max_position_embeddings=505,
    vocab_size=181,
    feature_dim=768,
)

# You need to download the training data in huggingface and put it in the corresponding folder. Here we use the DFT data as an example.
train_smiles_path = 'QM9S_DFT_train_smiles.txt'
train_ir_path = 'QM9S_DFT_train_ir.pt'
val_smiles_path = 'QM9S_DFT_val_smiles.txt'
val_ir_path = 'QM9S_DFT_val_ir.pt'
test_smiles_path = 'QM9S_DFT_test_smiles.txt'
test_ir_path = 'QM9S_DFT_test_ir.pt'

smiles_train, ir_train = load_smiles_ir(train_smiles_path, train_ir_path)
smiles_val, ir_val = load_smiles_ir(val_smiles_path, val_ir_path)
smiles_test, ir_test = load_smiles_ir(test_smiles_path, test_ir_path)

class IRSmilesDataset(Dataset):
    def __init__(self, ir_spectra, smiles):
        self.ir_spectra = ir_spectra
        self.smiles = smiles

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.ir_spectra[idx], self.smiles[idx]


train_dataset = IRSmilesDataset(ir_train, smiles_train)
val_dataset = IRSmilesDataset(ir_val, smiles_val)
test_dataset = IRSmilesDataset(ir_test, smiles_test)

# DataLoader
batch_size = 208
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Number of training sets: {len(train_dataset)}")
print(f"Number of validation sets: {len(val_dataset)}")
print(f"Number of test sets: {len(test_dataset)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SmilesModel.to(device)
IR_model.to(device)
print(f"SmilesModel Parameter: {count_parameters(SmilesModel)}")
print(f"IR_model Parameter: {count_parameters(IR_model)}")

optimizer = AdamW(list(SmilesModel.parameters()) + list(IR_model.parameters()), lr=5e-05, weight_decay=0.0001)

scheduler_warmup = LambdaLR(optimizer, lr_lambda=lr_lambda)
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=(num_epochs - warmup_epochs))
scaler = GradScaler()

def train_model(smiles_model, ir_model, train_loader, val_loader, optimizer, num_epochs=80, device='cuda'):
    smiles_model.to(device)
    ir_model.to(device)
    best_ratios = []
    best_epochs = []
    model_save_paths_smiles = []
    model_save_paths_ir = []


    training_losses = []
    validation_losses = []

    for epoch in range(0,num_epochs):
        smiles_model.train()
        ir_model.train()
        running_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for ir_spectra_batch,smiles_batch in train_loader_tqdm:
            ir_spectra_tensor = ir_spectra_batch.to(device)

            current_smiles_batch = []
            for smiles in smiles_batch:
                current_smiles_aug = augment_smiles(smiles,set(current_smiles_batch),sme)
                current_smiles_batch.append(current_smiles_aug)

            tokenizer = smiles_model.smiles_tokenizer
            encoded_smiles = [tokenizer.encode_plus(
                text=smiles,
                max_length=300,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ) for smiles in current_smiles_batch]

            input_ids = torch.cat([item['input_ids'] for item in encoded_smiles], dim=0).to(device)
            attention_mask = torch.cat([item['attention_mask'] for item in encoded_smiles], dim=0).to(device)
            lengths = attention_mask.sum(dim=1)

            optimizer.zero_grad()

            with autocast():
                smiles_features = smiles_model.encode((input_ids, attention_mask),lengths)
                ir_features = ir_model(ir_spectra_tensor)

                t = torch.exp(smiles_model.t_prime)
                b = smiles_model.bias
                logits = torch.matmul(ir_features, smiles_features.T) * t + b

                n = logits.size(0)
                labels = 2 * torch.eye(n).to(device) - torch.ones(n, n).to(device)

                loss = -torch.sum(F.logsigmoid(labels * logits)) / n

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * ir_spectra_tensor.size(0)

            train_loader_tqdm.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        training_losses.append(epoch_loss)

        val_loss, top_1_ratio = validate_model(smiles_model, ir_model, val_loader, device)
        validation_losses.append(val_loss)

        print(f'Epoch {epoch}/{num_epochs - 1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f},top_1_ratio:{top_1_ratio:.4f}')


        if epoch < 10:
            scheduler_warmup.step()
        else:
            scheduler_cosine.step()

        # Save top 5 best models
        if len(best_ratios) < 5:
            model_save_path_smiles = f'best_smiles_model_epoch_{epoch + 1}_ratio_{top_1_ratio:.4f}.pth'
            model_save_path_ir = f'best_ir_model_epoch_{epoch + 1}_ratio_{top_1_ratio:.4f}.pth'
            torch.save(smiles_model.state_dict(), model_save_path_smiles)
            torch.save(ir_model.state_dict(), model_save_path_ir)

            best_ratios.append(top_1_ratio)
            best_epochs.append(epoch)
            model_save_paths_smiles.append(model_save_path_smiles)
            model_save_paths_ir.append(model_save_path_ir)
        elif len(best_ratios) == 5:
            if top_1_ratio > min(best_ratios):
                worst_index = best_ratios.index(min(best_ratios))
                os.remove(model_save_paths_ir[worst_index])
                os.remove(model_save_paths_smiles[worst_index])
                best_ratios.pop(worst_index)
                best_epochs.pop(worst_index)
                model_save_paths_ir.pop(worst_index)
                model_save_paths_smiles.pop(worst_index)

                model_save_path_smiles = f'best_smiles_model_epoch_{epoch + 1}_ratio_{top_1_ratio:.4f}.pth'
                model_save_path_ir = f'best_ir_model_epoch_{epoch + 1}_ratio_{top_1_ratio:.4f}.pth'
                torch.save(smiles_model.state_dict(), model_save_path_smiles)
                torch.save(ir_model.state_dict(), model_save_path_ir)

                best_ratios.append(top_1_ratio)
                best_epochs.append(epoch)
                model_save_paths_smiles.append(model_save_path_smiles)
                model_save_paths_ir.append(model_save_path_ir)

    print('Training complete. Best validation ratios: ', best_ratios)
    print('Best epochs: ', best_epochs)

    loss_data = {
        'training_losses': training_losses,
        'validation_losses': validation_losses
    }
    with open('loss_data.json', 'w') as f:
        json.dump(loss_data, f)

def validate_model(smiles_model, ir_model, val_loader, device='cuda'):
    smiles_model.eval()
    ir_model.eval()
    running_loss = 0.0
    result_smiles = []
    result_ir = []

    val_loader_tqdm = tqdm(val_loader, desc="Validating", unit="batch")

    with torch.no_grad():
        for ir_spectra_batch,smiles_batch in val_loader_tqdm:
            ir_spectra_tensor = ir_spectra_batch.to(device)

            current_smiles_batch = []
            for smiles in smiles_batch:
                current_smiles_aug = augment_smiles(smiles, set(current_smiles_batch), sme)
                current_smiles_batch.append(current_smiles_aug)

            tokenizer = smiles_model.smiles_tokenizer
            encoded_smiles = [tokenizer.encode_plus(
                text=smiles,
                max_length=300,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ) for smiles in current_smiles_batch]

            input_ids = torch.cat([item['input_ids'] for item in encoded_smiles], dim=0).to(device)
            attention_mask = torch.cat([item['attention_mask'] for item in encoded_smiles], dim=0).to(device)
            lengths = attention_mask.sum(dim=1)

            with autocast():
                smiles_features = smiles_model.encode((input_ids, attention_mask),lengths)
                ir_features = ir_model(ir_spectra_tensor)


                result_smiles.append(smiles_features)
                result_ir.append(ir_features)

                t = torch.exp(smiles_model.t_prime)
                b = smiles_model.bias
                logits = torch.matmul(ir_features, smiles_features.T) * t + b

                n = logits.size(0)
                labels = 2 * torch.eye(n).to(device) - torch.ones(n, n).to(device)

                loss = -torch.sum(F.logsigmoid(labels * logits)) / n

            running_loss += loss.item() * ir_spectra_tensor.size(0)

            val_loader_tqdm.set_postfix(loss=loss.item())

        result_smiles = torch.cat(result_smiles, 0)
        result_ir = torch.cat(result_ir, 0)

        correct_matches = 0
        total_samples = result_smiles.size(0)
        batch_size = 100 

        for i in range(0, total_samples, batch_size):
            end = min(i + batch_size, total_samples)
            logits = torch.matmul(result_ir[i:end], result_smiles.T)

            for j in range(end - i):
                if logits[j, i + j] == logits[j].max():
                    correct_matches += 1

        top_1_ratio = correct_matches / total_samples if total_samples > 0 else 0

    val_loss = running_loss / len(val_loader.dataset)
    return val_loss, top_1_ratio

train_model(SmilesModel, IR_model, train_loader, val_loader, optimizer, num_epochs=80, device=device)
