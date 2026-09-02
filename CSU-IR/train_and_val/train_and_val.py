# This is a training script for Stage II (DFT) with Early Stopping based on Validation Recall@1
import sys
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Setup project root path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from model.IR_encoder import IRModel
from model.SMILES_encoder import SmilesModel


# ==============================================================================
# 1. Helper Functions & Dataset Definition
# ==============================================================================
def load_smiles_ir(smiles_path, ir_path):
    with open(smiles_path, 'r', encoding='utf-8') as f:
        smiles = f.read().splitlines()
    ir = torch.load(ir_path, map_location='cpu')
    return smiles, ir


def get_lr_multiplier(epoch, warmup_epochs):
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    return 1.0


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class IRSmilesDataset(Dataset):
    def __init__(self, ir_spectra, smiles):
        self.ir_spectra = ir_spectra
        self.smiles = smiles

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.ir_spectra[idx], self.smiles[idx]


# ==============================================================================
# 2. Fast Validation Logic (Recall@1 via Vectorized Top-1 Accuracy)
# ==============================================================================
def validate_model(smiles_model, ir_model, val_loader, device):
    smiles_model.eval()
    ir_model.eval()
    running_loss = 0.0
    result_smiles_features = []
    result_ir_features = []

    with torch.no_grad():
        for ir_spectra_batch, smiles_batch in tqdm(val_loader, desc="Validating", unit="batch", leave=False):
            ir_spectra_tensor = ir_spectra_batch.to(device)

            tokenizer = smiles_model.smiles_tokenizer
            encoded_smiles = [
                tokenizer.encode_plus(
                    text=s,
                    max_length=smiles_model.smiles_maxlen,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                ) for s in smiles_batch
            ]
            input_ids = torch.cat([item['input_ids'] for item in encoded_smiles], dim=0).to(device)
            attention_mask = torch.cat([item['attention_mask'] for item in encoded_smiles], dim=0).to(device)
            lengths = attention_mask.sum(dim=1)

            with autocast():
                smiles_features = smiles_model.encode((input_ids, attention_mask), lengths)
                ir_features = ir_model(ir_spectra_tensor)
                result_smiles_features.append(smiles_features)
                result_ir_features.append(ir_features)

                t = torch.exp(smiles_model.t_prime)
                b = smiles_model.bias
                logits = torch.matmul(ir_features, smiles_features.T) * t + b
                n = logits.size(0)
                labels = 2 * torch.eye(n).to(device) - torch.ones(n, n).to(device)
                loss = -torch.sum(F.logsigmoid(labels * logits)) / n

            running_loss += loss.item() * ir_spectra_tensor.size(0)

    # Vectorized Top-1 Accuracy (Recall@1) Calculation
    all_smiles_features = torch.cat(result_smiles_features, dim=0)
    all_ir_features = torch.cat(result_ir_features, dim=0)
    
    logits_full = torch.matmul(all_ir_features, all_smiles_features.T)
    top1_indices = torch.argmax(logits_full, dim=1)
    
    correct_matches = (top1_indices == torch.arange(len(top1_indices), device=device)).sum().item()
    total_samples = len(top1_indices)
    top_1_ratio = correct_matches / total_samples if total_samples > 0 else 0

    val_loss = running_loss / len(val_loader.dataset)
    return val_loss, top_1_ratio


# ==============================================================================
# 3. Training Loop with Early Stopping Supervised by Val Recall@1
# ==============================================================================
def train_model(smiles_model, ir_model, train_loader, val_loader, optimizer, 
                num_epochs=80, warmup_epochs=10, patience=15, output_dir=None, device='cuda'):
    smiles_model.to(device)
    ir_model.to(device)
    scaler = GradScaler()

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "check_points", "Multi-stage_training_Stage_II_DFT", "DFT_without_MD_pretraining")
    os.makedirs(output_dir, exist_ok=True)

    best_val_ratio = -1.0
    early_stop_counter = 0

    best_smiles_path = os.path.join(output_dir, 'best_smiles_model.pth')
    best_ir_path = os.path.join(output_dir, 'best_ir_model.pth')

    scheduler_warmup = LambdaLR(optimizer, lr_lambda=lambda epoch: get_lr_multiplier(epoch, warmup_epochs))
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=(num_epochs - warmup_epochs))

    training_losses = []
    validation_losses = []
    validation_recalls = []

    print(f"\nStarting training with Early Stopping: Max Epochs={num_epochs}, Patience={patience}, Warmup={warmup_epochs}")

    for epoch in range(num_epochs):
        smiles_model.train()
        ir_model.train()
        running_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

        for ir_spectra_batch, smiles_batch in train_loader_tqdm:
            ir_spectra_tensor = ir_spectra_batch.to(device)

            tokenizer = smiles_model.smiles_tokenizer
            encoded_smiles = [
                tokenizer.encode_plus(
                    text=s,
                    max_length=smiles_model.smiles_maxlen,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                ) for s in smiles_batch
            ]
            input_ids = torch.cat([item['input_ids'] for item in encoded_smiles], dim=0).to(device)
            attention_mask = torch.cat([item['attention_mask'] for item in encoded_smiles], dim=0).to(device)
            lengths = attention_mask.sum(dim=1)

            optimizer.zero_grad()

            with autocast():
                smiles_features = smiles_model.encode((input_ids, attention_mask), lengths)
                ir_features = ir_model(ir_spectra_tensor)

                t = torch.exp(smiles_model.t_prime)
                b = smiles_model.bias
                logits = torch.matmul(ir_features, smiles_features.T) * t + b

                n = logits.size(0)
                labels = 2 * torch.eye(n, device=device) - torch.ones(n, n, device=device)
                loss = -torch.sum(F.logsigmoid(labels * logits)) / n

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * ir_spectra_tensor.size(0)
            train_loader_tqdm.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        training_losses.append(epoch_loss)

        # Validation Phase
        val_loss, top_1_ratio = validate_model(smiles_model, ir_model, val_loader, device)
        validation_losses.append(val_loss)
        validation_recalls.append(top_1_ratio)

        print(f"Epoch {epoch + 1}/{num_epochs} -> Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Recall@1: {top_1_ratio:.4f}")

        # Update Learning Rate Schedule
        (scheduler_warmup if epoch < warmup_epochs else scheduler_cosine).step()

        # Check Early Stopping & Save Best Weights
        if top_1_ratio > best_val_ratio:
            best_val_ratio = top_1_ratio
            early_stop_counter = 0
            torch.save(smiles_model.state_dict(), best_smiles_path)
            torch.save(ir_model.state_dict(), best_ir_path)
            print(f"  ✨ [New Best] Recall@1 improved to {best_val_ratio:.4f}. Best weights saved!")
        else:
            early_stop_counter += 1
            print(f"  ⚠️ [No Improvement] Early stopping counter: {early_stop_counter}/{patience}")

        # Save training history JSON per epoch
        history_data = {
            "training_losses": training_losses,
            "validation_losses": validation_losses,
            "validation_recalls": validation_recalls
        }
        with open(os.path.join(output_dir, 'history.json'), 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=4)

        # Early Stopping Termination
        if early_stop_counter >= patience:
            print(f"\n🛑 [Early Stopping Triggered] Val Recall@1 has not improved for {patience} consecutive epochs. Terminating training.")
            break

    print('\n================ Training Summary ================')
    print(f'Final Best Validation Recall@1: {best_val_ratio:.4f}')
    print(f'Best models saved to:\n  - {best_smiles_path}\n  - {best_ir_path}')


# ==============================================================================
# 4. Main Execution Entry Point
# ==============================================================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")

    TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "model", "tokenizer-smiles-roberta-1e_new")

    # Initialize Models
    ir_model = IRModel().to(device)
    smiles_model = SmilesModel(
        roberta_model_path=None,
        roberta_tokenizer_path=TOKENIZER_PATH,
        smiles_maxlen=300,
        max_position_embeddings=505,
        vocab_size=181,
        feature_dim=768
    ).to(device)

    print(f"SmilesModel Parameters: {count_parameters(smiles_model)}")
    print(f"IR_model Parameters   : {count_parameters(ir_model)}")

    # Check for Stage I (MD) Pre-trained Weights
    stage_1_dir = os.path.join(PROJECT_ROOT, "check_points", "Multi-stage_training_Stage_I_MD")
    s1_smiles_ckpt = os.path.join(stage_1_dir, "best_smiles_model.pth")
    s1_ir_ckpt = os.path.join(stage_1_dir, "best_ir_model.pth")

    if os.path.exists(s1_smiles_ckpt) and os.path.exists(s1_ir_ckpt):
        smiles_model.load_state_dict(torch.load(s1_smiles_ckpt, map_location=device))
        ir_model.load_state_dict(torch.load(s1_ir_ckpt, map_location=device))
        output_dir = os.path.join(PROJECT_ROOT, "check_points", "Multi-stage_training_Stage_II_DFT", "MD_DFT")
        print(f"--> Found Stage I (MD) weights! Inheriting weights from: {stage_1_dir}")
    else:
        output_dir = os.path.join(PROJECT_ROOT, "check_points", "Multi-stage_training_Stage_II_DFT", "DFT_without_MD_pretraining")
        print("--> No Stage I weights found. Training Stage II (DFT) from scratch.")

    # Load Dataset
    data_dir = os.path.join(PROJECT_ROOT, "data", "Multi-staged_training_data", "Density functional simulation data")
    train_smiles_path = os.path.join(data_dir, "train_smiles.txt")
    train_ir_path = os.path.join(data_dir, "train_ir.pt")
    val_smiles_path = os.path.join(data_dir, "val_smiles.txt")
    val_ir_path = os.path.join(data_dir, "val_ir.pt")

    smiles_train, ir_train = load_smiles_ir(train_smiles_path, train_ir_path)
    smiles_val, ir_val = load_smiles_ir(val_smiles_path, val_ir_path)

    train_dataset = IRSmilesDataset(ir_train, smiles_train)
    val_dataset = IRSmilesDataset(ir_val, smiles_val)

    batch_size = 208
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Number of training samples  : {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    # Optimizer & Training Configuration
    optimizer = AdamW(list(smiles_model.parameters()) + list(ir_model.parameters()), lr=5e-05, weight_decay=0.0001)
    
    train_model(
        smiles_model=smiles_model,
        ir_model=ir_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=200,
        warmup_epochs=10,
        patience=15,
        output_dir=output_dir,
        device=device
    )
