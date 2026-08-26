import sys
import os
import yaml
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Ensure project root is in system path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CSU-IR'))
if not os.path.exists(PROJECT_ROOT):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from model.IR_encoder import IRModel
from model.SMILES_encoder import SmilesModel


def load_smiles_and_spectra(smiles_path, ir_path, raman_path=None):
    """
    Loads SMILES strings and corresponding spectral tensors.
    If raman_path is provided, concatenates IR and Raman tensors along feature dimension.
    """
    with open(smiles_path, 'r', encoding='utf-8') as f:
        smiles = f.read().splitlines()
    
    ir_tensor = torch.load(ir_path, map_location='cpu')
    if ir_tensor.dim() == 3 and ir_tensor.size(1) == 1:
        ir_tensor = ir_tensor.squeeze(1)

    # Multi-modal concatenation: IR + Raman
    if raman_path and os.path.exists(raman_path):
        raman_tensor = torch.load(raman_path, map_location='cpu')
        if raman_tensor.dim() == 3 and raman_tensor.size(1) == 1:
            raman_tensor = raman_tensor.squeeze(1)
        
        assert len(ir_tensor) == len(raman_tensor), "Sample count mismatch between IR and Raman tensors!"
        combined_spectra = torch.cat([ir_tensor, raman_tensor], dim=-1)
        print(f"📊 Multi-modal data loaded: IR {tuple(ir_tensor.shape)} + Raman {tuple(raman_tensor.shape)} -> Combined {tuple(combined_spectra.shape)}")
        return smiles, combined_spectra

    print(f"📊 Single-modality data loaded: IR {tuple(ir_tensor.shape)}")
    return smiles, ir_tensor


def get_lr_multiplier(epoch, warmup_epochs):
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    return 1.0


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MultiModalDataset(Dataset):
    def __init__(self, spectra, smiles):
        self.spectra = spectra
        self.smiles = smiles

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.spectra[idx], self.smiles[idx]


# ==========================================
# SOTA Comparison Checkpoint Auto-Resolver
# ==========================================
def resolve_sota_checkpoints_and_paths(config, project_root):
    """
    Resolves pre-trained checkpoints and output directories specifically
    for SOTA Comparison tasks.
    """
    ckpt_root = os.path.join(project_root, "check_points", "SOTA_Comparison")
    
    dir_qm9s_ir = os.path.join(ckpt_root, "QM9S_ir_only")
    dir_eb_chonf = os.path.join(ckpt_root, "EB_CHONF_finetuning")
    dir_qm9s_ir_raman = os.path.join(ckpt_root, "QM9S_ir_raman")
    dir_sdbs_chonf = os.path.join(ckpt_root, "SDBS_CHONF_ir_raman_finetuning")

    def check_weights(folder):
        smiles_p = os.path.join(folder, "best_smiles_model.pth")
        ir_p = os.path.join(folder, "best_ir_model.pth")
        if os.path.exists(smiles_p) and os.path.exists(ir_p):
            return smiles_p, ir_p
        return None, None

    task = config.get("training_params", {}).get("task", "").strip().lower()
    load_smiles_ckpt, load_ir_ckpt = None, None
    target_output_dir = None

    print(f"\n{'='*25} SOTA Comparison Auto Routing: [TASK: {task.upper()}] {'='*25}")

    # Task 1: QM9S IR-only Pre-training
    if task in ["qm9s_ir_only", "task1"]:
        target_output_dir = dir_qm9s_ir
        print(f"--> [Task 1] QM9S IR-Only Pre-training. Training from scratch.")
        print(f"--> Output directory: {target_output_dir}")

    # Task 2: EB_CHONF Fine-tuning (Inherits ONLY from QM9S_ir_only)
    elif task in ["eb_chonf_finetuning", "eb_chonf", "task2"]:
        target_output_dir = dir_eb_chonf
        s_ckpt, i_ckpt = check_weights(dir_qm9s_ir)
        if s_ckpt and i_ckpt:
            load_smiles_ckpt, load_ir_ckpt = s_ckpt, i_ckpt
            print(f"--> [Task 2] Found QM9S_ir_only pre-trained weights! Loading from: {dir_qm9s_ir}")
        else:
            print(f"--> [Task 2] No QM9S_ir_only weights found. Fine-tuning from scratch.")
        print(f"--> Output directory: {target_output_dir}")

    # Task 3: QM9S IR+Raman Pre-training
    elif task in ["qm9s_ir_raman", "task3"]:
        target_output_dir = dir_qm9s_ir_raman
        print(f"--> [Task 3] QM9S IR+Raman Multi-modal Pre-training. Training from scratch.")
        print(f"--> Output directory: {target_output_dir}")

    # Task 4: SDBS_CHONF IR+Raman Fine-tuning (Inherits ONLY from QM9S_ir_raman)
    elif task in ["sdbs_chonf_ir_raman_finetuning", "sdbs_chonf", "task4"]:
        target_output_dir = dir_sdbs_chonf
        s_ckpt, i_ckpt = check_weights(dir_qm9s_ir_raman)
        if s_ckpt and i_ckpt:
            load_smiles_ckpt, load_ir_ckpt = s_ckpt, i_ckpt
            print(f"--> [Task 4] Found QM9S_ir_raman pre-trained weights! Loading from: {dir_qm9s_ir_raman}")
        else:
            print(f"--> [Task 4] No QM9S_ir_raman weights found. Fine-tuning from scratch.")
        print(f"--> Output directory: {target_output_dir}")

    else:
        # Fallback to configured output_dir
        target_output_dir = config['paths'].get('output_dir', dir_qm9s_ir)
        print(f"--> Custom Task: Output directory set to: {target_output_dir}")

    config['paths']['output_dir'] = target_output_dir
    os.makedirs(target_output_dir, exist_ok=True)

    return load_smiles_ckpt, load_ir_ckpt


# ==========================================
# Validation Logic
# ==========================================
def validate_model(smiles_model, ir_model, val_loader, device):
    smiles_model.eval()
    ir_model.eval()
    running_loss = 0.0
    result_smiles_features = []
    result_ir_features = []

    with torch.no_grad():
        for spectra_batch, smiles_batch in tqdm(val_loader, desc="Validating", unit="batch", leave=False):
            spectra_tensor = spectra_batch.to(device)

            tokenizer = smiles_model.smiles_tokenizer
            encoded_smiles = [
                tokenizer.encode_plus(text=s, max_length=smiles_model.smiles_maxlen, padding='max_length',
                                      truncation=True, return_tensors='pt') for s in smiles_batch
            ]
            input_ids = torch.cat([item['input_ids'] for item in encoded_smiles], dim=0).to(device)
            attention_mask = torch.cat([item['attention_mask'] for item in encoded_smiles], dim=0).to(device)
            lengths = attention_mask.sum(dim=1)

            with autocast():
                smiles_features = smiles_model.encode((input_ids, attention_mask), lengths)
                ir_features = ir_model(spectra_tensor)
                result_smiles_features.append(smiles_features)
                result_ir_features.append(ir_features)

                t = torch.exp(smiles_model.t_prime)
                b = smiles_model.bias
                logits = torch.matmul(ir_features, smiles_features.T) * t + b
                n = logits.size(0)
                labels = 2 * torch.eye(n).to(device) - torch.ones(n, n).to(device)
                loss = -torch.sum(F.logsigmoid(labels * logits)) / n

            running_loss += loss.item() * spectra_tensor.size(0)

    # Calculate Recall@1 (Top-1 Accuracy)
    all_smiles_features = torch.cat(result_smiles_features, 0)
    all_ir_features = torch.cat(result_ir_features, 0)
    logits_full = torch.matmul(all_ir_features, all_smiles_features.T)
    top1_indices = torch.argmax(logits_full, dim=1)
    correct_matches = (top1_indices == torch.arange(len(top1_indices)).to(device)).sum().item()
    total_samples = len(top1_indices)
    top_1_ratio = correct_matches / total_samples if total_samples > 0 else 0

    val_loss = running_loss / len(val_loader.dataset)
    return val_loss, top_1_ratio


# ==========================================
# Main Training Loop with Early Stopping
# ==========================================
def train_model(config, smiles_model, ir_model, train_loader, val_loader, optimizer, device):
    scaler = GradScaler()
    output_dir = config['paths']['output_dir']
    num_epochs = config['training_params']['num_epochs']
    patience = config['training_params'].get('patience', 15)
    warmup_epochs = config['scheduler_params']['warmup_epochs']

    best_val_ratio = -1.0
    early_stop_counter = 0

    best_smiles_path = os.path.join(output_dir, 'best_smiles_model.pth')
    best_ir_path = os.path.join(output_dir, 'best_ir_model.pth')

    scheduler_warmup = LambdaLR(optimizer, lr_lambda=lambda epoch: get_lr_multiplier(epoch, warmup_epochs))
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=(num_epochs - warmup_epochs))

    training_losses, validation_losses, validation_recalls = [], [], []

    for epoch in range(num_epochs):
        smiles_model.train()
        ir_model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

        for spectra_batch, smiles_batch in train_loader_tqdm:
            spectra_tensor = spectra_batch.to(device)

            tokenizer = smiles_model.smiles_tokenizer
            encoded_smiles = [
                tokenizer.encode_plus(text=s, max_length=smiles_model.smiles_maxlen, padding='max_length',
                                      truncation=True, return_tensors='pt') for s in smiles_batch
            ]
            input_ids = torch.cat([item['input_ids'] for item in encoded_smiles], dim=0).to(device)
            attention_mask = torch.cat([item['attention_mask'] for item in encoded_smiles], dim=0).to(device)
            lengths = attention_mask.sum(dim=1)

            optimizer.zero_grad()
            with autocast():
                smiles_features = smiles_model.encode((input_ids, attention_mask), lengths)
                ir_features = ir_model(spectra_tensor)
                t = torch.exp(smiles_model.t_prime)
                b = smiles_model.bias
                logits = torch.matmul(ir_features, smiles_features.T) * t + b
                n = logits.size(0)
                labels = 2 * torch.eye(n).to(device) - torch.ones(n, n).to(device)
                loss = -torch.sum(F.logsigmoid(labels * logits)) / n

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * spectra_tensor.size(0)
            train_loader_tqdm.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        training_losses.append(epoch_loss)

        # Validation phase
        val_loss, top_1_ratio = validate_model(smiles_model, ir_model, val_loader, device)
        validation_losses.append(val_loss)
        validation_recalls.append(top_1_ratio)

        print(f"Epoch {epoch + 1}/{num_epochs} -> Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Recall@1: {top_1_ratio:.4f}")

        # Update learning rate schedule
        (scheduler_warmup if epoch < warmup_epochs else scheduler_cosine).step()

        # Check Early Stopping & Save Best Models
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
            "train_losses": training_losses,
            "val_losses": validation_losses,
            "val_recalls": validation_recalls
        }
        with open(os.path.join(output_dir, 'history.json'), 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=4)

        # Trigger Early Stop Termination
        if early_stop_counter >= patience:
            print(f"\n🛑 [Early Stopping Triggered] Val Recall@1 has not improved for {patience} consecutive epochs. Terminating training.")
            break

    print('\n================ Training Summary ================')
    print(f'Best Validation Recall@1: {best_val_ratio:.4f}')
    print(f'Model weights & history saved to: {output_dir}')


def main():
    parser = argparse.ArgumentParser(description="Train CSU-IR for SOTA Comparison.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, '..', config_path)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Convert paths relative to PROJECT_ROOT
    for key, path in config['paths'].items():
        if path and not os.path.isabs(path):
            config['paths'][key] = os.path.join(PROJECT_ROOT, path)

    # Auto-resolve SOTA comparison checkpoints and output directories
    auto_smiles_ckpt, auto_ir_ckpt = resolve_sota_checkpoints_and_paths(config, PROJECT_ROOT)

    # Setup device
    if config['training_params']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['training_params']['device'])
    print(f"Using device: {device}")

    # Initialize models
    print("Initializing models...")
    ir_model_config = config['model_params']['ir_model']
    IR_model = IRModel(**ir_model_config)
    smiles_model_config = config['model_params']['smiles_model']
    Smiles_Model = SmilesModel(roberta_model_path=None, roberta_tokenizer_path=config['paths']['tokenizer'],
                               **smiles_model_config)

    print(f"SmilesModel Parameters: {count_parameters(Smiles_Model)}")
    print(f"IR_model Parameters: {count_parameters(IR_model)}")

    IR_model.to(device)
    Smiles_Model.to(device)

    # Checkpoint loading priority: Config explicit path > Auto-resolved SOTA path
    final_ir_ckpt = config['paths'].get('ir_model_check_point') or auto_ir_ckpt
    if final_ir_ckpt and os.path.exists(final_ir_ckpt):
        IR_model.load_state_dict(torch.load(final_ir_ckpt, map_location=device))
        print(f"✅ Loaded IR_model checkpoint from: {final_ir_ckpt}")
    else:
        print("ℹ️ Training IR_model from scratch.")

    final_smiles_ckpt = config['paths'].get('smiles_model_check_point') or auto_smiles_ckpt
    if final_smiles_ckpt and os.path.exists(final_smiles_ckpt):
        Smiles_Model.load_state_dict(torch.load(final_smiles_ckpt, map_location=device))
        print(f"✅ Loaded Smiles_Model checkpoint from: {final_smiles_ckpt}")
    else:
        print("ℹ️ Training Smiles_Model from scratch.")

    # Load datasets (supports optional Raman for multi-modal concat)
    print("Loading datasets...")
    train_raman = config['paths'].get('train_raman')
    val_raman = config['paths'].get('val_raman')

    smiles_train, spectra_train = load_smiles_and_spectra(
        config['paths']['train_smiles'], config['paths']['train_ir'], train_raman
    )
    smiles_val, spectra_val = load_smiles_and_spectra(
        config['paths']['val_smiles'], config['paths']['val_ir'], val_raman
    )

    train_dataset = MultiModalDataset(spectra_train, smiles_train)
    val_dataset = MultiModalDataset(spectra_val, smiles_val)

    dl_params = config['dataloader_params']
    train_loader = DataLoader(train_dataset, batch_size=dl_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=dl_params['batch_size'], shuffle=False)

    print(f"Training samples  : {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Setup optimizer
    opt_params = config['optimizer_params']
    optimizer = AdamW(list(Smiles_Model.parameters()) + list(IR_model.parameters()), 
                      lr=opt_params['learning_rate'],
                      weight_decay=opt_params['weight_decay'])

    # Start training loop
    train_model(config, Smiles_Model, IR_model, train_loader, val_loader, optimizer, device)


if __name__ == '__main__':
    main()
