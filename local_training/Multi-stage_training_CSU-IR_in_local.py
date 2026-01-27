#  python -m local/Multi-staged_training_CSU-IR_in_local --config configs/config_CSU-IR_Multi-stage_training_I_MD.yaml
#  python -m local/Multi-staged_training_CSU-IR_in_local --config configs/config_CSU-IR_Multi-stage_training_II_DFT.yaml
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


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CSU-IR'))
print(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# If there is a red underline below, don't worry, it will not affect the code running
from model.IR_encoder import IRModel
from model.SMILES_encoder import SmilesModel


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


class IRSmilesDataset(Dataset):
    def __init__(self, ir_spectra, smiles):
        self.ir_spectra = ir_spectra
        self.smiles = smiles

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.ir_spectra[idx], self.smiles[idx]



def validate_model(smiles_model, ir_model, val_loader, device):
    smiles_model.eval()
    ir_model.eval()
    running_loss = 0.0
    result_smiles_features = []
    result_ir_features = []
    val_loader_tqdm = tqdm(val_loader, desc="Validating", unit="batch", leave=False)

    with torch.no_grad():
        for ir_spectra_batch, smiles_batch in val_loader_tqdm:
            ir_spectra_tensor = ir_spectra_batch.to(device)

            tokenizer = smiles_model.smiles_tokenizer
            encoded_smiles = [tokenizer.encode_plus(text=s, max_length=smiles_model.smiles_maxlen, padding='max_length',
                                                    truncation=True, return_tensors='pt') for s in smiles_batch]
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
            val_loader_tqdm.set_postfix(loss=loss.item())

    # Calculate Top-1 accuracy
    all_smiles_features = torch.cat(result_smiles_features, 0)
    all_ir_features = torch.cat(result_ir_features, 0)
    logits_full = torch.matmul(all_ir_features, all_smiles_features.T)
    top1_indices = torch.argmax(logits_full, dim=1)
    correct_matches = (top1_indices == torch.arange(len(top1_indices)).to(device)).sum().item()
    total_samples = len(top1_indices)
    top_1_ratio = correct_matches / total_samples if total_samples > 0 else 0

    val_loss = running_loss / len(val_loader.dataset)
    return val_loss, top_1_ratio


def train_model(config, smiles_model, ir_model, train_loader, val_loader, optimizer, device):
    scaler = GradScaler()
    output_dir = config['paths']['output_dir']
    num_epochs = config['training_params']['num_epochs']
    warmup_epochs = config['scheduler_params']['warmup_epochs']
    num_best_models = config['model_save_params']['num_best_models']

    scheduler_warmup = LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=(num_epochs - warmup_epochs))

    os.makedirs(output_dir, exist_ok=True)

    best_models_tracker = []  # List of tuples (ratio, epoch, smiles_path, ir_path)
    training_losses, validation_losses = [], []

    for epoch in range(num_epochs):
        smiles_model.train()
        ir_model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

        for ir_spectra_batch, smiles_batch in train_loader_tqdm:
            ir_spectra_tensor = ir_spectra_batch.to(device)

            tokenizer = smiles_model.smiles_tokenizer
            encoded_smiles = [tokenizer.encode_plus(text=s, max_length=smiles_model.smiles_maxlen, padding='max_length',
                                                    truncation=True, return_tensors='pt') for s in smiles_batch]
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
                labels = 2 * torch.eye(n).to(device) - torch.ones(n, n).to(device)
                loss = -torch.sum(F.logsigmoid(labels * logits)) / n

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        training_losses.append(epoch_loss)

        val_loss, top_1_ratio = validate_model(smiles_model, ir_model, val_loader, device, sme)
        validation_losses.append(val_loss)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Top-1 Ratio: {top_1_ratio:.4f}')

        # Update learning rate
        if epoch < 10:
            scheduler_warmup.step()
        else:
            scheduler_cosine.step()

        # Save top N best models
        current_model_info = (top_1_ratio, epoch + 1, None, None)
        if len(best_models_tracker) < num_best_models:
            best_models_tracker.append(current_model_info)
        elif top_1_ratio > min(best_models_tracker, key=lambda x: x[0])[0]:
            worst_model_info = min(best_models_tracker, key=lambda x: x[0])
            if worst_model_info[2] and os.path.exists(worst_model_info[2]): os.remove(worst_model_info[2])
            if worst_model_info[3] and os.path.exists(worst_model_info[3]): os.remove(worst_model_info[3])
            best_models_tracker.remove(worst_model_info)
            best_models_tracker.append(current_model_info)

        # Save the current state of best models to disk
        for i, (ratio, ep, _, _) in enumerate(best_models_tracker):
            if ep == epoch + 1:  # Only save if it's the current epoch's model
                smiles_path = os.path.join(output_dir, f'smiles_model_epoch_{ep}_ratio_{ratio:.4f}.pth')
                ir_path = os.path.join(output_dir, f'ir_model_epoch_{ep}_ratio_{ratio:.4f}.pth')
                torch.save(smiles_model.state_dict(), smiles_path)
                torch.save(ir_model.state_dict(), ir_path)
                best_models_tracker[i] = (ratio, ep, smiles_path, ir_path)

    print('\nTraining complete.')
    print('Best validation models saved:')
    for ratio, epoch, smiles_path, _ in sorted(best_models_tracker, key=lambda x: x[0], reverse=True):
        print(f"  - Epoch {epoch}: Top-1 Ratio = {ratio:.4f}, Path: {smiles_path}")

    # Save loss data
    loss_data = {'training_losses': training_losses, 'validation_losses': validation_losses}
    loss_file_path = os.path.join(output_dir, config['model_save_params']['loss_log_file'])
    with open(loss_file_path, 'w') as f:
        json.dump(loss_data, f, indent=2)
    print(f"\nLoss data saved to {loss_file_path}")



def main():
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Train CSU-IR models.")
        parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
        args = parser.parse_args()
        config_path = args.config
    else:
        default_config_relative_path = "configs/config_CSU-IR_Multi-stage_training_I_MD.yaml"
        config_path = os.path.join(PROJECT_ROOT,'..' ,default_config_relative_path)
        print(f"No config provided via command line. Using default: {config_path}")

    # Load configuration from YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve all paths relative to the project root
    for key, path in config['paths'].items():
        if not os.path.isabs(path):
            config['paths'][key] = os.path.join(PROJECT_ROOT, path)

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
    ir_checkpoint_path = config['paths'].get('ir_model_check_point')
    if ir_checkpoint_path and os.path.exists(ir_checkpoint_path):
        try:
            IR_model.load_state_dict(torch.load(ir_checkpoint_path, map_location=device))
            print(f"Successfully loaded IR_model checkpoint from: {ir_checkpoint_path}")
        except Exception as e:
            print(f"Warning: Failed to load IR_model checkpoint from {ir_checkpoint_path}. Error: {e}")
    else:
        print("No valid IR_model checkpoint path provided. Training from scratch.")

    smiles_checkpoint_path = config['paths'].get('smiles_model_check_point')
    if smiles_checkpoint_path and os.path.exists(smiles_checkpoint_path):
        try:
            Smiles_Model.load_state_dict(torch.load(smiles_checkpoint_path, map_location=device))
            print(f"Successfully loaded Smiles_Model checkpoint from: {smiles_checkpoint_path}")
        except Exception as e:
            print(f"Warning: Failed to load Smiles_Model checkpoint from {smiles_checkpoint_path}. Error: {e}")
    else:
        print("No valid Smiles_Model checkpoint path provided. Training from scratch.")

    # Load data
    print("Loading data...")
    smiles_train, ir_train = load_smiles_ir(config['paths']['train_smiles'], config['paths']['train_ir'])
    smiles_val, ir_val = load_smiles_ir(config['paths']['val_smiles'], config['paths']['val_ir'])

    train_dataset = IRSmilesDataset(ir_train, smiles_train)
    val_dataset = IRSmilesDataset(ir_val, smiles_val)

    dl_params = config['dataloader_params']
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=dl_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=dl_params['batch_size'], shuffle=False)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    # Setup optimizer and schedulers
    opt_params = config['optimizer_params']
    sched_params = config['scheduler_params']
    optimizer = AdamW(list(Smiles_Model.parameters()) + list(IR_model.parameters()), lr=opt_params['learning_rate'],
                      weight_decay=opt_params['weight_decay'])
    

    # Start training
    train_model(config, Smiles_Model, IR_model, train_loader, val_loader, optimizer, device)


if __name__ == '__main__':
    main()

