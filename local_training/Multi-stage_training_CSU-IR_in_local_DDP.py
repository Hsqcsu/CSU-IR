import sys
import os
import yaml
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
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


# ==========================================
# Distributed & Device Helpers
# ==========================================
def is_dist_avail_and_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()


def is_main_process():
    return get_rank() == 0


def get_raw_model(model):
    """Unwraps model from DataParallel or DistributedDataParallel."""
    return model.module if hasattr(model, 'module') else model


def setup_device_and_distributed():
    """
    Automatically detects environment: Single-GPU, Multi-GPU (DDP or DataParallel), or CPU.
    """
    # 1. Check if launched with torchrun / DDP environment variables
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=rank
        )
        device = torch.device(f"cuda:{local_rank}")
        mode = "DDP"
        gpu_count = world_size
    elif torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            device = torch.device("cuda:0")
            mode = "DataParallel"
        else:
            device = torch.device("cuda:0")
            mode = "Single-GPU"
    else:
        device = torch.device("cpu")
        mode = "CPU"
        gpu_count = 0

    if is_main_process():
        print(f"\n🖥️  [Hardware Environment Detected]")
        print(f"    - Execution Mode : {mode}")
        print(f"    - Primary Device : {device}")
        print(f"    - Available GPUs : {gpu_count}")
        if gpu_count > 0:
            for i in range(gpu_count):
                print(f"      * GPU {i}: {torch.cuda.get_device_name(i)}")
        print("=" * 60)

    return device, mode


def load_smiles_ir(smiles_path, ir_path):
    with open(smiles_path, 'r', encoding='utf-8') as f:
        smiles = f.read().splitlines()
    ir = torch.load(ir_path, map_location='cpu')
    if ir.dim() == 3 and ir.size(1) == 1:
        ir = ir.squeeze(1)
    return smiles, ir


def get_lr_multiplier(epoch, warmup_epochs):
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    return 1.0


def count_parameters(model):
    raw_model = get_raw_model(model)
    return sum(p.numel() for p in raw_model.parameters() if p.requires_grad)


class IRSmilesDataset(Dataset):
    def __init__(self, ir_spectra, smiles):
        self.ir_spectra = ir_spectra
        self.smiles = smiles

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.ir_spectra[idx], self.smiles[idx]


# ==========================================
# Multi-Stage Checkpoint Auto-Resolver
# ==========================================
def resolve_multi_stage_checkpoints_and_paths(config, project_root):
    """
    Automatically resolves pre-trained weights and destination output directory
    based on multi-stage training logic.
    """
    ckpt_root = os.path.join(project_root, "check_points")
    stage_1_dir = os.path.join(ckpt_root, "Multi-stage_training_Stage_I_MD")
    
    stage_2_base = os.path.join(ckpt_root, "Multi-stage_training_Stage_II_DFT")
    stage_2_md_dft = os.path.join(stage_2_base, "MD_DFT")
    stage_2_no_md = os.path.join(stage_2_base, "DFT_without_MD_pretraining")
    
    stage_3_base = os.path.join(ckpt_root, "Multi-stage_training_Stage_III_EXP")
    stage_3_md_dft_exp = os.path.join(stage_3_base, "MD_DFT_EXP")
    stage_3_dft_exp = os.path.join(stage_3_base, "DFT_EXP")
    stage_3_md_exp = os.path.join(stage_3_base, "MD_EXP")
    stage_3_no_pretrain = os.path.join(stage_3_base, "EXP_without_any_pretraining")

    def check_weights(folder):
        smiles_p = os.path.join(folder, "best_smiles_model.pth")
        ir_p = os.path.join(folder, "best_ir_model.pth")
        if os.path.exists(smiles_p) and os.path.exists(ir_p):
            return smiles_p, ir_p
        return None, None

    stage = config.get("training_params", {}).get("stage", "MD").upper()
    load_smiles_ckpt, load_ir_ckpt = None, None
    target_output_dir = None

    if is_main_process():
        print(f"\n{'='*25} Multi-Stage Auto Routing: [STAGE {stage}] {'='*25}")

    if stage in ["MD", "STAGE_I"]:
        target_output_dir = stage_1_dir
        if is_main_process():
            print(f"--> Stage I (MD): Training from scratch. Output: {target_output_dir}")

    elif stage in ["DFT", "STAGE_II"]:
        # Check Stage I (MD)
        s_ckpt, i_ckpt = check_weights(stage_1_dir)
        if s_ckpt and i_ckpt:
            load_smiles_ckpt, load_ir_ckpt = s_ckpt, i_ckpt
            target_output_dir = stage_2_md_dft
            if is_main_process():
                print(f"--> Found Stage I (MD) weights! Inheriting weights from: {stage_1_dir}")
                print(f"--> Output directory set to: {target_output_dir}")
        else:
            target_output_dir = stage_2_no_md
            if is_main_process():
                print(f"--> No Stage I weights found. Training DFT from scratch. Output: {target_output_dir}")

    elif stage in ["EXP", "EB", "STAGE_III"]:
        # Priority 1: Stage II MD_DFT
        s_ckpt, i_ckpt = check_weights(stage_2_md_dft)
        if s_ckpt and i_ckpt:
            load_smiles_ckpt, load_ir_ckpt = s_ckpt, i_ckpt
            target_output_dir = stage_3_md_dft_exp
            if is_main_process():
                print(f"--> [Priority 1 Hit] Loaded MD_DFT weights from: {stage_2_md_dft}")
                print(f"--> Output directory set to: {target_output_dir}")
        else:
            # Priority 2: Stage II DFT without MD
            s_ckpt, i_ckpt = check_weights(stage_2_no_md)
            if s_ckpt and i_ckpt:
                load_smiles_ckpt, load_ir_ckpt = s_ckpt, i_ckpt
                target_output_dir = stage_3_dft_exp
                if is_main_process():
                    print(f"--> [Priority 2 Hit] Loaded DFT_without_MD weights from: {stage_2_no_md}")
                    print(f"--> Output directory set to: {target_output_dir}")
            else:
                # Priority 3: Stage I MD only
                s_ckpt, i_ckpt = check_weights(stage_1_dir)
                if s_ckpt and i_ckpt:
                    load_smiles_ckpt, load_ir_ckpt = s_ckpt, i_ckpt
                    target_output_dir = stage_3_md_exp
                    if is_main_process():
                        print(f"--> [Priority 3 Hit] Loaded MD ONLY weights from: {stage_1_dir}")
                        print(f"--> Output directory set to: {target_output_dir}")
                else:
                    target_output_dir = stage_3_no_pretrain
                    if is_main_process():
                        print(f"--> [Priority 4 Hit] No pre-trained weights found. Training EXP from scratch.")
                        print(f"--> Output directory set to: {target_output_dir}")

    # Override with config output_dir if explicitly provided and not dynamically resolved
    if target_output_dir:
        config['paths']['output_dir'] = target_output_dir
    if is_main_process():
        os.makedirs(config['paths']['output_dir'], exist_ok=True)

    return load_smiles_ckpt, load_ir_ckpt


# ==========================================
# Validation Logic
# ==========================================
def validate_model(smiles_model, ir_model, val_loader, device):
    smiles_model.eval()
    ir_model.eval()
    raw_smiles_model = get_raw_model(smiles_model)

    running_loss = 0.0
    result_smiles_features = []
    result_ir_features = []

    with torch.no_grad():
        for ir_spectra_batch, smiles_batch in tqdm(val_loader, desc="Validating", unit="batch", leave=False, disable=not is_main_process()):
            ir_spectra_tensor = ir_spectra_batch.to(device)

            tokenizer = raw_smiles_model.smiles_tokenizer
            encoded_smiles = [
                tokenizer.encode_plus(text=s, max_length=raw_smiles_model.smiles_maxlen, padding='max_length',
                                      truncation=True, return_tensors='pt') for s in smiles_batch
            ]
            input_ids = torch.cat([item['input_ids'] for item in encoded_smiles], dim=0).to(device)
            attention_mask = torch.cat([item['attention_mask'] for item in encoded_smiles], dim=0).to(device)
            lengths = attention_mask.sum(dim=1)

            with autocast():
                smiles_features = raw_smiles_model.encode((input_ids, attention_mask), lengths)
                ir_features = ir_model(ir_spectra_tensor)
                result_smiles_features.append(smiles_features)
                result_ir_features.append(ir_features)

                t = torch.exp(raw_smiles_model.t_prime)
                b = raw_smiles_model.bias
                logits = torch.matmul(ir_features, smiles_features.T) * t + b
                n = logits.size(0)
                labels = 2 * torch.eye(n).to(device) - torch.ones(n, n).to(device)
                loss = -torch.sum(F.logsigmoid(labels * logits)) / n

            running_loss += loss.item() * ir_spectra_tensor.size(0)

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
def train_model(config, smiles_model, ir_model, train_loader, val_loader, optimizer, device, train_sampler=None):
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

    raw_smiles_model = get_raw_model(smiles_model)

    for epoch in range(num_epochs):
        if train_sampler and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        smiles_model.train()
        ir_model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch", disable=not is_main_process())

        for ir_spectra_batch, smiles_batch in train_loader_tqdm:
            ir_spectra_tensor = ir_spectra_batch.to(device)

            tokenizer = raw_smiles_model.smiles_tokenizer
            encoded_smiles = [
                tokenizer.encode_plus(text=s, max_length=raw_smiles_model.smiles_maxlen, padding='max_length',
                                      truncation=True, return_tensors='pt') for s in smiles_batch
            ]
            input_ids = torch.cat([item['input_ids'] for item in encoded_smiles], dim=0).to(device)
            attention_mask = torch.cat([item['attention_mask'] for item in encoded_smiles], dim=0).to(device)
            lengths = attention_mask.sum(dim=1)

            optimizer.zero_grad()
            with autocast():
                smiles_features = raw_smiles_model.encode((input_ids, attention_mask), lengths)
                ir_features = ir_model(ir_spectra_tensor)
                t = torch.exp(raw_smiles_model.t_prime)
                b = raw_smiles_model.bias
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

        # Validation phase
        val_loss, top_1_ratio = validate_model(smiles_model, ir_model, val_loader, device)
        validation_losses.append(val_loss)
        validation_recalls.append(top_1_ratio)

        if is_main_process():
            print(f"Epoch {epoch + 1}/{num_epochs} -> Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Recall@1: {top_1_ratio:.4f}")

        # Update learning rate schedule
        (scheduler_warmup if epoch < warmup_epochs else scheduler_cosine).step()

        # Check Early Stopping & Save Best Models (Main process only)
        if is_main_process():
            if top_1_ratio > best_val_ratio:
                best_val_ratio = top_1_ratio
                early_stop_counter = 0
                torch.save(get_raw_model(smiles_model).state_dict(), best_smiles_path)
                torch.save(get_raw_model(ir_model).state_dict(), best_ir_path)
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

        # Broadcast early stopping decision if DDP
        if is_dist_avail_and_initialized():
            stop_tensor = torch.tensor([1 if early_stop_counter >= patience else 0], device=device)
            torch.distributed.broadcast(stop_tensor, src=0)
            if stop_tensor.item() == 1:
                if is_main_process():
                    print(f"\n🛑 [Early Stopping Triggered] Val Recall@1 has not improved for {patience} consecutive epochs. Terminating training.")
                break
        else:
            if early_stop_counter >= patience:
                print(f"\n🛑 [Early Stopping Triggered] Val Recall@1 has not improved for {patience} consecutive epochs. Terminating training.")
                break

    if is_main_process():
        print('\n================ Training Summary ================')
        print(f'Best Validation Recall@1: {best_val_ratio:.4f}')
        print(f'Model weights & history saved to: {output_dir}')


def main():
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--local-rank') and not sys.argv[1].startswith('--local_rank'):
        parser = argparse.ArgumentParser(description="Train CSU-IR models.")
        parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
        args, _ = parser.parse_known_args()
        config_path = args.config
    else:
        default_config_relative_path = "configs/config_CSU-IR_Multi-stage_training_I_MD.yaml"
        config_path = os.path.join(PROJECT_ROOT, '..', default_config_relative_path)
        if is_main_process():
            print(f"No config provided via command line. Using default: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Convert paths relative to PROJECT_ROOT
    for key, path in config['paths'].items():
        if path and not os.path.isabs(path):
            config['paths'][key] = os.path.join(PROJECT_ROOT, path)

    # 1. Setup device & multi-GPU environment
    device, mode = setup_device_and_distributed()

    # 2. Auto-resolve multi-stage checkpoints and output directories
    auto_smiles_ckpt, auto_ir_ckpt = resolve_multi_stage_checkpoints_and_paths(config, PROJECT_ROOT)

    # 3. Initialize models
    if is_main_process():
        print("Initializing models...")
    ir_model_config = config['model_params']['ir_model']
    IR_model = IRModel(**ir_model_config)
    smiles_model_config = config['model_params']['smiles_model']
    Smiles_Model = SmilesModel(roberta_model_path=None, roberta_tokenizer_path=config['paths']['tokenizer'],
                               **smiles_model_config)

    if is_main_process():
        print(f"SmilesModel Parameters: {count_parameters(Smiles_Model)}")
        print(f"IR_model Parameters: {count_parameters(IR_model)}")

    IR_model.to(device)
    Smiles_Model.to(device)

    # 4. Checkpoint loading priority: Config explicit path > Auto-resolved multi-stage path
    final_ir_ckpt = config['paths'].get('ir_model_check_point') or auto_ir_ckpt
    if final_ir_ckpt and os.path.exists(final_ir_ckpt):
        IR_model.load_state_dict(torch.load(final_ir_ckpt, map_location=device))
        if is_main_process():
            print(f"✅ Loaded IR_model checkpoint from: {final_ir_ckpt}")
    else:
        if is_main_process():
            print("ℹ️ Training IR_model from scratch.")

    final_smiles_ckpt = config['paths'].get('smiles_model_check_point') or auto_smiles_ckpt
    if final_smiles_ckpt and os.path.exists(final_smiles_ckpt):
        Smiles_Model.load_state_dict(torch.load(final_smiles_ckpt, map_location=device))
        if is_main_process():
            print(f"✅ Loaded Smiles_Model checkpoint from: {final_smiles_ckpt}")
    else:
        if is_main_process():
            print("ℹ️ Training Smiles_Model from scratch.")

    # 5. Multi-GPU Model Wrapping (DP or DDP)
    if mode == "DDP":
        local_rank = int(os.environ["LOCAL_RANK"])
        IR_model = nn.parallel.DistributedDataParallel(IR_model, device_ids=[local_rank], find_unused_parameters=True)
        Smiles_Model = nn.parallel.DistributedDataParallel(Smiles_Model, device_ids=[local_rank], find_unused_parameters=True)
    elif mode == "DataParallel":
        IR_model = nn.DataParallel(IR_model)
        Smiles_Model = nn.DataParallel(Smiles_Model)

    # 6. Load datasets
    if is_main_process():
        print("Loading datasets...")
    smiles_train, ir_train = load_smiles_ir(config['paths']['train_smiles'], config['paths']['train_ir'])
    smiles_val, ir_val = load_smiles_ir(config['paths']['val_smiles'], config['paths']['val_ir'])

    train_dataset = IRSmilesDataset(ir_train, smiles_train)
    val_dataset = IRSmilesDataset(ir_val, smiles_val)

    dl_params = config['dataloader_params']
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if mode == "DDP" else None

    train_loader = DataLoader(
        train_dataset, 
        batch_size=dl_params['batch_size'], 
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=dl_params.get('num_workers', 0),
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=dl_params['batch_size'], 
        shuffle=False,
        num_workers=dl_params.get('num_workers', 0),
        pin_memory=True if torch.cuda.is_available() else False
    )

    if is_main_process():
        print(f"Training samples  : {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

    # 7. Setup optimizer
    opt_params = config['optimizer_params']
    optimizer = AdamW(
        list(Smiles_Model.parameters()) + list(IR_model.parameters()), 
        lr=opt_params['learning_rate'],
        weight_decay=opt_params['weight_decay']
    )

    # 8. Start training loop
    train_model(config, Smiles_Model, IR_model, train_loader, val_loader, optimizer, device, train_sampler=train_sampler)

    # Clean up distributed process group if needed
    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
