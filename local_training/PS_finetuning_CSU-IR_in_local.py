import os
import sys
import yaml
import json
import random
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rdkit import Chem

# Disable tokenizers parallelism in multi-processing environments
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# Locate project root dynamically
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CSU-IR'))
if not os.path.exists(PROJECT_ROOT):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from model.IR_encoder import IRModel
from model.SMILES_encoder import SmilesModel


# ==============================================================================
#  Basic Utilities
# ==============================================================================
def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_smiles(smiles):
    """Removes stereochemistry and canonicalizes SMILES via RDKit."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    except Exception:
        pass
    return smiles


# ==============================================================================
#  Dataset & DualEncoder Model Definition
# ==============================================================================
def load_data(smiles_path, ir_path):
    """Loads pre-prepared SMILES strings and IR tensor."""
    with open(smiles_path, "r", encoding="utf-8") as f:
        smiles_list = f.read().splitlines()
    ir_tensor = torch.load(ir_path, map_location="cpu")
    if ir_tensor.dim() == 3 and ir_tensor.size(1) == 1:
        ir_tensor = ir_tensor.squeeze(1)
    return smiles_list, ir_tensor


class IRSmilesDataset(Dataset):
    def __init__(self, ir_spectra, smiles, tokenizer, max_len=300):
        self.ir_spectra = ir_spectra
        self.smiles = smiles
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        ir = self.ir_spectra[idx]
        s = self.smiles[idx]
        s_str = s if s is not None else ""

        encoded = self.tokenizer.encode_plus(
            s_str,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return (
            ir,
            encoded["input_ids"].squeeze(0),
            encoded["attention_mask"].squeeze(0),
            s_str,
        )


class DualEncoder(nn.Module):
    def __init__(self, smiles_model, ir_model):
        super().__init__()
        self.smiles_model = smiles_model
        self.ir_model = ir_model

    def forward(self, smiles_inputs, smiles_lengths, ir_spectra):
        input_ids, attention_mask = smiles_inputs
        smiles_features = self.smiles_model.encode((input_ids, attention_mask), smiles_lengths)
        ir_features = self.ir_model(ir_spectra)

        t = torch.exp(self.smiles_model.t_prime)
        b = self.smiles_model.bias
        logits = torch.matmul(ir_features, smiles_features.T) * t + b
        n = logits.size(0)
        labels = 2 * torch.eye(n, device=logits.device) - 1
        loss = -torch.sum(F.logsigmoid(labels * logits)) / n

        return loss


# ==============================================================================
#  Feature Embedding & Derivative Library Retrieval Evaluation
# ==============================================================================
def embed_ir_tensors(ir_data, ir_model, device, batch_size=256):
    ir_model.eval()
    all_features = []
    for i in range(0, len(ir_data), batch_size):
        batch_ir = ir_data[i: i + batch_size].to(device)
        with torch.no_grad():
            with autocast():
                features = ir_model(batch_ir)
                all_features.append(features.cpu())
    return torch.cat(all_features, dim=0)


def embed_smiles_list(smiles_list, smiles_model, tokenizer, device, batch_size=256):
    smiles_model.eval()
    all_features = []
    for i in range(0, len(smiles_list), batch_size):
        batch_s = smiles_list[i: i + batch_size]
        batch_s = [s if s is not None else "" for s in batch_s]

        encoded = tokenizer(batch_s, max_length=300, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        lengths = attention_mask.sum(dim=1)

        with torch.no_grad():
            with autocast():
                features = smiles_model.encode((input_ids, attention_mask), lengths)
                all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


def evaluate_derivative_library_retrieval(
    library_path, query_smiles, query_ir_features, smiles_model, tokenizer, device
):
    """
    Evaluates Validation IR spectrum queries against the PS Derivative background library.
    """
    if not os.path.exists(library_path):
        print(f"❌ [Error] Derivative background library not found at: {library_path}")
        return None

    with open(library_path, "r", encoding="utf-8") as f:
        bg_smiles = [line.strip() for line in f if line.strip()]

    bg_smiles_norm = [normalize_smiles(s) for s in bg_smiles if normalize_smiles(s)]
    query_smiles_norm = [normalize_smiles(s) for s in query_smiles]

    # Gallery = Background Derivative Library + Query Ground-truth Molecules
    gallery_smiles = list(set(bg_smiles_norm + [s for s in query_smiles_norm if s is not None]))
    smiles_to_gallery_idx = {s: idx for idx, s in enumerate(gallery_smiles)}

    target_indices = []
    valid_query_mask = []
    for s in query_smiles_norm:
        if s in smiles_to_gallery_idx:
            target_indices.append(smiles_to_gallery_idx[s])
            valid_query_mask.append(True)
        else:
            target_indices.append(-1)
            valid_query_mask.append(False)

    target_indices = torch.tensor(target_indices, device=device)
    valid_query_mask = torch.tensor(valid_query_mask, dtype=torch.bool, device=device)
    total_valid_queries = valid_query_mask.sum().item()

    gallery_smiles_features = embed_smiles_list(
        gallery_smiles, smiles_model, tokenizer, device, batch_size=256
    )

    q_feats = query_ir_features.to(device)
    g_feats = gallery_smiles_features.to(device)

    correct_matches_top1, correct_matches_top5, correct_matches_top10 = 0, 0, 0
    eval_batch_size = 256

    for i in range(0, len(q_feats), eval_batch_size):
        end = min(i + eval_batch_size, len(q_feats))
        batch_mask = valid_query_mask[i:end]
        if not batch_mask.any():
            continue

        logits_batch = torch.matmul(q_feats[i:end], g_feats.T)
        k_val = min(10, g_feats.size(0))
        _, top_k_indices = torch.topk(logits_batch, k=k_val, dim=1)

        predicted_ids = top_k_indices
        true_ids = target_indices[i:end].unsqueeze(1)
        matches = (predicted_ids == true_ids) & batch_mask.unsqueeze(1)

        correct_matches_top1 += matches[:, :1].any(dim=1).sum().item()
        correct_matches_top5 += matches[:, :5].any(dim=1).sum().item()
        correct_matches_top10 += matches[:, :10].any(dim=1).sum().item()

    top_1_ratio = correct_matches_top1 / total_valid_queries if total_valid_queries > 0 else 0
    top_5_ratio = correct_matches_top5 / total_valid_queries if total_valid_queries > 0 else 0
    top_10_ratio = correct_matches_top10 / total_valid_queries if total_valid_queries > 0 else 0

    return {
        "Recall@1": top_1_ratio,
        "Recall@5": top_5_ratio,
        "Recall@10": top_10_ratio,
        "Gallery_Size": len(gallery_smiles),
    }


# ==============================================================================
#  Training Loop: Fixed Epochs with Best Checkpoint Selection
# ==============================================================================
def run_finetune_pipeline(
    dual_encoder, train_loader, val_smiles, val_ir,
    derivative_lib_path, tokenizer, config, device
):
    t_params = config['training_params']
    num_epochs = t_params['num_epochs']
    lr = float(t_params['learning_rate'])
    warmup_epochs = t_params['warmup_epochs']
    output_dir = config['paths']['output_dir']

    optimizer = AdamW(dual_encoder.parameters(), lr=lr, weight_decay=float(t_params['weight_decay']))
    scheduler_warmup = LambdaLR(optimizer, lambda e: min(1.0, (e + 1) / warmup_epochs))
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs))
    scaler = GradScaler()

    best_val_score = -1.0
    best_epoch = -1

    raw_smiles = dual_encoder.smiles_model
    raw_ir = dual_encoder.ir_model

    best_smiles_path = os.path.join(output_dir, "best_smiles_model.pth")
    best_ir_path = os.path.join(output_dir, "best_ir_model.pth")

    history_log = []

    print("\n" + "=" * 80)
    print(f" 🚀 Starting PS Fine-tuning (Fixed {num_epochs} Epochs)")
    print(" Best Model Selection Metric: Validation Set Recall@10 in PS Derivative Library")
    print("=" * 80)

    for epoch in range(num_epochs):
        dual_encoder.train()
        running_loss, total_train_samples = 0.0, 0

        for ir_b, ids_b, mask_b, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            ir_t = ir_b.to(device)
            ids = ids_b.to(device)
            mask = mask_b.to(device)
            lengths = mask.sum(dim=1)

            optimizer.zero_grad()
            with autocast():
                loss = dual_encoder((ids, mask), lengths, ir_t)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * ir_t.size(0)
            total_train_samples += ir_t.size(0)

        epoch_loss = running_loss / total_train_samples if total_train_samples > 0 else 0

        # Evaluate on Validation set against the PS Derivative background library
        val_ir_feats = embed_ir_tensors(val_ir, raw_ir, device)
        val_res = evaluate_derivative_library_retrieval(
            derivative_lib_path, val_smiles, val_ir_feats,
            raw_smiles, tokenizer, device
        )

        val_r1 = val_res["Recall@1"] if val_res else 0.0
        val_r5 = val_res["Recall@5"] if val_res else 0.0
        val_r10 = val_res["Recall@10"] if val_res else 0.0
        val_score = val_r10  # Selection criterion: Recall@10

        print(
            f"Epoch [{epoch + 1:02d}/{num_epochs:02d}] Train Loss: {epoch_loss:.4f} | "
            f"Val-in-DerivativeLib [R@1: {val_r1:.2%}, R@5: {val_r5:.2%}, R@10: {val_r10:.2%}]"
        )

        # Update best checkpoint
        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch + 1
            torch.save(raw_smiles.state_dict(), best_smiles_path)
            torch.save(raw_ir.state_dict(), best_ir_path)
            print(f"  ✨ [New Best at Epoch {best_epoch}] Val Recall@10 reached: {best_val_score:.2%}. Checkpoints updated!")

        history_log.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_recall@1": val_r1,
            "val_recall@5": val_r5,
            "val_recall@10": val_r10
        })

        (scheduler_warmup if epoch < warmup_epochs else scheduler_cosine).step()

    # Save training history
    with open(os.path.join(output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history_log, f, indent=4)

    print("\n" + "=" * 80)
    print(f" 🎉 Training Complete!")
    print(f"    - Best Epoch            : Epoch {best_epoch}")
    print(f"    - Best Val Recall@10    : {best_val_score:.2%}")
    print(f"    - Best Weights Saved To : {output_dir}")
    print("=" * 80)


# ==============================================================================
#  Main Execution Entry
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="PS Subset Fine-tuning & Derivative Retrieval Evaluation.")
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

    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    set_seed(config['training_params'].get('seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize models
    ir_model_config = config['model_params']['ir_model']
    IR_model = IRModel(**ir_model_config)
    smiles_model_config = config['model_params']['smiles_model']
    Smiles_Model = SmilesModel(
        roberta_model_path=None,
        roberta_tokenizer_path=config['paths']['tokenizer'],
        **smiles_model_config
    )

    # Load initial pre-trained weights
    pretrained_ir = config['paths'].get('pretrained_ir_model')
    pretrained_smiles = config['paths'].get('pretrained_smiles_model')

    if pretrained_ir and os.path.exists(pretrained_ir):
        IR_model.load_state_dict(torch.load(pretrained_ir, map_location=device))
        print(f"✅ Loaded pre-trained IR model: {pretrained_ir}")
    if pretrained_smiles and os.path.exists(pretrained_smiles):
        Smiles_Model.load_state_dict(torch.load(pretrained_smiles, map_location=device))
        print(f"✅ Loaded pre-trained SMILES model: {pretrained_smiles}")

    IR_model.to(device)
    Smiles_Model.to(device)

    tokenizer = Smiles_Model.smiles_tokenizer
    dual_encoder = DualEncoder(smiles_model=Smiles_Model, ir_model=IR_model).to(device)

    # Load pre-prepared datasets (Train & Val only)
    print("\nLoading pre-prepared PS subset datasets...")
    train_smiles, train_ir = load_data(config['paths']['train_smiles'], config['paths']['train_ir'])
    val_smiles, val_ir = load_data(config['paths']['val_smiles'], config['paths']['val_ir'])

    train_dataset = IRSmilesDataset(train_ir, train_smiles, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training_params']['batch_size'],
        shuffle=True
    )

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_smiles)}")

    # Start Fine-tuning Pipeline
    run_finetune_pipeline(
        dual_encoder=dual_encoder,
        train_loader=train_loader,
        val_smiles=val_smiles,
        val_ir=val_ir,
        derivative_lib_path=config['paths']['ps_derivative_library'],
        tokenizer=tokenizer,
        config=config,
        device=device
    )


if __name__ == '__main__':
    main()
