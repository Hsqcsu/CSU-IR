import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast
from rdkit import Chem

# Project path configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
print(f"Project Root: {PROJECT_ROOT}")

from model.IR_encoder import IRModel
from model.SMILES_encoder import SmilesModel

TOKENIZER_PATH = os.path.join(PROJECT_ROOT, 'model', "tokenizer-smiles-roberta-1e_new")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==============================================================================
# 1. Basic Preprocessing Functions
# ==============================================================================
def normalize_smiles(smiles):
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    except Exception:
        pass
    return smiles


def load_smiles(smiles_path, normalize=True):
    with open(smiles_path, 'r', encoding='utf-8') as f:
        smiles_list = f.read().splitlines()
    if normalize:
        smiles_list = [normalize_smiles(s) for s in smiles_list if s.strip()]
        smiles_list = [s for s in smiles_list if s is not None]
    return smiles_list


def load_data(smiles_path, ir_path, normalize=True):
    smiles_list = load_smiles(smiles_path, normalize=normalize)
    ir_data = torch.load(ir_path)
    if not isinstance(ir_data, torch.Tensor):
        ir_data = torch.tensor(ir_data)
    return smiles_list, ir_data


# ==============================================================================
# 2. Batch Feature Extraction (AMP Supported)
# ==============================================================================
def embed_ir_tensors(ir_data, ir_model, device, batch_size=256):
    ir_model.eval()
    all_features = []
    for i in range(0, len(ir_data), batch_size):
        batch_ir = ir_data[i: i + batch_size].to(device)
        with torch.no_grad():
            with autocast():
                features = ir_model(batch_ir) if callable(ir_model) else ir_model.encode(batch_ir)
                all_features.append(features.cpu())
    return torch.cat(all_features, dim=0)


def embed_smiles_list(smiles_list, smiles_model, tokenizer, device, batch_size=256):
    smiles_model.eval()
    all_features = []
    for i in range(0, len(smiles_list), batch_size):
        batch_s = smiles_list[i: i + batch_size]
        batch_s = [s if s is not None else "" for s in batch_s]

        encoded = tokenizer(
            batch_s,
            max_length=300,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        lengths = attention_mask.sum(dim=1)

        with torch.no_grad():
            with autocast():
                try:
                    features = smiles_model.encode((input_ids, attention_mask), lengths)
                except TypeError:
                    features = smiles_model.encode((input_ids, attention_mask), CLS_pooling=True)
                all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


# ==============================================================================
# 3. Vectorized Retrieval Evaluation
# ==============================================================================
def evaluate_retrieval(query_smiles, query_ir_feats, gallery_smiles, gallery_smiles_feats, loss_type="sigmoid"):
    total_samples = len(query_smiles)

    smiles_to_gallery_idx = {s: idx for idx, s in enumerate(gallery_smiles)}
    target_indices = torch.tensor([smiles_to_gallery_idx[s] for s in query_smiles], device=device)

    if loss_type == "sigmoid":
        q_feats = query_ir_feats.to(device)
        g_feats = gallery_smiles_feats.to(device)
    else:
        q_feats = F.normalize(query_ir_feats, p=2, dim=-1).to(device)
        g_feats = F.normalize(gallery_smiles_feats, p=2, dim=-1).to(device)

    logits = torch.matmul(q_feats, g_feats.T)
    topk_scores, topk_indices = torch.topk(logits, k=min(10, g_feats.size(0)), dim=1)

    true_ids = target_indices.unsqueeze(1)
    matches = (topk_indices == true_ids)

    top_1_matches = matches[:, :1].any(dim=1).sum().item()
    top_5_matches = matches[:, :5].any(dim=1).sum().item()
    top_10_matches = matches[:, :10].any(dim=1).sum().item()

    top_1_ratio = top_1_matches / total_samples if total_samples > 0 else 0
    top_5_ratio = top_5_matches / total_samples if total_samples > 0 else 0
    top_10_ratio = top_10_matches / total_samples if total_samples > 0 else 0

    print(f"\nResults for NPS against Existing library:")
    print(f"Total Samples: {total_samples}")
    print(f"Recall@1  : {top_1_ratio:.4f}")
    print(f"Recall@5  : {top_5_ratio:.4f}")
    print(f"Recall@10 : {top_10_ratio:.4f}")

    return topk_scores.cpu().numpy(), topk_indices.cpu().numpy()


# ==============================================================================
# 4. Main Entry Point
# ==============================================================================
if __name__ == "__main__":
    Interference_library_path = os.path.join(PROJECT_ROOT, 'data', 'processed_library', 'PS', 'smiles_Existing_PS.txt')
    NPS_smiles_path = os.path.join(PROJECT_ROOT, 'data', 'test_data', 'NPS', 'filtered_final_NPS_smiles.txt')
    NPS_ir_path = os.path.join(PROJECT_ROOT, 'data', 'test_data', 'NPS', 'filtered_final_NPS_ir.pt')

    smiles_model_weight = os.path.join(PROJECT_ROOT, 'check_points', 'PS_finetuned', 'best_smiles_model.pth')
    ir_model_weight = os.path.join(PROJECT_ROOT, 'check_points', 'PS_finetuned', 'best_ir_model.pth')

    print("Loading and normalizing data...")
    NPS_smiles, NPS_ir = load_data(NPS_smiles_path, NPS_ir_path, normalize=True)
    Interference_library = load_smiles(Interference_library_path, normalize=True)

    gallery_smiles = list(set(Interference_library + NPS_smiles))
    print(f"Query Count: {len(NPS_smiles)}, Unique Gallery Size: {len(gallery_smiles)}")

    ir_model = IRModel().to(device)
    smiles_model = SmilesModel(
        roberta_model_path=None,
        roberta_tokenizer_path=TOKENIZER_PATH,
        smiles_maxlen=300,
        feature_dim=768
    ).to(device)

    tokenizer = smiles_model.smiles_tokenizer

    print("Loading model weights...")
    ir_model.load_state_dict(torch.load(ir_model_weight, map_location=device))
    smiles_model.load_state_dict(torch.load(smiles_model_weight, map_location=device))

    print("Extracting Query IR features...")
    query_ir_features = embed_ir_tensors(NPS_ir, ir_model, device)

    print("Extracting Gallery SMILES features...")
    gallery_smiles_features = embed_smiles_list(gallery_smiles, smiles_model, tokenizer, device)

    topk_scores, topk_indices = evaluate_retrieval(
        query_smiles=NPS_smiles,
        query_ir_feats=query_ir_features,
        gallery_smiles=gallery_smiles,
        gallery_smiles_feats=gallery_smiles_features,
        loss_type="sigmoid"
    )

    with open('NPS against Existing library_results.txt', 'w', encoding='utf-8') as file:
        for idx, (scores, indices) in enumerate(zip(topk_scores, topk_indices)):
            top_smiles = [gallery_smiles[i] for i in indices]
            file.write(f'Query SMILES: {NPS_smiles[idx]}\n')
            file.write(f'Top SMILES: {top_smiles}\n')
            file.write(f'Scores: {scores.tolist()}\n\n')
