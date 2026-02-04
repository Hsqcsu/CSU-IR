import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from collections import defaultdict

FEATURE_DIM = 1024
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_MW_Formula(path):
    with open(path, 'r') as f:
        MW_or_Formula = f.read().splitlines()
    return MW_or_Formula
    
def get_final_query_metadata(runtime_val, file_path):
    if runtime_val is not None and str(runtime_val).strip() != "":
        print(f"Using manually provided input: {runtime_val}")
        return str(runtime_val).strip()
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                lines = f.read().splitlines()
                if lines and lines[0].strip() != "":
                    print(f"Using input from file ({file_path}): {lines[0]}")
                    return lines[0].strip()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    return None
    
class UnifiedCombinedLibrary:
    """
    Unified library management class: Supports streaming reading and fast indexing based on molecular formula/molecular weight.
    """
    def __init__(self, library_configs):
        self.mmap_list = []
        self.smiles_list = []
        self.formulas_list = []
        self.mw_list = [] 
        self.cumulative_sizes = [0]
        
        self.formula_to_indices = defaultdict(list)
        self.mw_to_indices = defaultdict(list)

        total_count = 0
        for config in library_configs:
            print(f"Loading metadata for {config['name']}...")
            with open(config['smiles'], 'r', encoding='utf-8') as f:
                smi = f.read().splitlines()
            with open(config['formulas'], 'r', encoding='utf-8') as f:
                form = f.read().splitlines()
            count = len(smi)
            mmap = np.memmap(config['dat'], dtype='float16', mode='r', shape=(count, FEATURE_DIM))
            self.mmap_list.append(mmap)
            mws = []
            if 'mw' in config and os.path.exists(config['mw']):
                with open(config['mw'], 'r', encoding='utf-8') as f:
                    mws = f.read().splitlines()
            for i in range(count):
                global_idx = total_count + i
                self.formula_to_indices[form[i]].append(global_idx)
                if mws:
                    mw_val = int(float(mws[i]))
                    self.mw_to_indices[mw_val].append(global_idx)
            self.smiles_list.extend(smi)
            self.formulas_list.extend(form)
            if mws: self.mw_list.extend(mws)
            total_count += count
            self.cumulative_sizes.append(total_count)

        for k in self.formula_to_indices:
            self.formula_to_indices[k] = np.array(self.formula_to_indices[k], dtype=np.int32)
        for k in self.mw_to_indices:
            self.mw_to_indices[k] = np.array(self.mw_to_indices[k], dtype=np.int32)

        self.total_count = total_count
        print(f"Library Loaded. Total Size: {self.total_count:,}, Unique Formulas: {len(self.formula_to_indices)}")

    def get_features_chunk(self, start_idx, end_idx):
        chunks = []
        for i in range(len(self.mmap_list)):
            part_start = self.cumulative_sizes[i]
            part_end = self.cumulative_sizes[i + 1]
            overlap_start = max(start_idx, part_start)
            overlap_end = min(end_idx, part_end)
            if overlap_start < overlap_end:
                local_start = overlap_start - part_start
                local_end = overlap_end - part_start
                chunks.append(self.mmap_list[i][local_start:local_end])
        return np.concatenate(chunks, axis=0) if chunks else np.array([])

    def get_features_by_indices(self, global_indices):
        feats = np.zeros((len(global_indices), FEATURE_DIM), dtype='float16')
        for i, idx in enumerate(global_indices):
            part_idx = 0
            while part_idx < len(self.cumulative_sizes) - 1 and idx >= self.cumulative_sizes[part_idx + 1]:
                part_idx += 1
            local_idx = idx - self.cumulative_sizes[part_idx]
            feats[i] = self.mmap_list[part_idx][local_idx]
        return feats

def unified_retrieval_100M(lib_manager, ir_feature, mw=None, formula=None, top_k=100, search_range=None):
    q_feat = ir_feature.to(device).to(torch.float32)
    if q_feat.dim() == 1: q_feat = q_feat.unsqueeze(0)
    q_feat = F.normalize(q_feat, p=2, dim=1)

    candidate_indices = None
    mode = "IR-Only"

    if formula and str(formula).strip():
        target_f = str(formula).strip()
        candidate_indices = lib_manager.formula_to_indices.get(target_f, np.array([], dtype=np.int32))
        mode = "Formula-Filtered"
    elif mw is not None and str(mw).strip() != "":
        try:
            target_mw = int(float(mw))
            candidate_indices = lib_manager.mw_to_indices.get(target_mw, np.array([], dtype=np.int32))
            mode = "MW-Filtered"
        except:
            pass

    print(f"Running search in [{mode}] mode...")

    if mode in ["Formula-Filtered", "MW-Filtered"]:
        if len(candidate_indices) == 0:
            print(f"Warning: No candidates found for {mode} search.")
            return []

        subset_feats = torch.from_numpy(lib_manager.get_features_by_indices(candidate_indices).astype(np.float32)).to(
            device)
        subset_feats = F.normalize(subset_feats, p=2, dim=1)
        sims = torch.matmul(q_feat, subset_feats.t()).squeeze(0)

        actual_k = min(top_k, len(sims))
        topk_scores, topk_rel_indices = torch.topk(sims, k=actual_k)

        final_indices = candidate_indices[topk_rel_indices.cpu().numpy()]
        final_scores = topk_scores.cpu().numpy()

    else:
        total_lib_size = lib_manager.total_count
        if search_range is not None:
            total_lib_size = min(total_lib_size, int(search_range))
            print(f"Restricting IR-only search to first {total_lib_size:,} samples.")

        chunk_size = 500000
        global_best_scores = torch.full((1, top_k), -float('inf'), device=device)
        global_best_indices = torch.zeros((1, top_k), dtype=torch.long, device=device)

        for k in tqdm(range(0, total_lib_size, chunk_size), desc=f"Streaming {mode}"):
            end_k = min(k + chunk_size, total_lib_size)
            lib_chunk = torch.from_numpy(lib_manager.get_features_chunk(k, end_k).astype(np.float32)).to(device)
            lib_chunk = F.normalize(lib_chunk, p=2, dim=1)
            chunk_sims = torch.matmul(q_feat, lib_chunk.t())

            combined_scores = torch.cat([global_best_scores, chunk_sims], dim=1)
            chunk_indices = torch.arange(k, end_k, device=device).unsqueeze(0)
            combined_indices = torch.cat([global_best_indices, chunk_indices], dim=1)

            global_best_scores, rel_idx = torch.topk(combined_scores, k=top_k, dim=1)
            global_best_indices = torch.gather(combined_indices, 1, rel_idx)

        final_indices = global_best_indices.squeeze(0).cpu().numpy()
        final_scores = global_best_scores.squeeze(0).cpu().numpy()

    results = []
    for rank in range(len(final_indices)):
        idx = int(final_indices[rank])
        results.append({
            "rank": rank + 1,
            "smiles": lib_manager.smiles_list[idx],
            "formula": lib_manager.formulas_list[idx],
            "similarity": float(final_scores[rank])
        })
    return results

def load_confidence_mappings(path):
    mappings = {}
    print("Loading Calibration Data...")
    for i in range(1, 11):
        file_path = os.path.join(path, f'top{i}_calib_data.pt')
        if os.path.exists(file_path):
            data = torch.load(file_path, map_location='cpu')
            prob_true, prob_pred = calibration_curve(data['flags'], data['scores'], n_bins=10, strategy='quantile')
            mappings[i] = (prob_pred, prob_true)
        else:
            print(f"Warning: Recall@{i} mapping missing in {path}.")
    return mappings

def calculate_confidence(score, recall_k):
    if recall_k not in CONFIDENCE_MAPPINGS:
        return 0.0
    prob_pred, prob_true = CONFIDENCE_MAPPINGS[recall_k]
    min_score, min_conf = prob_pred[0], prob_true[0]
    if score < min_score:
        conf = score * (min_conf / min_score) if min_score > 0 else 0.0
    else:
        conf = np.interp(score, prob_pred, prob_true)
    return max(0.0, float(conf))
