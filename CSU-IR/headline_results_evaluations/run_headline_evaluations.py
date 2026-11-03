# -*- coding: utf-8 -*-
"""
================================================================================
Unified Inference & Evaluation Entrypoint for Headline Experimental Results
Supported Tasks:
  1. retrieval_100m     - 100-Million-Scale Retrieval (Public data & weights)
  2. functional_groups  - Functional Group Detection (Public data & weights)
  3. ood                - Out-of-Distribution (OOD) Generalization Evaluation (User data required)
  4. ps                 - NPS Identification (User data required)
  5. all                - Run all tasks sequentially
================================================================================
"""

import os
import sys
import time
import json
import glob
import hashlib
import warnings
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast
import importlib

# RDKit Modules
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors, AllChem, Fragments
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup project root path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from model.IR_encoder import IRModel
from model.SMILES_encoder import SmilesModel

# ==============================================================================
# 0. Global Utilities & Model Definitions
# ==============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def contains_C_and_H(smiles_norm):
    """Check if molecule contains both Carbon and Hydrogen (filter inorganic)."""
    if not smiles_norm:
        return False, "N/A"
    try:
        mol = Chem.MolFromSmiles(smiles_norm)
        if mol is None:
            return False, "Invalid Mol"
        formula = rdMolDescriptors.CalcMolFormula(mol)
        has_c = any(atom.GetSymbol() == 'C' for atom in mol.GetAtoms())
        has_h = 'H' in formula
        return (has_c and has_h), formula
    except Exception:
        return False, "Error"

def get_tanimoto_similarity(smi1, smi2):
    if not smi1 or not smi2: return 0.0
    try:
        mol1, mol2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)
        if mol1 is None or mol2 is None: return 0.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0

def is_scaffold_match(smi1, smi2):
    if not smi1 or not smi2: return False
    try:
        mol1, mol2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)
        if mol1 is None or mol2 is None: return False
        scaff1 = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol1), canonical=True)
        scaff2 = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol2), canonical=True)
        return scaff1 == scaff2
    except Exception:
        return False

def get_all_fragments(smiles: str):
    present_groups = set()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return present_groups
        for func_name in dir(Fragments):
            if func_name.startswith('fr_'):
                func = getattr(Fragments, func_name)
                if func(mol) > 0:
                    present_groups.add(func_name)
    except:
        pass
    return present_groups

# ==============================================================================
# Task 1: 100-Million-Scale Compound Retrieval
# ==============================================================================

def formula_to_hash_id(formula_str):
    return int(hashlib.md5(formula_str.encode('utf-8')).hexdigest()[:15], 16)

def load_chunk_meta_fast(tsv_path):
    cache_mw = tsv_path.replace(".tsv", "_mw.npy")
    cache_hash = tsv_path.replace(".tsv", "_hash.npy")
    if os.path.exists(cache_mw) and os.path.exists(cache_hash):
        return np.load(cache_mw), np.load(cache_hash)
    df = pd.read_csv(tsv_path, sep='\t', header=None, usecols=[2, 3], names=['mw', 'form'], dtype={'mw': np.float32, 'form': str})
    df['mw'] = df['mw'].fillna(0.0)
    df['form'] = df['form'].fillna("N/A")
    bg_mws = df['mw'].values.astype(np.float32)
    bg_formulas_hash = np.array([formula_to_hash_id(f) for f in df['form']], dtype=np.int64)
    np.save(cache_mw, bg_mws)
    np.save(cache_hash, bg_formulas_hash)
    return bg_mws, bg_formulas_hash

def get_chunk_sizes(database_dir, prefix="meta_chunk_", num_chunks=54):
    sizes = []
    for i in range(num_chunks):
        tsv_path = os.path.join(database_dir, f"{prefix}{i}.tsv")
        if not os.path.exists(tsv_path): break
        cache_mw = tsv_path.replace(".tsv", "_mw.npy")
        if os.path.exists(cache_mw):
            sizes.append(len(np.load(cache_mw)))
        else:
            with open(tsv_path, 'rb') as f:
                sizes.append(sum(1 for _ in f))
    return sizes

def build_search_chunks_spec(scale_size, orig_chunk_sizes, db_dir):
    search_chunks = []
    if scale_size == 0: return search_chunks
    np.random.seed(42)
    shuffled_indices = list(np.random.permutation(len(orig_chunk_sizes)))
    current_sum = 0
    for idx in shuffled_indices:
        size = orig_chunk_sizes[idx]
        if current_sum + size < scale_size:
            search_chunks.append({
                "dat_path": os.path.join(db_dir, f"features_chunk_{idx}.dat"),
                "tsv_path": os.path.join(db_dir, f"meta_chunk_{idx}.tsv"),
                "limit": None, "size": size
            })
            current_sum += size
        else:
            needed = scale_size - current_sum
            search_chunks.append({
                "dat_path": os.path.join(db_dir, f"features_chunk_{idx}.dat"),
                "tsv_path": os.path.join(db_dir, f"meta_chunk_{idx}.tsv"),
                "limit": needed, "size": needed
            })
            current_sum += needed
            break
    return search_chunks

def update_running_topk(sims, mask, global_offset, running_vals, running_idxs):
    current_sims = sims.clone() if mask is None else sims.clone().masked_fill_(~mask, -float('inf'))
    k = min(100, current_sims.shape[1])
    vals, idxs = torch.topk(current_sims, k=k, dim=1)
    global_idxs = idxs + global_offset
    comb_vals = torch.cat([running_vals, vals], dim=1)
    comb_idxs = torch.cat([running_idxs, global_idxs], dim=1)
    new_vals, topk_idx = torch.topk(comb_vals, k=100, dim=1)
    new_idxs = torch.gather(comb_idxs, dim=1, index=topk_idx)
    running_vals.copy_(new_vals)
    running_idxs.copy_(new_idxs)

def search_worker_process(gpu_id, query_slice_indices, test_ir_feats, test_smiles_feats, 
                          test_mws, test_formulas_hash, search_chunks, mw_tolerance, batch_size, return_dict):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    q_ir = test_ir_feats[query_slice_indices].to(device).half()
    q_mws = torch.tensor(test_mws, dtype=torch.float32)[query_slice_indices].to(device)
    q_form_hash = torch.tensor(test_formulas_hash, dtype=torch.long)[query_slice_indices].to(device)
    n_queries = len(query_slice_indices)

    running_vals_pure = torch.full((n_queries, 100), -float('inf'), device=device, dtype=torch.float16)
    running_idxs_pure = torch.full((n_queries, 100), -1, dtype=torch.long, device=device)
    running_vals_mw = torch.full((n_queries, 100), -float('inf'), device=device, dtype=torch.float16)
    running_idxs_mw = torch.full((n_queries, 100), -1, dtype=torch.long, device=device)
    running_vals_form = torch.full((n_queries, 100), -float('inf'), device=device, dtype=torch.float16)
    running_idxs_form = torch.full((n_queries, 100), -1, dtype=torch.long, device=device)

    global_offset = 0
    for chunk_spec in search_chunks:
        limit = chunk_spec["limit"]
        bg_features = np.fromfile(chunk_spec["dat_path"], dtype=np.float16).reshape(-1, 1024)
        if limit is not None: bg_features = bg_features[:limit]
        bg_mws, bg_formulas_hash = load_chunk_meta_fast(chunk_spec["tsv_path"])
        if limit is not None:
            bg_mws, bg_formulas_hash = bg_mws[:limit], bg_formulas_hash[:limit]

        bg_features_gpu = torch.from_numpy(bg_features).to(device)
        bg_mws_gpu = torch.tensor(bg_mws, dtype=torch.float32, device=device)
        bg_form_gpu = torch.tensor(bg_formulas_hash, dtype=torch.long, device=device)
        num_elements = len(bg_features)

        for q_start in range(0, n_queries, batch_size):
            q_end = min(q_start + batch_size, n_queries)
            batch_ir = q_ir[q_start:q_end]
            sims = torch.matmul(batch_ir, bg_features_gpu.t())

            update_running_topk(sims, None, global_offset, running_vals_pure[q_start:q_end], running_idxs_pure[q_start:q_end])
            mw_mask = torch.abs(q_mws[q_start:q_end].unsqueeze(1) - bg_mws_gpu.unsqueeze(0)) <= mw_tolerance
            update_running_topk(sims, mw_mask, global_offset, running_vals_mw[q_start:q_end], running_idxs_mw[q_start:q_end])
            form_mask = (q_form_hash[q_start:q_end].unsqueeze(1) == bg_form_gpu.unsqueeze(0))
            update_running_topk(sims, form_mask, global_offset, running_vals_form[q_start:q_end], running_idxs_form[q_start:q_end])

        global_offset += num_elements
        del bg_features_gpu, bg_mws_gpu, bg_form_gpu
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Append Target (One-by-One Evaluation Setup)
    test_smiles_gpu = test_smiles_feats[query_slice_indices].to(device).half()
    for q_start in range(0, n_queries, batch_size):
        q_end = min(q_start + batch_size, n_queries)
        sims_self = torch.sum(q_ir[q_start:q_end] * test_smiles_gpu[q_start:q_end], dim=-1, keepdim=True)
        batch_global_indices = torch.tensor(global_offset + query_slice_indices[q_start:q_end], device=device, dtype=torch.long).unsqueeze(1)
        self_mask = torch.ones_like(sims_self, dtype=torch.bool, device=device)

        update_running_topk(sims_self, None, batch_global_indices, running_vals_pure[q_start:q_end], running_idxs_pure[q_start:q_end])
        update_running_topk(sims_self, self_mask, batch_global_indices, running_vals_mw[q_start:q_end], running_idxs_mw[q_start:q_end])
        update_running_topk(sims_self, self_mask, batch_global_indices, running_vals_form[q_start:q_end], running_idxs_form[q_start:q_end])

    return_dict[gpu_id] = {
        "pure_idxs": running_idxs_pure.cpu().numpy(),
        "mw_idxs": running_idxs_mw.cpu().numpy(),
        "form_idxs": running_idxs_form.cpu().numpy()
    }

def run_retrieval_100m(args):
    print("\n" + "=" * 80)
    print(" [Task 1] 100-million-scale Compound Retrieval Evaluation")
    print("=" * 80)

    if not os.path.exists(args.test_descr) or not os.path.exists(args.test_ir):
        print(f"[Error] Required dataset files not found:\n  {args.test_descr}\n  {args.test_ir}")
        return

    test_cids, test_smiles, test_mws, test_formulas, test_formulas_hash = [], [], [], [], []
    with open(args.test_descr, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split('\t')
            cid, smiles = (parts[0], parts[1]) if len(parts) >= 2 else (f"test_{idx}", parts[0])
            norm_sm = normalize_smiles(smiles)
            try:
                mol = Chem.MolFromSmiles(norm_sm)
                mw = rdMolDescriptors.CalcExactMolWt(mol) if mol else 0.0
                formula = rdMolDescriptors.CalcMolFormula(mol) if mol else "N/A"
            except:
                mw, formula = 0.0, "N/A"
            test_cids.append(cid)
            test_smiles.append(norm_sm)
            test_mws.append(mw)
            test_formulas.append(formula)
            test_formulas_hash.append(formula_to_hash_id(formula))

    df_test = pd.DataFrame({'cid': test_cids, 'smiles': test_smiles, 'mw': test_mws, 'form': test_formulas, 'hash': test_formulas_hash, 'idx': range(len(test_cids))})
    df_test = df_test.drop_duplicates(subset=['smiles'], keep='first')
    df_test = df_test[df_test['smiles'].apply(lambda s: contains_C_and_H(s)[0])]
    kept_indices = df_test['idx'].values

    test_cids = df_test['cid'].tolist()
    test_smiles = df_test['smiles'].tolist()
    test_mws = df_test['mw'].tolist()
    test_formulas_hash = df_test['hash'].tolist()
    n_queries = len(test_smiles)

    test_ir_raw = torch.load(args.test_ir, map_location='cpu')[kept_indices]

    device = torch.device(f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    ir_model = IRModel().to(device)
    ir_model.load_weights(args.ir_weights)
    ir_model.eval()

    with torch.no_grad():
        test_ir_feats = torch.cat([ir_model(test_ir_raw[i:i+512].to(device)).cpu() for i in range(0, len(test_ir_raw), 512)], dim=0)
    test_ir_feats = F.normalize(test_ir_feats, p=2, dim=-1)

    smiles_model = SmilesModel(
        roberta_model_path=None, roberta_tokenizer_path=args.tokenizer_path,
        smiles_maxlen=300, max_position_embeddings=505, vocab_size=181, feature_dim=768
    ).to(device)
    smiles_model.load_state_dict(torch.load(args.smiles_weights, map_location=device))
    smiles_model.eval()

    tokenizer = smiles_model.smiles_tokenizer
    encoded_sm_list = []
    with torch.no_grad():
        for i in range(0, n_queries, 256):
            enc = tokenizer.batch_encode_plus(test_smiles[i:i+256], max_length=300, padding='max_length', truncation=True, return_tensors='pt')
            feats = smiles_model.encode((enc['input_ids'].to(device), enc['attention_mask'].to(device)), enc['attention_mask'].sum(dim=1).to(device))
            encoded_sm_list.append(feats.cpu())
    test_smiles_feats = F.normalize(torch.cat(encoded_sm_list, dim=0), p=2, dim=-1)

    orig_chunk_sizes = get_chunk_sizes(args.db_dir, prefix="meta_chunk_", num_chunks=54)

    scales = {"10k": 10000, "100k": 100000, "1 Million": 1000000, "10 Million": 10000000, "100 Million": 100000000}

    k_list = [1, 5, 10, 50, 100]

    for scale_name, scale_size in scales.items():
        print(f"\n>>>> Searching against Library Scale: {scale_size} <<<<")
        search_chunks = build_search_chunks_spec(scale_size, orig_chunk_sizes, args.db_dir)
        actual_bg_size = sum(c["size"] for c in search_chunks)

        manager = mp.Manager()
        return_dict = manager.dict()
        indices_split = np.array_split(np.arange(n_queries), len(args.gpu_ids))

        processes = []
        for rank, gpu_id in enumerate(args.gpu_ids):
            p = mp.Process(target=search_worker_process, args=(gpu_id, indices_split[rank], test_ir_feats, test_smiles_feats, test_mws, test_formulas_hash, search_chunks, 0.5, 500, return_dict))
            p.start()
            processes.append(p)
        for p in processes: p.join()

        all_pure_idxs = np.zeros((n_queries, 100), dtype=np.int64)
        all_mw_idxs = np.zeros((n_queries, 100), dtype=np.int64)
        all_form_idxs = np.zeros((n_queries, 100), dtype=np.int64)

        for rank, gpu_id in enumerate(args.gpu_ids):
            all_pure_idxs[indices_split[rank]] = return_dict[gpu_id]["pure_idxs"]
            all_mw_idxs[indices_split[rank]] = return_dict[gpu_id]["mw_idxs"]
            all_form_idxs[indices_split[rank]] = return_dict[gpu_id]["form_idxs"]

        unique_idxs_needed = set()
        for arr in [all_pure_idxs, all_mw_idxs, all_form_idxs]:
            unique_idxs_needed.update(arr.flatten())
        unique_idxs_needed.discard(-1)

        resolved_metadata = {}
        offsets = [0]
        for c in search_chunks:
            offsets.append(offsets[-1] + c["size"])

        chunk_to_local_lines = {idx: [] for idx in range(len(search_chunks))}
        test_indices = []

        for g_idx in unique_idxs_needed:
            if g_idx >= actual_bg_size:
                test_indices.append(g_idx)
            else:
                chunk_idx = np.searchsorted(offsets, g_idx, side='right') - 1
                local_line = g_idx - offsets[chunk_idx]
                chunk_to_local_lines[chunk_idx].append((local_line, g_idx))

        for chunk_idx, lines_info in chunk_to_local_lines.items():
            if not lines_info: continue
            lines_info.sort(key=lambda x: x[0])
            local_lines = [x[0] for x in lines_info]
            global_indices = [x[1] for x in lines_info]

            tsv_path = search_chunks[chunk_idx]["tsv_path"]
            df_chunk = pd.read_csv(tsv_path, sep='\t', header=None, usecols=[1], names=['smiles'])
            selected_smiles = df_chunk.iloc[local_lines]['smiles'].values
            for g_idx, sm in zip(global_indices, selected_smiles):
                resolved_metadata[g_idx] = normalize_smiles(str(sm))

        for g_idx in test_indices:
            resolved_metadata[g_idx] = test_smiles[g_idx - actual_bg_size]

        def get_rank_by_smiles(pred_top100_idxs, target_sm):
            for rank, g_idx in enumerate(pred_top100_idxs):
                if g_idx == -1: continue
                if resolved_metadata.get(g_idx) == target_sm:
                    return rank
            return -1

        recalls_pure = {k: 0 for k in k_list}
        recalls_mw = {k: 0 for k in k_list}
        recalls_form = {k: 0 for k in k_list}

        for q_idx in range(n_queries):
            target_sm = test_smiles[q_idx]
            r_pure = get_rank_by_smiles(all_pure_idxs[q_idx], target_sm)
            r_mw = get_rank_by_smiles(all_mw_idxs[q_idx], target_sm)
            r_form = get_rank_by_smiles(all_form_idxs[q_idx], target_sm)

            for k in k_list:
                if r_pure != -1 and r_pure < k: recalls_pure[k] += 1
                if r_mw != -1 and r_mw < k: recalls_mw[k] += 1
                if r_form != -1 and r_form < k: recalls_form[k] += 1

        print("-" * 88)
        print(f"| {'Scale: ' + scale_name:<20} | {'Recall@1':<10} | {'Recall@5':<10} | {'Recall@10':<10} | {'Recall@50':<10} | {'Recall@100':<10} |")
        print("-" * 88)
        for name, r_dict in [("IR only", recalls_pure), ("MW filtered (±0.5 Da)", recalls_mw), ("Formula filtered", recalls_form)]:
            print(f"| {name:<20} | {r_dict[1]/n_queries*100:>8.2f}% | {r_dict[5]/n_queries*100:>8.2f}% | {r_dict[10]/n_queries*100:>8.2f}% | {r_dict[50]/n_queries*100:>8.2f}% | {r_dict[100]/n_queries*100:>8.2f}% |")
        print("-" * 88)

# ==============================================================================
# Task 2: Functional Group Detection
# ==============================================================================

def run_functional_groups(args):
    print("\n" + "=" * 80)
    print(" [Task 2] Functional Group Detection")
    print("=" * 80)

    device = torch.device(f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.test_descr) or not os.path.exists(args.test_ir):
        print(f"[Error] Required test files missing:\n  {args.test_descr}\n  {args.test_ir}")
        return

    # 1. Load Data
    with open(args.test_descr, 'r') as f:
        test_smiles_list = [line.strip() for line in f if line.strip()]
    test_ir_tensor = torch.load(args.test_ir)

    # 2. Load Models
    ir_model_arch = IRModel().to(device)
    ir_model_arch.load_weights(args.ir_weights)
    ir_model_arch.eval()

    smiles_model_arch = SmilesModel(
        roberta_model_path=None,
        roberta_tokenizer_path=args.tokenizer_path,
        smiles_maxlen=300,
        max_position_embeddings=505,
        vocab_size=181,
        feature_dim=768
    ).to(device)
    smiles_model_arch.load_state_dict(torch.load(args.smiles_weights, map_location=device))
    smiles_model_arch.eval()

    # Helper functions to mimic original ModelInference class logic
    def smiles_encode(smiles_list):
        with torch.no_grad():
            tokenizer = smiles_model_arch.smiles_tokenizer
            # Original logic: encode_plus one by one then cat
            encoded = [tokenizer.encode_plus(s, max_length=300, padding='max_length',
                                             truncation=True, return_tensors='pt') for s in smiles_list]
            input_ids = torch.cat([i['input_ids'] for i in encoded], dim=0).to(device)
            attention_mask = torch.cat([i['attention_mask'] for i in encoded], dim=0).to(device)
            lengths = attention_mask.sum(dim=1)
            features = smiles_model_arch.encode((input_ids, attention_mask), lengths)
            return F.normalize(features, p=2, dim=1)

    def ir_encode(ir_tensor_single):
        with torch.no_grad():
            # Original logic: view(1, -1)
            ir_input = ir_tensor_single.view(1, -1).to(device)
            features = ir_model_arch(ir_input)
            return F.normalize(features, p=2, dim=1)

    # 3. Pre-encode Library Features
    print("--> Encoding library features...")
    lib_features = []
    batch_size_enc = 256  # Original code used 256 for library encoding
    for i in range(0, len(test_smiles_list), batch_size_enc):
        batch = test_smiles_list[i:i + batch_size_enc]
        lib_features.append(smiles_encode(batch).cpu())
    lib_features = torch.cat(lib_features, 0).to(device)
    
    # Normalize library SMILES for comparison
    lib_smiles_norm = [normalize_smiles(s) for s in test_smiles_list]

    # 4. Define Target Functional Groups
    TARGET_GROUPS = {
        'fr_nitrile', 'fr_C_O', 'fr_ester', 'fr_imide', 'fr_nitro_arom_nonortho', 
        'fr_C_O_noCOO', 'fr_amide', 'fr_ketone_Topliss', 'fr_ketone', 'fr_nitro', 
        'fr_nitro_arom', 'fr_Al_OH', 'fr_Al_COO', 'fr_COO', 'fr_Al_OH_noTert', 
        'fr_COO2', 'fr_sulfonamd', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_aldehyde', 
        'fr_priamide', 'fr_Imine', 'fr_Ar_OH', 'fr_urea', 'fr_phos_acid', 'fr_Ar_COO', 
        'fr_benzene', 'fr_ether', 'fr_alkyl_halide', 'fr_NH1', 'fr_methoxy', 'fr_halogen', 
        'fr_piperdine', 'fr_aniline', 'fr_thiazole', 'fr_furan', 'fr_Nhpyrrole', 'fr_Ar_NH', 
        'fr_hdrzone', 'fr_NH2', 'fr_morpholine', 'fr_sulfide', 'fr_pyridine', 'fr_imidazole', 
        'fr_hdrzine', 'fr_SH', 'fr_C_S', 'fr_piperzine'
    }
    
    group_stats = {g: {'base': 0, 'detected': 0} for g in TARGET_GROUPS}

    # 5. Create Custom Dataset and DataLoader (Fixing the AttributeError)
    class IRSmilesDataset(torch.utils.data.Dataset):
        def __init__(self, ir_spectra, smiles):
            self.ir_spectra = ir_spectra
            self.smiles = smiles

        def __len__(self):
            return len(self.smiles)

        def __getitem__(self, idx):
            return self.ir_spectra[idx], self.smiles[idx]

    dataset = IRSmilesDataset(test_ir_tensor, test_smiles_list)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

    print("--> Analyzing Top-1 functional group consistency...")
    
    # 6. Inference Loop
    for ir_batch, smiles_batch in tqdm(loader, desc="Analyzing"):
        for i in range(ir_batch.shape[0]):
            # Normalize true smiles
            true_s = normalize_smiles(smiles_batch[i])
            
            # Encode IR
            ir_feat = ir_encode(ir_batch[i])
            
            # Calculate Similarity
            sim_scores = (ir_feat @ lib_features.t()).squeeze(0)
            _, top_indices = sim_scores.topk(10)
            top_indices = top_indices.cpu().numpy()
            
            # Get Top-1 SMILES from library
            top1_idx = top_indices[0]
            top1_s = lib_smiles_norm[top1_idx]
            
            # Get Fragments
            true_groups = get_all_fragments(true_s)
            top1_groups = get_all_fragments(top1_s)
            
            # Update Stats for Target Groups Only
            for g in TARGET_GROUPS:
                if g in true_groups:
                    group_stats[g]['base'] += 1
                    if g in top1_groups:
                        group_stats[g]['detected'] += 1

    # 7. Print Results
    print(f"\n{'RDKit Group Name':<30} | {'Base':<8} | {'Detected':<10} | {'Detection Rate':<12}")
    print("-" * 65)
    
    valid_groups = [(name, stats) for name, stats in group_stats.items() if stats['base'] > 0]
    valid_groups.sort(key=lambda x: x[1]['detected'] / x[1]['base'], reverse=True)
    
    total_base = 0
    total_detected = 0
    
    for name, stat in valid_groups:
        acc = stat['detected'] / stat['base'] if stat['base'] > 0 else 0
        total_base += stat['base']
        total_detected += stat['detected']
        print(f"{name:<30} | {stat['base']:<8} | {stat['detected']:<10} | {acc:.4f}")
        
    print("-" * 65)
    overall_rate = total_detected / total_base if total_base > 0 else 0
    print(f"Total Target Groups Analyzed: {len(valid_groups)}")
    print(f"Overall Detection Rate (Total Detected / Total Base): {overall_rate:.4f}")
    
    
# ==============================================================================
# Task 3: Out-of-Distribution (OOD) Evaluation
# ==============================================================================

def run_ood(args):
    print("\n" + "=" * 80)
    print(" [Task 3] Out-of-Distribution (OOD) Generalization Evaluation")
    print("=" * 80)
    
    if not os.path.exists(args.ood_dir):
        print(f"[Error] OOD directory not found: {args.ood_dir}")
        return

    device = torch.device(f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Models
    print("--> Loading models for OOD evaluation...")
    ir_model = IRModel().to(device)
    ir_model.load_weights(args.ir_weights)
    ir_model.eval()

    smiles_model = SmilesModel(
        roberta_model_path=None, 
        roberta_tokenizer_path=args.tokenizer_path,
        smiles_maxlen=300, 
        max_position_embeddings=505, 
        vocab_size=181, 
        feature_dim=768
    ).to(device)
    smiles_model.load_state_dict(torch.load(args.smiles_weights, map_location=device))
    smiles_model.eval()
    tokenizer = smiles_model.smiles_tokenizer
    print("--> Models loaded successfully.")

    # 2. Define Subsets
    subsets = ["id_sim_gt0.7", "ood_near_sim0.5-0.7", "ood_moderate_sim0.3-0.5", "ood_far_sim_lt0.3"]
    
    results_summary = {}
    
    print("  -> Loading raw data")
    
    for name in subsets:
        sm_p = os.path.join(args.ood_dir, f"{name}_thermo_smiles.txt")
        ir_p = os.path.join(args.ood_dir, f"{name}_thermo_data.pt")
        
        if not (os.path.exists(sm_p) and os.path.exists(ir_p)):
            print(f"[Skip] Files missing for subset: {name}")
            continue
        
        # --- Load Data ---
        with open(sm_p, 'r') as f: 
            sm_list_raw = [s.strip() for s in f.read().splitlines() if s.strip()]
        
        ir_t = torch.load(ir_p)
        print(f"  -> Raw samples from {name} loaded")
        
        # Normalize and Filter
        valid_pairs = []
        for i, sm in enumerate(sm_list_raw):
            norm_sm = normalize_smiles(sm)
            if norm_sm is not None:
                valid_pairs.append((norm_sm, ir_t[i]))
        
        if not valid_pairs:
            print("  -> [Warning] No valid molecules remaining after normalization.")
            continue
            
        query_smiles = [p[0] for p in valid_pairs]
        query_ir_tensor = torch.stack([p[1] for p in valid_pairs])
        n_queries = len(query_smiles)


        # --- Encode Gallery (SMILES) ---
        gallery_feats_list = []
        batch_size_enc = 256
        with torch.no_grad():
            for i in range(0, n_queries, batch_size_enc):
                batch_smi = query_smiles[i:i+batch_size_enc]
                enc = tokenizer.batch_encode_plus(batch_smi, max_length=300, padding='max_length', truncation=True, return_tensors='pt')
                feats = smiles_model.encode((enc['input_ids'].to(device), enc['attention_mask'].to(device)), enc['attention_mask'].sum(dim=1).to(device))
                gallery_feats_list.append(feats.cpu())
        
        gallery_feats = F.normalize(torch.cat(gallery_feats_list, dim=0), p=2, dim=-1).to(device)
        print(f"  -> SMILES features encoded")
        
        # --- Encode Query (IR) ---
        ir_feats_list = []
        with torch.no_grad():
            for i in range(0, n_queries, 512):
                batch_ir = query_ir_tensor[i:i+512].to(device)
                feats = ir_model(batch_ir)
                ir_feats_list.append(feats.cpu())
        
        q_ir_feats = F.normalize(torch.cat(ir_feats_list, dim=0), p=2, dim=-1).to(device)
        print(f"  -> IR features encoded")
        
        # --- Calculate Similarity & Accuracy ---
        sim_matrix = torch.matmul(q_ir_feats, gallery_feats.t())
        
        print("  -> Retrieving Top-1 matches")
        top1_vals, top1_indices = torch.max(sim_matrix, dim=1)
        top1_indices_cpu = top1_indices.cpu().numpy()
        
        correct = 0
        for i in range(n_queries):
            retrieved_smiles = query_smiles[top1_indices_cpu[i]]
            # Check if the retrieved SMILES matches the ground truth SMILES for this IR
            if retrieved_smiles == query_smiles[i]:
                correct += 1
                
        acc = correct / n_queries if n_queries > 0 else 0
        results_summary[name] = acc

    if results_summary:
        print("\n" + "-" * 50)
        print(f"{'Subset':<30} | {'Top-1 Accuracy':<15}")
        print("-" * 50)
        for name in subsets:
            if name in results_summary:
                print(f"{name:<30} | {results_summary[name]:>13.4f}")
        print("-" * 50)
    else:
        print("[Warning] No OOD subsets were successfully evaluated.")
# ==============================================================================
# Task 4: NPS Across 2 Libraries Evaluation
# ==============================================================================

def run_ps(args):
    print("\n" + "=" * 80)
    print(" [Task 4] NPS Identification")
    print("=" * 80)
    
    # 1. Check Files
    if not os.path.exists(args.ps_query_smiles) or not os.path.exists(args.ps_query_ir):
        print(f"[Notice] NPS query files not found at:\n  {args.ps_query_smiles}\n  {args.ps_query_ir}")
        return
        
    existing_lib_path = args.ps_exsiting_library
    derivative_lib_path = args.ps_derivative_library
    
    if not os.path.exists(existing_lib_path):
        print(f"[Warning] Existing library not found: {existing_lib_path}")
        existing_lib_path = None
        
    if not os.path.exists(derivative_lib_path):
        print(f"[Warning] Derivative library not found: {derivative_lib_path}")
        derivative_lib_path = None

    if existing_lib_path is None and derivative_lib_path is None:
        print("[Error] No libraries found for evaluation.")
        return

    device = torch.device(f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Models
    print("--> Loading PS-finetuned models...")
    ir_model = IRModel().to(device)
    ir_model.load_weights(args.ir_weights)
    ir_model.eval()

    smiles_model = SmilesModel(
        roberta_model_path=None, 
        roberta_tokenizer_path=args.tokenizer_path,
        smiles_maxlen=300, 
        max_position_embeddings=505, 
        vocab_size=181, 
        feature_dim=768
    ).to(device)
    smiles_model.load_state_dict(torch.load(args.smiles_weights, map_location=device))
    smiles_model.eval()
    tokenizer = smiles_model.smiles_tokenizer

    # 3. Load & Preprocess Query Data (NPS Test Set)
    print("--> Loading NPS Query Data...")
    with open(args.ps_query_smiles, 'r') as f:
        query_smiles_raw = [line.strip() for line in f if line.strip()]
    
    query_ir_tensor = torch.load(args.ps_query_ir)
    
    # Normalize and Filter Queries
    valid_query_pairs = []
    for i, sm in enumerate(query_smiles_raw):
        norm_sm = normalize_smiles(sm)
        if norm_sm is not None and contains_C_and_H(norm_sm)[0]:
            valid_query_pairs.append((norm_sm, query_ir_tensor[i]))
            
    if not valid_query_pairs:
        print("[Error] No valid organic molecules found in query set.")
        return

    query_smiles_norm = [p[0] for p in valid_query_pairs]
    query_ir_valid = torch.stack([p[1] for p in valid_query_pairs])
    n_queries = len(query_smiles_norm)

    # 4. Encode Query IR Features
    print("--> Encoding Query IR features...")
    q_ir_feats_list = []
    with torch.no_grad():
        for i in range(0, n_queries, 512):
            batch_ir = query_ir_valid[i:i+512].to(device)
            feats = ir_model(batch_ir)
            q_ir_feats_list.append(feats.cpu())
    q_ir_feats = F.normalize(torch.cat(q_ir_feats_list, dim=0), p=2, dim=-1).to(device)

    # Helper function to process a library
    def evaluate_library(lib_name, lib_path):
        if not lib_path or not os.path.exists(lib_path):
            return None
            
        print(f"\n---- Evaluating against Library: [{lib_name}] ----")
        
        # Load Library SMILES
        with open(lib_path, 'r') as f:
            lib_smiles_raw = [line.strip() for line in f if line.strip()]
        
        # Normalize Library SMILES
        lib_smiles_norm = []
        for sm in lib_smiles_raw:
            norm_sm = normalize_smiles(sm)
            if norm_sm is not None:
                lib_smiles_norm.append(norm_sm)
                
        if not lib_smiles_norm:
            return None
            
        print(f"--> Library Size (valid): {len(lib_smiles_norm)}")

        # Encode Library SMILES Features
        lib_feats_list = []
        batch_size_enc = 256
        with torch.no_grad():
            for i in range(0, len(lib_smiles_norm), batch_size_enc):
                batch_smi = lib_smiles_norm[i:i+batch_size_enc]
                enc = tokenizer.batch_encode_plus(batch_smi, max_length=300, padding='max_length', truncation=True, return_tensors='pt')
                feats = smiles_model.encode((enc['input_ids'].to(device), enc['attention_mask'].to(device)), enc['attention_mask'].sum(dim=1).to(device))
                lib_feats_list.append(feats.cpu())
        
        lib_feats = F.normalize(torch.cat(lib_feats_list, dim=0), p=2, dim=-1).to(device)
        
        # Calculate Similarity Matrix
        # q_ir_feats: [N_queries, D], lib_feats: [N_lib, D]
        # Result: [N_queries, N_lib]
        print("--> Calculating similarity matrix...")
        sim_matrix = torch.matmul(q_ir_feats, lib_feats.t())
        
        # Define K values for evaluation
        k_values = [1, 5, 10]
        max_k = max(k_values)
        
        # Ensure we don't ask for more than library size
        actual_max_k = min(max_k, lib_feats.shape[0])
        
        topk_vals, topk_indices = torch.topk(sim_matrix, k=actual_max_k, dim=1)
        topk_indices_cpu = topk_indices.cpu().numpy()
        
        # Evaluate Recall@K
        recalls = {k: 0 for k in k_values}
        
        for q_idx in range(n_queries):
            target_sm = query_smiles_norm[q_idx]
            retrieved_indices = topk_indices_cpu[q_idx]
            
            # Check if target is in top K
            found_at_rank = -1
            for rank, lib_idx in enumerate(retrieved_indices):
                if lib_smiles_norm[lib_idx] == target_sm:
                    found_at_rank = rank
                    break
            
            if found_at_rank != -1:
                for k in k_values:
                    if found_at_rank < k:
                        recalls[k] += 1
                        
        # Print Results
        print("-" * 65)
        print(f"| {'Library':<20} | {'Recall@1':<10} | {'Recall@5':<10} | {'Recall@10':<10} |")
        print("-" * 65)
        
        row_str = f"| {lib_name:<20} |"
        for k in k_values:
            rate = recalls[k] / n_queries * 100
            row_str += f" {rate:>8.2f}% |"
        print(row_str)
        print("-" * 65)
        
        return recalls

    # 5. Run Evaluation for Each Library
    results = {}
    if existing_lib_path:
        results['Existing'] = evaluate_library('Existing_PS', existing_lib_path)
        
    if derivative_lib_path:
        results['Derivative'] = evaluate_library('Derivative_PS', derivative_lib_path)

    print("\n--> NPS cross-library evaluation completed.")

# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate headline results reported in manuscript.")
    parser.add_argument('--task', type=str, default='all', choices=['retrieval_100m', 'functional_groups', 'ood', 'ps', 'all'],
                        help="Select specific headline experiment to reproduce.")
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0], help="GPU device IDs to use.")
    
    # Paths for Task 1 & 2
    parser.add_argument('--test_descr', type=str, 
                        default=os.path.join(PROJECT_ROOT, "data", "Multi-staged_training_data", "Experimental_Benching_data", "EB", "data_with_NIST_IR", "test_smiles.txt"))
    parser.add_argument('--test_ir', type=str, 
                        default=os.path.join(PROJECT_ROOT, "data", "Multi-staged_training_data", "Experimental_Benching_data", "EB", "data_with_NIST_IR", "test_ir.pt"))
    
    parser.add_argument('--ir_weights_EB', type=str, 
                        default=os.path.join(PROJECT_ROOT, "check_points", "Multi-stage_training_Stage_III_EXP", "MD_DFT_EXP", "best_ir_model_EB.pth"))
    parser.add_argument('--smiles_weights_EB', type=str, 
                        default=os.path.join(PROJECT_ROOT, "check_points", "Multi-stage_training_Stage_III_EXP", "MD_DFT_EXP", "best_smiles_model_EB.pth"))
    
    parser.add_argument('--ir_weights_EB_TF', type=str, 
                        default=os.path.join(PROJECT_ROOT, "check_points", "Multi-stage_training_Stage_III_EXP", "MD_DFT_EXP", "best_ir_model_EB_TF.pth"))
    parser.add_argument('--smiles_weights_EB_TF', type=str, 
                        default=os.path.join(PROJECT_ROOT, "check_points", "Multi-stage_training_Stage_III_EXP", "MD_DFT_EXP", "best_smiles_model_EB_TF.pth"))

    parser.add_argument('--tokenizer_path', type=str, 
                        default=os.path.join(PROJECT_ROOT, "model", "tokenizer-smiles-roberta-1e_new"))
    parser.add_argument('--db_dir', type=str, 
                        default=os.path.join(PROJECT_ROOT, "data", "100-Million-library-Retrieval"))
    parser.add_argument('--eval_full_scale', action='store_true', help="Evaluate all 5 scales (1w to 100M); default evaluates 3 scales (1w to 100w).")

    # Paths for Task 3 (OOD) & Task 4 (PS)
    parser.add_argument('--ir_weights_PS', type=str, 
                        default=os.path.join(PROJECT_ROOT, "check_points", "PS_finetuned", "best_ir_model.pth"))
    parser.add_argument('--smiles_weights_PS', type=str, 
                        default=os.path.join(PROJECT_ROOT, "check_points", "PS_finetuned", "best_smiles_model.pth"))
    
    parser.add_argument('--ood_dir', type=str, 
                        default=os.path.join(PROJECT_ROOT, "data", "OOD_data"))
    parser.add_argument('--ps_query_smiles', type=str, 
                        default=os.path.join(PROJECT_ROOT, "data", "test_data", "NPS", "filtered_final_NPS_smiles.txt"))
    parser.add_argument('--ps_query_ir', type=str, 
                        default=os.path.join(PROJECT_ROOT, "data", "test_data", "NPS", "filtered_final_NPS_ir.pt"))
    parser.add_argument('--ps_exsiting_library', type=str, 
                        default=os.path.join(PROJECT_ROOT, "data", "processed_library", "PS", "smiles_Existing_PS.txt"))
    parser.add_argument('--ps_derivative_library', type=str, 
                        default=os.path.join(PROJECT_ROOT, "data", "PS Derivative Library", "smiles_Derivative_PS.txt"))

    args = parser.parse_args()
    set_seed(42)


    if args.task in ['retrieval_100m', 'all']:
        args.ir_weights = args.ir_weights_EB_TF
        args.smiles_weights = args.smiles_weights_EB_TF
        run_retrieval_100m(args)

    if args.task in ['functional_groups', 'all']:
        args.ir_weights = args.ir_weights_EB
        args.smiles_weights = args.smiles_weights_EB
        run_functional_groups(args)

    if args.task in ['ood', 'all']:
        args.ir_weights = args.ir_weights_EB
        args.smiles_weights = args.smiles_weights_EB
        run_ood(args)

    if args.task in ['ps', 'all']:
        args.ir_weights = args.ir_weights_PS
        args.smiles_weights = args.smiles_weights_PS
        run_ps(args)

if __name__ == '__main__':
    mp.freeze_support()
    main()
