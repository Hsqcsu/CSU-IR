'''
This script includes three functions for retrieving the 100-Million-Scale libray:
---infrared spectroscopy only;
---infrared spectroscopy plus molecular weight;
---and infrared spectroscopy plus molecular formula.
Before using this script, please download the 00-Million-Scale data to the corresponding folder.
place the spectra (csv or jdx format), molecular weight, or molecular formula of the unknown substance to be searched in the specified location.
'''

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import json
import gc
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from collections import defaultdict
import pandas as pd
import jcamp

# --- 1. Set up the project root directory, which is the base for all relative paths ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"Project Root Path: {PROJECT_ROOT}")
sys.path.insert(0, PROJECT_ROOT)

# --- 2. Import custom modules from within the project ---
# If there is a red underline below, don't worry, it will not affect the code running
from model.IR_encoder import IRModel
from model.SMILES_encoder import SmilesModel

from data_process.ir_process import preprocess_absorbances_spectra_higer_500
from data_process.ir_process import preprocess_absorbances_spectra_lower_500
from data_process.ir_process import preprocess_transmittances_spectra_higer_500
from data_process.ir_process import preprocess_transmittances_spectra_lower_500

from test_and_infer.infer import ModelInference
from test_and_infer.infer import get_feature_from_smiles

# 100_Million_Scale_libray
FEATURE_DIM = 1024
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

100_Million_Scale_libray = [
    {
        "dat_sub1": os.path.join(PROJECT_ROOT,'data','100-Million-library-Retrieval','global_pool_features_100M_1024dim_fp16_part_I_sub1.dat'),
        "formulas_sub1": os.path.join(PROJECT_ROOT,'data','100-Million-library-Retrieval','global_pool_features_100M_1024dim_fp16_part_I_formulas_part_I_sub1.txt'),
        "smiles_sub1": os.path.join(PROJECT_ROOT,'data','100-Million-library-Retrieval','global_pool_features_100M_1024dim_fp16_part_I_smiles_part_I_sub1.txt')
    },
    {
        "dat_sub2": os.path.join(PROJECT_ROOT,'data','100-Million-library-Retrieval','global_pool_features_100M_1024dim_fp16_part_I_sub2.dat'),
        "formulas_sub2": os.path.join(PROJECT_ROOT,'data','100-Million-library-Retrieval','global_pool_features_100M_1024dim_fp16_part_I_formulas_part_I_sub2.txt'),
        "smiles_sub2": os.path.join(PROJECT_ROOT,'data','100-Million-library-Retrieval','global_pool_features_100M_1024dim_fp16_part_I_smiles_part_I_sub2.txt')
    },
    {
        "dat_sub3": os.path.join(PROJECT_ROOT,'data','100-Million-library-Retrieval','global_pool_features_100M_1024dim_fp16_part_II_sub1.dat'),
        "formulas_sub3": os.path.join(PROJECT_ROOT,'data','100-Million-library-Retrieval','global_pool_features_100M_1024dim_fp16_part_II_formulas_part_II_sub1.txt'),
        "smiles_sub3": os.path.join(PROJECT_ROOT,'data','100-Million-library-Retrieval','global_pool_features_100M_1024dim_fp16_part_II_smiles_part_II_sub1.txt')
    },
    {
        "dat_sub4": os.path.join(PROJECT_ROOT,'data','100-Million-library-Retrieval','global_pool_features_100M_1024dim_fp16_part_II_sub2.dat'),
        "formulas_sub4": os.path.join(PROJECT_ROOT,'data','100-Million-library-Retrieval','global_pool_features_100M_1024dim_fp16_part_II_formulas_part_II_sub2.txt'),
        "smiles_sub4": os.path.join(PROJECT_ROOT,'data','100-Million-library-Retrieval','global_pool_features_100M_1024dim_fp16_part_II_smiles_part_II_sub2.txt')
    },
    {
        "dat_sub5": os.path.join(PROJECT_ROOT,'data','100-Million-library-Retrieval','global_pool_features_100M_1024dim_fp16_part_III_sub1.dat'),
        "formulas_sub5": os.path.join(PROJECT_ROOT,'data','100-Million-library-Retrieval','global_pool_features_100M_1024dim_fp16_part_III_formulas_part_III_sub1.txt'),
        "smiles_sub5": os.path.join(PROJECT_ROOT,'data','100-Million-library-Retrieval','global_pool_features_100M_1024dim_fp16_part_III_smiles_part_III_sub1.txt')
    },
    {
        "dat_sub6": os.path.join(PROJECT_ROOT,'data','100-Million-library-Retrieval','global_pool_features_100M_1024dim_fp16_part_III_sub2.dat'),
        "formulas_sub6": os.path.join(PROJECT_ROOT,'data','100-Million-library-Retrieval','global_pool_features_100M_1024dim_fp16_part_III_formulas_part_III_sub2.txt'),
        "smiles_sub6": os.path.join(PROJECT_ROOT,'data','100-Million-library-Retrieval','global_pool_features_100M_1024dim_fp16_part_III_smiles_part_III_sub2.txt')
    },
    
]

# [User Operation Area]
# If you want to enter the values ​​directly at runtime, please fill in the variables below. If you leave None or "", the text file will be read automatically.

RUNTIME_MW = None       # For example: 180 or "180"
RUNTIME_FORMULA = None  # For example: "C9H8O4"

# unknown_data
unknown_data_to_test = {"ir_path": os.path.join(PROJECT_ROOT, 'data', 'unknown_data', '100-Million-library-Retrieval','example_unknown_ir.jdx'),
"MW_path": os.path.join(PROJECT_ROOT, 'data', 'unknown_data', '100-Million-library-Retrieval','example_unknown_MW.txt'),
"Formula_path": os.path.join(PROJECT_ROOT, 'data', 'unknown_data', '100-Million-library-Retrieval','example_unknown_formula.txt'),}


# model
'''
You need to download the model weight file in hugging_face and save it in the check_points folder.
'''
TOKENIZER_PATH = os.path.join(PROJECT_ROOT,'model',"tokenizer-smiles-roberta-1e_new")
PRETRAIN_SMILES_MODEL_PATH = os.path.join(PROJECT_ROOT, "check_points", "best_smiles_model_0.9230379746835443.pth")
PRETRAIN_IR_MODEL_PATH = os.path.join(PROJECT_ROOT, "check_points", "best_ir_model_0.9230379746835443.pth")

IR_model = IRModel()
SmilesModel = SmilesModel(roberta_model_path=None,
    roberta_tokenizer_path= TOKENIZER_PATH,
    smiles_maxlen=300,
    max_position_embeddings=505,
    vocab_size=181,
    feature_dim=768,
)

IR_model.to(device)
SmilesModel.to(device)

ModelInferenc = ModelInference(
        SmilesModel,
        IR_model,
        pretrain_model_path_sm= PRETRAIN_SMILES_MODEL_PATH,
        pretrain_model_path_ir= PRETRAIN_IR_MODEL_PATH,
        device=None
    )

# process unknown ir
def process_ir(ir_spectra_file, spectrum_type):
    if hasattr(ir_spectra_file, 'name'):
        file_path = ir_spectra_file.name
    else:
        file_path = ir_spectra_file
    print(f"Processing file: {file_path}")
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
        wavenumbers = df.iloc[:, 0].values
        transmittances = df.iloc[:, 1].values
    elif file_path.lower().endswith('.jdx'):
        data = jcamp.jcamp_readfile(file_path)
        wavenumbers = np.array(data['x'], dtype=float)
        transmittances = np.array(data['y'], dtype=float)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or JDX file.")
    if spectrum_type == "absorbance spectrum":
        if wavenumbers[0] > 500:
            ir_data = preprocess_absorbances_spectra_higer_500(wavenumbers, transmittances)
        else:
            ir_data = preprocess_absorbances_spectra_lower_500(wavenumbers, transmittances)
    else:
        if wavenumbers[0] > 500:
            ir_data = preprocess_transmittances_spectra_higer_500(wavenumbers, transmittances)
        else:
            ir_data = preprocess_transmittances_spectra_lower_500(wavenumbers, transmittances)
    ir_spectra_tensor = torch.tensor(ir_data, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        ir_feature = ModelInferenc.ir_encode(ir_spectra_tensor)
    return ir_feature

unknown_ir_feature = process_ir(unknown_data_to_test["ir_path"],spectrum_type='absorbance spectrum')
    



# Auxiliary functions
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

def unified_retrieval_100M(lib_manager, ir_feature, mw=None, formula=None, top_k=100):
    """
    Unified Retrieval Interface
    Logical Priority: Formula > MW > IR Only
    """
    q_feat = ir_feature.to(device).to(torch.float32)
    if q_feat.dim() == 1: q_feat = q_feat.unsqueeze(0)
    q_feat = F.normalize(q_feat, p=2, dim=1)
    
    candidate_indices = None
    mode = "IR-Only"

    if formula and str(formula).strip():
        target_f = str(formula).strip()
        candidate_indices = lib_manager.formula_to_indices.get(target_f, np.array([], dtype=np.int32))
        mode = "Formula-Filtered"
    elif mw is not None:
        try:
            target_mw = int(float(mw))
            candidate_indices = lib_manager.mw_to_indices.get(target_mw, np.array([], dtype=np.int32))
            mode = "MW-Filtered"
        except: pass

    print(f"Running search in [{mode}] mode...")
    
    if mode in ["Formula-Filtered", "MW-Filtered"]:
        if len(candidate_indices) == 0:
            print(f"Warning: No candidates found for {mode} search.")
            return []
        
        subset_feats = torch.from_numpy(lib_manager.get_features_by_indices(candidate_indices).astype(np.float32)).to(device)
        subset_feats = F.normalize(subset_feats, p=2, dim=1)
        sims = torch.matmul(q_feat, subset_feats.t()).squeeze(0)
        
        actual_k = min(top_k, len(sims))
        topk_scores, topk_rel_indices = torch.topk(sims, k=actual_k)
        
        final_indices = candidate_indices[topk_rel_indices.cpu().numpy()]
        final_scores = topk_scores.cpu().numpy()

    else:
        total_lib_size = lib_manager.total_count
        chunk_size = 500000
        global_best_scores = torch.full((1, top_k), -float('inf'), device=device)
        global_best_indices = torch.zeros((1, top_k), dtype=torch.long, device=device)
        for k in tqdm(range(0, total_lib_size, chunk_size), desc="Streaming 100M Library"):
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


# Retrieval
lib_manager = UnifiedCombinedLibrary(unified_configs)

u_ir = unknown_ir_feature 
final_mw = get_final_query_metadata(RUNTIME_MW, unknown_data_to_test["MW_path"])
final_formula = get_final_query_metadata(RUNTIME_FORMULA, unknown_data_to_test["Formula_path"])

top100_results = unified_retrieval_100M(
    lib_manager, 
    ir_feature=unknown_ir_feature, 
    mw=final_mw, 
    formula=final_formula, 
    top_k=100
)


print(f"Retrieved {len(top100_results)} candidates.")
for cand in top100_results[:5]:
    print(f"Rank {cand['rank']}: {cand['similarity']:.4f} | {cand['formula']} | {cand['smiles']}")

with open('Retrieval_Results.json', 'w') as f:
    json.dump(top100_results, f)




