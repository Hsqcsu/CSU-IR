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

# unknown_data
unknown_data_to_test = ["ir_path": os.path.join(PROJECT_ROOT,'data','unknown_data','100-Million-library-Retrieval','example_unknown_ir.jdx'),
             "MW_path": os.path.join(PROJECT_ROOT,'data','unknown_data','100-Million-library-Retrieval','example_unknown_MW.jdx'),
             "Formula_path": os.path.join(PROJECT_ROOT,'data','unknown_data','100-Million-library-Retrieval','example_unknown_formula.jdx'),
]

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

unknown_ir_feature = process_ir(unknown_data_to_test[0],spectrum_type='absorbance spectrum')
    
def load_MW_Formula(path):
    with open(path, 'r') as f:
        MW_or_Formula = f.read().splitlines()
    return MW_or_Formula

unknown_MW = load_MW_Formula(unknown_data_to_test[1])
unknown_Formula = load_MW_Formula(unknown_data_to_test[2])

# IR Only functions
class CombinedLibrary_IR_only:
    def __init__(self, library_configs):
        self.mmap_list = []
        self.smiles_list = []
        self.formulas_list = []
        self.cumulative_sizes = [0]

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
            self.smiles_list.extend(smi)
            self.formulas_list.extend(form)
            total_count += count
            self.cumulative_sizes.append(total_count)
        self.total_count = total_count
        print(f"Total Combined Library Size: {self.total_count:,}")

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


def IR_only_retrieval_unknown_ir_100M(lib_manager, unknown_ir_feature, top_k=100):
    q_feats = unknown_ir_feature.to(device).to(torch.float32)
    if q_feats.dim() == 1:
        q_feats = q_feats.unsqueeze(0)
    q_feats = F.normalize(q_feats, p=2, dim=1)
    
    num_queries = q_feats.shape[0]
    total_lib_size = lib_manager.total_count
    library_chunk_size = 500000 

    global_best_scores = torch.full((num_queries, top_k), -float('inf'), device=device)
    global_best_indices = torch.zeros((num_queries, top_k), dtype=torch.long, device=device)

    for k in tqdm(range(0, total_lib_size, library_chunk_size), desc="Streaming Search"):
        end_k = min(k + library_chunk_size, total_lib_size)

        lib_chunk_np = lib_manager.get_features_chunk(k, end_k)
        lib_chunk = torch.from_numpy(lib_chunk_np.astype(np.float32)).to(device)
        lib_chunk = F.normalize(lib_chunk, p=2, dim=1)

        chunk_sims = torch.matmul(q_feats, lib_chunk.t())


        # [num_queries, top_k + ChunkSize]
        combined_scores = torch.cat([global_best_scores, chunk_sims], dim=1)
        

        chunk_indices = torch.arange(k, end_k, device=device).expand(num_queries, -1)
        combined_indices = torch.cat([global_best_indices, chunk_indices], dim=1)


        global_best_scores, topk_rel_indices = torch.topk(combined_scores, k=top_k, dim=1)
        global_best_indices = torch.gather(combined_indices, 1, topk_rel_indices)

        del lib_chunk, chunk_sims, combined_scores, combined_indices
        # torch.cuda.empty_cache() # 如果显存非常紧张可以开启

    final_scores = global_best_scores.cpu().numpy()
    final_indices = global_best_indices.cpu().numpy()
    
    all_search_results = []
    
    for i in range(num_queries):
        query_results = []
        for rank in range(top_k):
            idx = int(final_indices[i][rank])
            score = float(final_scores[i][rank])
            
            query_results.append({
                "rank": rank + 1,
                "smiles": lib_manager.smiles_list[idx],
                "formula": lib_manager.formulas_list[idx],
                "similarity": score
            })
        all_search_results.append(query_results)

    return all_search_results


# IR Only Retrieval

lib_manager_IR_only = CombinedLibrary(lib_configs)
top100_candidates = IR_only_retrieval_unknown_ir_100M(lib_manager_IR_only, unknown_ir_feature, top_k=100)

for cand in top100_candidates[0][:5]:
    print(f"Rank {cand['rank']}: Score: {cand['similarity']:.4f}, SMILES: {cand['smiles']}, Formula: {cand['formula']}")

with open('IR_Only_top100_results.json', 'w') as f:
     json.dump(top100_candidates, f)

# IR+MW functions

class CombinedLibrary_IR_MW:
    def __init__(self, library_configs):
        self.mmap_list = []
        self.smiles_list = []
        self.mw_list = []
        self.cumulative_sizes = [0]
        self.mw_to_indices = defaultdict(list)

        total_count = 0
        for config in library_configs:
            print(f"Loading metadata for {config['name']}...")
            with open(config['smiles'], 'r', encoding='utf-8') as f:
                smi = f.read().splitlines()
            with open(config['mw'], 'r', encoding='utf-8') as f:
                mws = f.read().splitlines()
            count = len(smi)
            mmap = np.memmap(config['dat'], dtype='float16', mode='r', shape=(count, FEATURE_DIM))
            for i, mw_val in enumerate(mws):
                mw_int = int(mw_val)
                self.mw_to_indices[mw_int].append(total_count + i)
            self.mmap_list.append(mmap)
            self.smiles_list.extend(smi)
            self.mw_list.extend(mws)
            total_count += count
            self.cumulative_sizes.append(total_count)

        self.total_count = total_count
        for mw_val in self.mw_to_indices:
            self.mw_to_indices[mw_val] = np.array(self.mw_to_indices[mw_val], dtype=np.int32)

        print(f"Index built. Total unique MWs: {len(self.mw_to_indices)}")

    def get_features_by_indices(self, global_indices):
        feats = np.zeros((len(global_indices), FEATURE_DIM), dtype='float16')
        for i, idx in enumerate(global_indices):
            part_idx = 0
            while part_idx < len(self.cumulative_sizes) - 1 and idx >= self.cumulative_sizes[part_idx + 1]:
                part_idx += 1
            local_idx = idx - self.cumulative_sizes[part_idx]
            feats[i] = self.mmap_list[part_idx][local_idx]
        return feats



def MW_filtered_retrieval_unknown_ir_100M(lib_manager, unknown_ir_feature, unknown_mw_list, top_k=100):

    q_feats = unknown_ir_feature.to(device).to(torch.float32)
    if q_feats.dim() == 1:
        q_feats = q_feats.unsqueeze(0)
    q_feats = F.normalize(q_feats, p=2, dim=1)

    num_queries = q_feats.shape[0]
    all_search_results = []

    for i in range(num_queries):
        try:
            target_mw = int(float(unknown_mw_list[i]))
        except (ValueError, IndexError):
            print(f"Warning: Invalid MW value at index {i}")
            all_search_results.append([])
            continue
        candidate_global_indices = lib_manager.mw_to_indices.get(target_mw, np.array([], dtype=np.int32))

        if len(candidate_global_indices) == 0:
            print(f"No candidates found for MW: {target_mw}")
            all_search_results.append([])
            continue

        subset_feats_np = lib_manager.get_features_by_indices(candidate_global_indices)
        subset_feats = torch.from_numpy(subset_feats_np.astype(np.float32)).to(device)
        subset_feats = F.normalize(subset_feats, p=2, dim=1)

        # q_feats[i:i+1] -> [1, 1024], subset_feats.t() -> [1024, SubsetCount]
        sims = torch.matmul(q_feats[i:i+1], subset_feats.t()).squeeze(0)

        actual_k = min(top_k, len(sims))
        topk_scores, topk_rel_indices = torch.topk(sims, k=actual_k)

        query_results = []
        topk_scores = topk_scores.cpu().numpy()
        topk_rel_indices = topk_rel_indices.cpu().numpy()

        for rank in range(actual_k):
            global_idx = candidate_global_indices[topk_rel_indices[rank]]
            query_results.append({
                "rank": rank + 1,
                "smiles": lib_manager.smiles_list[global_idx],
                "mw": lib_manager.mw_list[global_idx],
                "similarity": float(topk_scores[rank])
            })
        
        all_search_results.append(query_results)

    return all_search_results

# IR+MW Retrieval

lib_manager_mw = CombinedLibrary_IR_MW(lib_configs_mw)
top100_mw_filtered = MW_filtered_retrieval_unknown_ir_100M(
    lib_manager_mw, 
    unknown_ir_feature, 
    unknown_MW, 
    top_k=100
)
if top100_mw_filtered[0]:
    print(f"Top 5 candidates with MW {int(float(unknown_MW[0]))}:")
    for cand in top100_mw_filtered[0][:5]:
        print(f"Rank {cand['rank']}: Score: {cand['similarity']:.4f}, SMILES: {cand['smiles']}")
        
with open('IR_MW_Filtered_top100_results.json', 'w') as f:
     json.dump(top100_mw_filtered, f)

# IR+Formula functions 
class CombinedLibrary_IR_Formula:
    def __init__(self, library_configs):
        self.mmap_list = []
        self.smiles_list = []
        self.formulas_list = []
        self.cumulative_sizes = [0]
        self.formula_to_indices = defaultdict(list)

        total_count = 0
        for config in library_configs:
            print(f"Loading metadata for {config['name']}...")
            with open(config['smiles'], 'r', encoding='utf-8') as f:
                smi = f.read().splitlines()
            with open(config['formulas'], 'r', encoding='utf-8') as f:
                form = f.read().splitlines()

            count = len(smi)
            mmap = np.memmap(config['dat'], dtype='float16', mode='r', shape=(count, FEATURE_DIM))

            for i, f_str in enumerate(form):
                self.formula_to_indices[f_str].append(total_count + i)

            self.mmap_list.append(mmap)
            self.smiles_list.extend(smi)
            self.formulas_list.extend(form)
            total_count += count
            self.cumulative_sizes.append(total_count)

        self.total_count = total_count
        for f_str in self.formula_to_indices:
            self.formula_to_indices[f_str] = np.array(self.formula_to_indices[f_str], dtype=np.int32)

        print(f"Index built. Total unique formulas: {len(self.formula_to_indices)}")

    def get_features_by_indices(self, global_indices):
        feats = np.zeros((len(global_indices), FEATURE_DIM), dtype='float16')
        for i, idx in enumerate(global_indices):
            part_idx = 0
            while part_idx < len(self.cumulative_sizes) - 1 and idx >= self.cumulative_sizes[part_idx + 1]:
                part_idx += 1
            local_idx = idx - self.cumulative_sizes[part_idx]
            feats[i] = self.mmap_list[part_idx][local_idx]
        return feats

def Formula_filtered_retrieval_unknown_ir_100M(lib_manager, unknown_ir_feature, unknown_formula_list, top_k=100):
    q_feats = unknown_ir_feature.to(device).to(torch.float32)
    if q_feats.dim() == 1:
        q_feats = q_feats.unsqueeze(0)
    q_feats = F.normalize(q_feats, p=2, dim=1)

    num_queries = q_feats.shape[0]
    all_search_results = []
    for i in range(num_queries):
        try:
            target_formula = str(unknown_formula_list[i]).strip()
        except (ValueError, IndexError):
            print(f"Warning: Invalid Formula value at index {i}")
            all_search_results.append([])
            continue
        candidate_global_indices = lib_manager.formula_to_indices.get(target_formula, np.array([], dtype=np.int32))

        if len(candidate_global_indices) == 0:
            print(f"No candidates found in library for Formula: {target_formula}")
            all_search_results.append([])
            continue

        subset_feats_np = lib_manager.get_features_by_indices(candidate_global_indices)
        subset_feats = torch.from_numpy(subset_feats_np.astype(np.float32)).to(device)
        subset_feats = F.normalize(subset_feats, p=2, dim=1)

        # q_feats[i:i+1] -> [1, 1024], subset_feats.t() -> [1024, SubsetCount]
        sims = torch.matmul(q_feats[i:i+1], subset_feats.t()).squeeze(0)

        actual_k = min(top_k, len(sims))
        topk_scores, topk_rel_indices = torch.topk(sims, k=actual_k)

        query_results = []
        topk_scores = topk_scores.cpu().numpy()
        topk_rel_indices = topk_rel_indices.cpu().numpy()

        for rank in range(actual_k):
            global_idx = candidate_global_indices[topk_rel_indices[rank]]
            query_results.append({
                "rank": rank + 1,
                "smiles": lib_manager.smiles_list[global_idx],
                "formula": lib_manager.formulas_list[global_idx],
                "similarity": float(topk_scores[rank])
            })
        
        all_search_results.append(query_results)

    return all_search_results

# IR+Formula Retrieval

lib_manager_formula = CombinedLibrary_IR_Formula(lib_configs_formula)

top100_formula_filtered = Formula_filtered_retrieval_unknown_ir_100M(
    lib_manager_formula, 
    unknown_ir_feature, 
    unknown_Formula, 
    top_k=100
)

if top100_formula_filtered[0]:
    print(f"Top 5 candidates for Formula {unknown_Formula[0]}:")
    for cand in top100_formula_filtered[0][:5]:
        print(f"Rank {cand['rank']}: Score: {cand['similarity']:.4f}, SMILES: {cand['smiles']}")

with open('IR_Formula_Filtered_top100_results.json', 'w') as f:
     json.dump(top100_formula_filtered, f)

