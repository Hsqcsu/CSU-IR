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
    
def load_MW_Formula(path):
    with open(path, 'r') as f:
        MW_or_Formula = f.read().splitlines()
    return MW_or_Formula





