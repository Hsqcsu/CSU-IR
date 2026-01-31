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

# --- 1. Set up the project root directory, which is the base for all relative paths ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"Project Root Path: {PROJECT_ROOT}")
sys.path.insert(0, PROJECT_ROOT)

# --- 2. Import custom modules from within the project ---
# If there is a red underline below, don't worry, it will not affect the code running
from model.IR_encoder import IRModel
from model.SMILES_encoder import SmilesModel


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

unknown_data_to_test = ["ir_path": os.path.join(PROJECT_ROOT,'data','unknown_data','100-Million-library-Retrieval','example_unknown_ir.jdx'),
             "MW_path": os.path.join(PROJECT_ROOT,'data','unknown_data','100-Million-library-Retrieval','example_unknown_MW.jdx'),
             "Formula_path": os.path.join(PROJECT_ROOT,'data','unknown_data','100-Million-library-Retrieval','example_unknown_formula.jdx'),
]




