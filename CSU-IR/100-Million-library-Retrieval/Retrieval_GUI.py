'''
This script includes three functions for retrieving the 100-Million-Scale libray:
---infrared spectroscopy only;
---infrared spectroscopy plus molecular weight;
---and infrared spectroscopy plus molecular formula.
Before using this script, please download the 00-Million-Scale data to the corresponding folder.
place the spectra (csv or jdx format), molecular weight, or molecular formula of the unknown substance to be searched in the specified location.
'''

import os
import sys
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
import gradio as gr


from Retrieval_functions import load_MW_Formula
from Retrieval_functions import get_final_query_metadata
from Retrieval_functions import UnifiedCombinedLibrary
from Retrieval_functions import unified_retrieval_100M

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

from test_and_infer.test_and_infer_functions import ModelInference
from test_and_infer.test_and_infer_functions import get_feature_from_smiles

# 100_Million_Scale_libray
FEATURE_DIM = 1024
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_ir(ir_file_path, spectrum_type, model_infer_instance):
    print(f"Processing file: {ir_file_path}")
    if ir_file_path.lower().endswith('.csv'):
        df = pd.read_csv(ir_file_path)
        wavenumbers = df.iloc[:, 0].values
        transmittances = df.iloc[:, 1].values
    elif ir_file_path.lower().endswith('.jdx'):
        data = jcamp.jcamp_readfile(ir_file_path)
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
        ir_feature = model_infer_instance.ir_encode(ir_spectra_tensor)
    return ir_feature

class IR_Retrieval_Engine_100:
    def __init__(self):
        print(f"Initializing Retrieval Engine on {device}...")
        self.tokenizer_path = os.path.join(PROJECT_ROOT,'model',"tokenizer-smiles-roberta-1e_new")
        self.pretrain_smiles_path = os.path.join(PROJECT_ROOT, "check_points", "best_smiles_model_0.9230379746835443.pth")
        self.pretrain_ir_path = os.path.join(PROJECT_ROOT, "check_points", "best_ir_model_0.9230379746835443.pth")
        
        ir_model = IRModel().to(device)
        sm_model = SmilesModel(
            roberta_model_path=None,
            roberta_tokenizer_path=self.tokenizer_path,
            smiles_maxlen=300,
            max_position_embeddings=505,
            vocab_size=181,
            feature_dim=768,
        ).to(device)

        self.model_infer = ModelInference(
            sm_model, ir_model,
            pretrain_model_path_sm=self.pretrain_smiles_path,
            pretrain_model_path_ir=self.pretrain_ir_path,
            device=device
        )
        self.lib_configs = [
            {
                "name": f"Sub-Library {i+1}",
                "dat": os.path.join(PROJECT_ROOT,'data','100-Million-library-Retrieval', f'global_pool_features_100M_1024dim_fp16_part_{"I" if i<2 else ("II" if i<4 else "III")}_sub{(i%2)+1}.dat'),
                "formulas": os.path.join(PROJECT_ROOT,'data','100-Million-library-Retrieval', f'global_pool_features_100M_1024dim_fp16_part_{"I" if i<2 else ("II" if i<4 else "III")}_formulas_part_{"I" if i<2 else ("II" if i<4 else "III")}_sub{(i%2)+1}.txt'),
                "smiles": os.path.join(PROJECT_ROOT,'data','100-Million-library-Retrieval', f'global_pool_features_100M_1024dim_fp16_part_{"I" if i<2 else ("II" if i<4 else "III")}_smiles_part_{"I" if i<2 else ("II" if i<4 else "III")}_sub{(i%2)+1}.txt')
            } for i in range(6)
        ]

        self.lib_manager = UnifiedCombinedLibrary(self.lib_configs)

    def search(self, ir_file, mw, formula, spectrum_type, top_k):
        try:
            ir_feature = process_ir(ir_file.name, spectrum_type, self.model_infer) 
            results = unified_retrieval_100M(
                self.lib_manager, 
                ir_feature=ir_feature, 
                mw=mw if mw else None, 
                formula=formula if formula else None, 
                top_k=top_k
            )
            
            if not results:
                return "No candidates found.", None
            df = pd.DataFrame(results)
            df = df[['rank', 'similarity', 'formula', 'smiles']]
            df['similarity'] = df['similarity'].map(lambda x: f"{x:.4f}")
            return f"Search completed! Found top-{len(df)} candidates.", df
        
        except Exception as e:
            return f"Error: {str(e)}", None


engine = IR_Retrieval_Engine_100M()

def ui_wrapper(ir_file, mw, formula, spectrum_type, top_k):
    if ir_file is None:
        return "Please upload an IR spectrum file (.jdx or .csv)", None
    return engine.search(ir_file, mw, formula, spectrum_type, int(top_k))

with gr.Blocks(title="100M-Scale IR Spectroscopy Retrieval System") as demo:
    gr.Markdown("# 🧪 100M-Scale IR Spectroscopy Retrieval")
    gr.Markdown("Upload an IR spectrum and optional molecular information to search the 100-million compound library.")
    
    with gr.Row():
        with gr.Column(scale=1):
            ir_file = gr.File(label="Upload IR Spectrum (.jdx, .csv)")
            spec_type = gr.Radio(
                choices=["absorbance spectrum", "transmittance spectrum"], 
                value="absorbance spectrum", 
                label="Spectrum Type"
            )
            mw_input = gr.Textbox(label="Molecular Weight (Optional)", placeholder="e.g. 180")
            formula_input = gr.Textbox(label="Molecular Formula (Optional)", placeholder="e.g. C9H8O4")
            top_k_slider = gr.Slider(minimum=10, maximum=100, value=50, step=10, label="Top-K Results")
            search_btn = gr.Button("🚀 Start Retrieval", variant="primary")
            
        with gr.Column(scale=2):
            status_output = gr.Textbox(label="Status")
            result_table = gr.DataFrame(label="Candidate Hits (Top-K)")

    search_btn.click(
        fn=ui_wrapper,
        inputs=[ir_file, mw_input, formula_input, spec_type, top_k_slider],
        outputs=[status_output, result_table]
    )

    gr.Markdown("### 📝 Notes:\n- The system uses Formula > MW > IR Only logic.\n- Ensure the 200GB .dat files are in the specified local directory.")

if __name__ == "__main__":
    demo.launch(share=False) 





