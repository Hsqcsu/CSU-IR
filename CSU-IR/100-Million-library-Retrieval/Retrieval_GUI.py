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
import numpy as np
import pandas as pd
import jcamp
import gradio as gr
from sklearn.calibration import calibration_curve


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

EXAMPLE_DIR = os.path.join(PROJECT_ROOT, 'data', "example_library_and_ir_for_user_dinfined")
Confidence_DIR = os.path.join(PROJECT_ROOT, 'data', "Confidence_curve_data")

from Retrieval_functions import load_MW_Formula
from Retrieval_functions import get_final_query_metadata
from Retrieval_functions import UnifiedCombinedLibrary
from Retrieval_functions import unified_retrieval_100M
from Retrieval_functions import load_confidence_mappings
from Retrieval_functions import calculate_confidence

from model.IR_encoder import IRModel
from model.SMILES_encoder import SmilesModel
from data_process.ir_process import (
    preprocess_absorbances_spectra_higer_500, preprocess_absorbances_spectra_lower_500,
    preprocess_transmittances_spectra_higer_500, preprocess_transmittances_spectra_lower_500
)
from test_and_infer.test_and_infer_functions import ModelInference, get_feature_from_smiles


CONFIDENCE_MAPPINGS = load_confidence_mappings(Confidence_DIR)

FEATURE_DIM = 1024
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CSS = """
* { font-family: 'Times New Roman', Times, serif !important; }
#header h1 { color: #00bfa5; text-align: center; }
.gradio-button { background: #00bfa5 !important; color: white !important; }
#white-text-example .gradio-label { color: #000000 !important; }
#white-text-example button { color: #000000 !important; }
.white-box { background-color: #ffffff !important; border-radius: 12px; }
.format-note { background-color: #ffffff; border: 1px solid #dddddd; color: #1a1a1a; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
"""

def process_ir(ir_file_path, spectrum_type, model_infer_instance):
    if ir_file_path.lower().endswith('.csv') and spectrum_type == "transmittance spectrum":
        df = pd.read_csv(ir_file_path)
        wavenumbers, transmittances = df.iloc[:, 0].values, df.iloc[:, 1].values
        ir_data = preprocess_transmittances_spectra_higer_500(wavenumbers, transmittances) if wavenumbers[0] > 500 else preprocess_transmittances_spectra_lower_500(wavenumbers, transmittances)
    elif ir_file_path.lower().endswith('.jdx') and spectrum_type == "absorbance spectrum":
        data = jcamp.jcamp_readfile(ir_file_path)
        wavenumbers, transmittances = np.array(data['x'], dtype=float), np.array(data['y'], dtype=float)
        ir_data = preprocess_absorbances_spectra_higer_500(wavenumbers, transmittances) if wavenumbers[0] > 500 else preprocess_absorbances_spectra_lower_500(wavenumbers, transmittances)
    else:
        raise ValueError("Unsupported file format or incorrect spectrum type.")

    ir_spectra_tensor = torch.tensor(ir_data, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        ir_feature = model_infer_instance.ir_encode(ir_spectra_tensor)
    return ir_feature

class IR_Retrieval_Engine_100M:
    def __init__(self):
        self.tokenizer_path = os.path.join(PROJECT_ROOT, 'model', "tokenizer-smiles-roberta-1e_new")
        self.pretrain_smiles_path = os.path.join(PROJECT_ROOT, "check_points", "Multi-stage_training_Stage_III_EXP",
                                                 "best_smiles_model_0.9230379746835443.pth")
        self.pretrain_ir_path = os.path.join(PROJECT_ROOT, "check_points", "Multi-stage_training_Stage_III_EXP",
                                             "best_ir_model_0.9230379746835443.pth")

        ir_model = IRModel().to(device)
        sm_model = SmilesModel(roberta_model_path=None, roberta_tokenizer_path=self.tokenizer_path, smiles_maxlen=300,
                               max_position_embeddings=505, vocab_size=181, feature_dim=768).to(device)
        self.model_infer = ModelInference(sm_model, ir_model, pretrain_model_path_sm=self.pretrain_smiles_path,
                                          pretrain_model_path_ir=self.pretrain_ir_path, device=device)

        
        self.lib_configs = []
        parts = ["I", "II", "III"]
        base_dir = os.path.join(PROJECT_ROOT, 'data', '100-Million-library-Retrieval')
        
        for p_name in parts:
            part_folder = os.path.join(base_dir, f'Part_{p_name}')
            
            for sub_i in range(1, 19): 
                config = {
                    "name": f"Part_{p_name}-Sub{sub_i}",
                    "dat": os.path.join(part_folder, f'global_pool_features_100M_1024dim_fp16_part_{p_name}_sub{sub_i}.dat'),
                    "formulas": os.path.join(part_folder, f'global_pool_features_100M_1024dim_fp16_part_{p_name}_formulas_part_{p_name}_sub{sub_i}.txt'),
                    "smiles": os.path.join(part_folder, f'global_pool_features_100M_1024dim_fp16_part_{p_name}_smiles_part_{p_name}_sub{sub_i}.txt'),
                    "mw": os.path.join(part_folder,f'global_pool_features_100M_1024dim_fp16_part_{p_name}_mw_part_{p_name}_sub{sub_i}.txt')
                }
                self.lib_configs.append(config)
    
        self.lib_manager = UnifiedCombinedLibrary(self.lib_configs)

    

    def search(self, ir_file, mw, formula, spectrum_type, top_k, search_range):
        try:
            ir_feature = process_ir(ir_file.name, spectrum_type, self.model_infer)


            limit_map = {"1w": 10000, "10w": 100000, "100w": 1000000, "1000w": 10000000, "Full Library": None}
            search_limit = limit_map.get(search_range, None)


            if (mw and mw.strip()) or (formula and formula.strip()):
                search_limit = None

            results = unified_retrieval_100M(
                self.lib_manager,
                ir_feature=ir_feature,
                mw=mw if mw else None,
                formula=formula if formula else None,
                top_k=top_k,
                search_range=search_limit
            )

            if not results:
                return "No candidates found.", None, gr.update(visible=False)

            df = pd.DataFrame(results)
            recall1_score = float(df.iloc[0]['similarity'])
            best_conf, best_k = -1.0, 1
            for k in range(1, 11):
                current_conf = calculate_confidence(recall1_score, k,CONFIDENCE_MAPPINGS)
                if current_conf > best_conf:
                    best_conf, best_k = current_conf, k

            conf_summary = (
                f'<div style="background-color: #ffffff; color: #1a1a1a; padding: 20px; border-radius: 12px; border: 1px solid #ddd; text-align: center;">'
                f'<h3 style="color: #00bfa5; margin-top: 0; margin-bottom: 10px;">📊 Confidence Analysis</h3>'
                f'<p style="font-size: 0.9em; color: #666; text-align: justify; line-height: 1.5; margin: 10px 0; padding: 0 10px;">'
                f'Users need not worry: during confidence analysis, it is inevitable that a high confidence score may be given '
                f'because the correct substance is not in the database and the model assigns a high score to similar substances. '
                f'Our method has proven that the top-ranked candidates have a high similarity to the correct substance, and the retrieval rate '
                f'of key substructures exceeds 90%. This is also of reference value.'
                f'</p>'
                f'<hr style="border: 0; border-top: 1px solid #eee; margin: 15px 0;">'
                f'<p style="font-size: 1.2em; margin: 10px 0;">There is a '
                f'<span style="color:#008c7a; font-weight:bold; font-size: 1.3em;">{best_conf * 100:.2f}%</span> '
                f'probability that the correct molecule is within '
                f'<span style="font-weight:bold; color:#333;">Recall@1-{best_k}</span>.</p>'
                f'<p style="font-size:0.9em; color:#666;">(Based on Similarity Score: {recall1_score:.4f})</p>'
                f'</div>'
            )
            df = df[['rank', 'similarity', 'formula', 'smiles']]
            df['similarity'] = df['similarity'].map(lambda x: f"{float(x):.4f}")
            return f"Search completed!", df, gr.update(value=conf_summary, visible=True)

        except Exception as e:
            return f"Error: {str(e)}", None, gr.update(visible=False)


engine = IR_Retrieval_Engine_100M()

def handle_range_visibility(mw, formula):
    if (mw and mw.strip()) or (formula and formula.strip()):
        return gr.update(visible=False, value="Full Library")
    return gr.update(visible=True)


with gr.Blocks(title="100-Million-Scale IR Retrieval System") as demo:
    gr.HTML('<div id="header"><h1>100-Million-Scale IR Retrieval</h1></div>')

    gr.Markdown("""
    ### 📝 Notes:
    - For optimal performance, the system selects methods in this order of preference: Formula-based, then MW-based, and finally IR-only.
    - When using only infrared signals, the retrieval scope can be selected. When using the entire 100 million library, the retrieval time can take several minutes.
    - ！！！ Users need not worry: during confidence analysis, it is inevitable that a high confidence score may be given because the correct substance is not in the database and the model assigns a high score to similar substances. Our method has proven that the top-ranked candidates have a high similarity to the correct substance, and the retrieval rate of key substructures exceeds 90%. This is also of reference value.
    - If MW or molecular formula information is added, the entire library will be used for retrieval (with high accuracy, eliminating the need to further narrow down the library).
    - Currently, only **jdx** or **csv** files are supported. Please download the corresponding jdx or csv file from the 'CSU-IR/data/example_library_and_ir_for_user_dinfined' folder to view the data format.
    - Example files are provided below for quick testing.
    - Ensure the 200GB .dat files are in the specified local directory.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            ir_file = gr.File(label="Upload IR Spectrum (.jdx, .csv)")

            gr.Markdown(
                "**🧪 Manual Test Info:**\n- `4-fluoro-ABUTINACA`: MW `369`, Formula `C22H28FN3O` (Absorbance)\n- `2-Butene`: MW `56`, Formula `C4H8` (Transmittance)")

            gr.Examples(
                examples=[[os.path.join(EXAMPLE_DIR, '4-fluoro-ABUTINACA.JDX')],
                          [os.path.join(EXAMPLE_DIR, '2-Butene.CSV')]],
                inputs=[ir_file], label="Step 1: Load Example Spectrum", elem_id="white-text-example"
            )

            spec_type = gr.Radio(choices=["absorbance spectrum", "transmittance spectrum"], value="absorbance spectrum",
                                 label="Spectrum Type")
            mw_input = gr.Textbox(label="Molecular Weight (Optional)", placeholder="e.g. 180")
            formula_input = gr.Textbox(label="Molecular Formula (Optional)", placeholder="e.g. C9H8O4")

            search_range = gr.Dropdown(
                choices=["1w", "10w", "100w", "1000w", "Full Library"],
                value="Full Library",
                label="Search Range (Only for IR-only search)"
            )

            mw_input.change(handle_range_visibility, inputs=[mw_input, formula_input], outputs=[search_range])
            formula_input.change(handle_range_visibility, inputs=[mw_input, formula_input], outputs=[search_range])

            top_k_slider = gr.Slider(minimum=10, maximum=100, value=50, step=10, label="Top-K Results")
            search_btn = gr.Button("🚀 Start Retrieval", variant="primary")

        with gr.Column(scale=2):
            status_output = gr.Textbox(label="Status")
            conf_display = gr.HTML(visible=False, elem_classes="white-box")
            result_table = gr.DataFrame(label="Candidate Hits (Top-K)")

    search_btn.click(
        fn=engine.search,
        inputs=[ir_file, mw_input, formula_input, spec_type, top_k_slider, search_range],
        outputs=[status_output, result_table, conf_display]
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        allowed_paths=[EXAMPLE_DIR, Confidence_DIR]
    )

