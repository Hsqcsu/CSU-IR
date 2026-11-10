'''
This script provides a Web service for retrieving the 100-Million-Scale library:
1. Infrared spectroscopy only;
2. Infrared spectroscopy plus monoisotopic mass;
3. Infrared spectroscopy plus molecular formula.

Spectra (.csv or .jdx format in either absorbance or transmittance) are 
automatically parsed and preprocessed via the unified physical pipeline.
'''

import os
import sys
import torch
import pandas as pd
import gradio as gr

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

EXAMPLE_DIR = os.path.join(PROJECT_ROOT, 'data', "example_library_and_ir_for_user_dinfined")

from Retrieval_functions import (
    process_ir,
    Chunked100MLibraryManager,
    unified_retrieval_100M,
    calculate_calibrated_confidence
)

from model.IR_encoder import IRModel
from model.SMILES_encoder import SmilesModel
from test_and_infer.test_and_infer_functions import ModelInference

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== 1. Retrieval Engine Interface ====================
class IR_Retrieval_Engine_100M:
    def __init__(self):
        self.tokenizer_path = os.path.join(PROJECT_ROOT, 'model', "tokenizer-smiles-roberta-1e_new")
        self.pretrain_smiles_path = os.path.join(PROJECT_ROOT, "check_points", "Multi-stage_training_Stage_III_EXP", "MD_DFT_EXP", "best_smiles_model_EB_TF.pth")
        self.pretrain_ir_path = os.path.join(PROJECT_ROOT, "check_points", "Multi-stage_training_Stage_III_EXP", "MD_DFT_EXP", "best_ir_model_EB_TF.pth")
        self.db_dir = os.path.join(PROJECT_ROOT, 'data', '100-Million-library-Retrieval')

        self.ir_model = IRModel().to(device)
        self.sm_model = SmilesModel(
            roberta_model_path=None,
            roberta_tokenizer_path=self.tokenizer_path,
            smiles_maxlen=300,
            max_position_embeddings=505,
            vocab_size=181,
            feature_dim=768
        ).to(device)
        self.model_infer = ModelInference(
            self.sm_model,
            self.ir_model,
            pretrain_model_path_sm=self.pretrain_smiles_path,
            pretrain_model_path_ir=self.pretrain_ir_path,
            device=device
        )
        self.lib_manager = Chunked100MLibraryManager(self.db_dir, num_chunks=54)
        self.total_db_size = self.lib_manager.total_size

    def search(self, ir_file, mw, formula, top_k, full_lib_cb, dynamic_scale):
        try:
            if ir_file is None:
                return "Please upload an IR spectrum file.", None, gr.update(visible=False)

            top_k = int(top_k)
            ir_feature = process_ir(ir_file.name, self.model_infer)

            has_mw = bool(mw and str(mw).strip())
            has_form = bool(formula and str(formula).strip())

            if has_mw or has_form or full_lib_cb:
                search_limit = None
            else:
                search_limit = int(dynamic_scale)

            results = unified_retrieval_100M(
                q_ir=ir_feature,
                lib_manager=self.lib_manager,
                mw=mw,
                formula=formula,
                top_k=top_k,
                search_range=search_limit,
                mw_tolerance=0.5
            )

            if not results:
                return "No candidates found matching the criteria.", None, gr.update(visible=False)

            df_res = pd.DataFrame(results)
            top1_score = float(df_res.iloc[0]['similarity'])

            top1_calibrated_conf = calculate_calibrated_confidence(
                top1_score,
                self.sm_model,
                scale=0.2940,
                bias=-0.8541
            )

            df_res['Calibrated Confidence'] = df_res['similarity'].map(
                lambda s: f"{calculate_calibrated_confidence(float(s), self.sm_model, scale=0.2940, bias=-0.8541) * 100:.2f}%"
            )

            statement_text = (
                f'The <span style="font-weight: bold;">Top-1 (Recall@1)</span> candidate of this retrieval has a '
                f'<span style="color: #008c7a; font-weight: bold; font-size: 1.25em;">{top1_calibrated_conf * 100:.2f}%</span> '
                f'probability of being the correct target molecule.'
            )

            note_text = (
                '<b>Note:</b> In an ultra-large database (100-million scale), substantial expansion of the '
                'candidate space may introduce statistical calibration drift. When the exact query molecule '
                'is absent from the library, structurally close analogues may still receive relatively high '
                'confidence scores. We demonstrate that top-ranked candidates maintain high structural '
                'similarity to true targets. Therefore, users are encouraged to evaluate the candidate list '
                'by combining confidence metrics with chemical domain expertise.'
            )

            conf_summary = (
                f'<div style="background-color: #ffffff; color: #1a1a1a; padding: 22px; border-radius: 12px; border: 1px solid #e0e0e0; text-align: left; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">'
                f'<h3 style="color: #00bfa5; margin-top: 0; margin-bottom: 14px; text-align: center; font-size: 1.3em;">📊 Confidence Analysis</h3>'
                f'<div style="background-color: #f8fafc; border-left: 4px solid #00bfa5; padding: 12px 16px; margin-bottom: 14px; border-radius: 4px;">'
                f'<p style="font-size: 1.05em; margin: 0; line-height: 1.6; color: #1e293b;">{statement_text}</p>'
                f'</div>'
                f'<p style="font-size: 0.88em; color: #64748b; line-height: 1.6; margin: 0; text-align: justify;">{note_text}</p>'
                f'<p style="font-size: 0.85em; color: #94a3b8; text-align: right; margin-top: 8px; margin-bottom: 0;">(Top-1 Cosine Similarity: {top1_score:.4f})</p>'
                f'</div>'
            )

            df_res = df_res[['rank', 'similarity', 'Calibrated Confidence', 'formula', 'mw', 'smiles']]
            df_res['similarity'] = df_res['similarity'].map(lambda x: f"{float(x):.4f}")
            df_res['mw'] = df_res['mw'].map(lambda x: f"{float(x):.4f}")
            df_res.columns = ['Rank', 'Cosine Similarity', 'Calibrated Confidence', 'Formula', 'Mol Weight', 'SMILES']

            return "Search completed!", df_res, gr.update(value=conf_summary, visible=True)

        except Exception as e:
            return f"Error: {str(e)}", None, gr.update(visible=False)


engine = IR_Retrieval_Engine_100M()


# ==================== 2. Gradio User Interface ====================
CSS = """
* { font-family: 'Times New Roman', Times, serif !important; }
#header h1 { color: #00bfa5; text-align: center; }
.gradio-button { background: #00bfa5 !important; color: white !important; }
#white-text-example .gradio-label { color: #000000 !important; }
#white-text-example button { color: #000000 !important; }
.white-box { background-color: #ffffff !important; border-radius: 12px; }
.format-note { background-color: #ffffff; border: 1px solid #dddddd; color: #1a1a1a; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
"""

with gr.Blocks(title="100-Million-Scale IR Retrieval System", css=CSS) as demo:
    gr.HTML('<div id="header"><h1>100-Million-Scale IR Retrieval System</h1></div>')

    gr.Markdown("""
    ### 📝 Notes:
    - **Adaptive Spectral Preprocessing**: Both **.jdx** and **.csv** files in either **absorbance** or **transmittance** mode are automatically parsed via unified physics-based interpolation.
    - **Retrieval Priority**: For optimal performance, the search follows: Formula-based filtering > Exact Mass filtering > IR-only search.
    - **Search Time**: Searching across the full 100M library in IR-only mode takes approximately 3 min 30 s (tested on our local setup). Providing molecular weight or chemical formula enables rapid screening (usually within seconds).
    - **Molecular Weight Specification**: Exact masses in the library are calculated via RDKit's `rdMolDescriptors.CalcExactMolWt(mol)`, which determines the monoisotopic mass using the most abundant isotope by default (e.g., <sup>79</sup>Br instead of <sup>81</sup>Br). Users should follow this convention for isotopic compounds. In practice, enter the nearest integer within a ±0.5 tolerance of the calculated mass (e.g., input **169** for a mass of **169.1103**).
    """)

    with gr.Row():
        with gr.Column(scale=1):
            ir_input = gr.File(label="Upload IR Spectrum (.jdx, .csv)")

            gr.Markdown(
                "**🧪 Example Spectral Data:**\n"
                "- `5-Cyclo-5alpha-androstan-6beta-ol, 17-ethylenedioxy-3alpha, p-nitrobenzoate` (SMILES: `CC12CCC3C(CC(OC(=O)c4ccc([N+](=O)[O-])cc4)C45CC4CCC35C)C1CCC21OCCO1`)\n"
                "- Formula: `C28H35NO6` | **Recommended Mass Input: `481`**"
            )

            gr.Examples(
                examples=[
                    [os.path.join(EXAMPLE_DIR, "5-Cyclo-5alpha-androstan-6beta-ol_17-ethylenedioxy-3alpha_p-nitrobenzoate.jdx")]
                ],
                inputs=[ir_input],
                label="Step 1: Load Example Spectrum",
                elem_id="white-text-example"
            )

            mw_input = gr.Textbox(
                label="Mass (Optional)", 
                placeholder="e.g. 481"
            )
            formula_input = gr.Textbox(
                label="Formula (Optional)", 
                placeholder="e.g. C28H35NO6"
            )

            with gr.Group(visible=True) as scale_group:
                gr.Markdown("#### 🔍 Search Scale Settings (IR-only mode)")
                full_lib_checkbox = gr.Checkbox(
                    label="Search Full 100M Library",
                    value=True,
                    info="When checked, searches the entire database. Uncheck to dynamically configure a smaller scale."
                )
                dynamic_scale_slider = gr.Slider(
                    minimum=10000,
                    maximum=engine.total_db_size if engine.total_db_size > 0 else 100000000,
                    value=1000000,
                    step=10000,
                    label="Custom Dynamic Library Scale (Samples)",
                    visible=False,
                    info="Slide to freely adjust the number of search candidates."
                )

            def toggle_scale_controls(is_full, mw_val, form_val):
                has_filters = bool((mw_val and str(mw_val).strip()) or (form_val and str(form_val).strip()))
                if has_filters:
                    return gr.update(visible=False), gr.update(value=True), gr.update(visible=False)
                if is_full:
                    return gr.update(visible=True), gr.update(value=True), gr.update(visible=False)
                else:
                    return gr.update(visible=True), gr.update(value=False), gr.update(visible=True)

            full_lib_checkbox.change(
                fn=toggle_scale_controls,
                inputs=[full_lib_checkbox, mw_input, formula_input],
                outputs=[scale_group, full_lib_checkbox, dynamic_scale_slider]
            )
            mw_input.change(
                fn=toggle_scale_controls,
                inputs=[full_lib_checkbox, mw_input, formula_input],
                outputs=[scale_group, full_lib_checkbox, dynamic_scale_slider]
            )
            formula_input.change(
                fn=toggle_scale_controls,
                inputs=[full_lib_checkbox, mw_input, formula_input],
                outputs=[scale_group, full_lib_checkbox, dynamic_scale_slider]
            )

            top_k_slider = gr.Slider(minimum=10, maximum=100, value=50, step=10, label="Top-K Candidates")
            search_btn = gr.Button("🚀 Start Retrieval", variant="primary")

        with gr.Column(scale=2):
            status_output = gr.Textbox(label="System Status")
            conf_display = gr.HTML(visible=False, elem_classes="white-box")
            result_table = gr.DataFrame(label="Candidate Hits (Top-K)")

    search_btn.click(
        fn=engine.search,
        inputs=[ir_input, mw_input, formula_input, top_k_slider, full_lib_checkbox, dynamic_scale_slider],
        outputs=[status_output, result_table, conf_display]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        share=False,
        allowed_paths=[EXAMPLE_DIR]
    )
