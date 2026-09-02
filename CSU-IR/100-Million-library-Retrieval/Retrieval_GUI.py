'''
This script provides a Web service for retrieving the 100-Million-Scale library:
1. Infrared spectroscopy only;
2. Infrared spectroscopy plus monoisotopic mass;
3. Infrared spectroscopy plus molecular formula.

Spectra (.csv or .jdx format in either absorbance or transmittance) are 
automatically parsed and preprocessed without manual mode selection.
'''

import os
import sys
import torch
import numpy as np
import pandas as pd
import jcamp
import gradio as gr
from scipy.interpolate import CubicSpline

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

EXAMPLE_DIR = os.path.join(PROJECT_ROOT, 'data', "example_library_and_ir_for_user_dinfined")

from Retrieval_functions import (
    load_MW_Formula,
    get_final_query_metadata,
    UnifiedCombinedLibrary,
    unified_retrieval_100M,
    calculate_calibrated_confidence
)

from model.IR_encoder import IRModel
from model.SMILES_encoder import SmilesModel
from test_and_infer.test_and_infer_functions import ModelInference

FEATURE_DIM = 1024
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== 1. Adaptive Spectral Preprocessing ====================
def _resample_and_pad_spectrum(wavenumbers, intensities):
    """
    Standardizes spectral resolution to 3500 points covering 500-4000 cm^-1 using cubic spline interpolation.
    """
    x = np.array(wavenumbers, dtype=float)
    y = np.array(intensities, dtype=float)
    valid_mask = (~np.isnan(x)) & (~np.isnan(y)) & (y != 0)
    x = x[valid_mask]
    y = y[valid_mask]
    if len(x) < 4:
        return None

    # Automatic micrometer to wavenumber conversion if needed
    if np.nanmax(x) < 100.0:
        x = 10000.0 / x

    _, unique_indices = np.unique(x, return_index=True)
    x = x[unique_indices]
    y = y[unique_indices]
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    min_x, max_x = x[0], x[-1]
    start = max(min_x, 500.0)
    end = min(max_x, 4000.0)
    if start >= end:
        return None

    mask = (x >= start) & (x <= end)
    x_crop = x[mask]
    y_crop = y[mask]
    if len(x_crop) < 4:
        return None

    start_int = int(np.ceil(start))
    end_int = int(np.floor(end))
    x_temp = np.arange(start_int, end_int + 1, 1.0)
    x_temp = np.clip(x_temp, x_crop[0], x_crop[-1])
    cs_crop = CubicSpline(x_crop, y_crop, extrapolate=True)
    y_temp = cs_crop(x_temp)

    x_full = np.arange(500, 4001, 1.0)
    y_full = np.zeros_like(x_full)
    idx_start = start_int - 500
    idx_end = end_int - 500
    y_full[idx_start:idx_end + 1] = y_temp
    if idx_start > 0:
        y_full[:idx_start] = y_temp[0]
    if idx_end < 3500:
        y_full[idx_end + 1:] = y_temp[-1]

    x_target = np.linspace(500.0, 4000.0, 3500)
    cs_full = CubicSpline(x_full, y_full, extrapolate=True)
    return cs_full(x_target)


def preprocess_universal_ir_spectrum(wavenumbers, intensities):
    """
    Intelligent adaptive preprocessor automatically supporting both CSV and JDX in transmittance or absorbance.
    """
    try:
        raw_y = np.array(intensities, dtype=float)

        # Scale percentage transmittance (0-100%) to (0-1.0)
        if np.nanmax(raw_y) > 1.5:
            raw_y = raw_y / 100.0

        processed_y = _resample_and_pad_spectrum(wavenumbers, raw_y)
        if processed_y is None:
            return None

        # Min-max normalization for preliminary baseline assessment
        min_val = np.min(processed_y)
        max_val = np.max(processed_y)
        if max_val > min_val:
            norm_spec = (processed_y - min_val) / (max_val - min_val)
        else:
            norm_spec = np.zeros_like(processed_y)

        # Baseline heuristic: Transmittance median > 0.45; Absorbance median <= 0.45
        spec_median = np.median(norm_spec)
        if spec_median > 0.45:
            spec_clipped = np.clip(norm_spec, 1e-4, 1.0)
            abs_spec = -np.log10(spec_clipped)
            min_val_abs = np.min(abs_spec)
            max_val_abs = np.max(abs_spec)
            if max_val_abs > min_val_abs:
                final_spec = (abs_spec - min_val_abs) / (max_val_abs - min_val_abs)
            else:
                final_spec = np.zeros_like(abs_spec)
        else:
            final_spec = norm_spec

        if np.any(np.isnan(final_spec)):
            return None
        return final_spec
    except Exception:
        return None


def process_ir(ir_file_path, model_infer_instance):
    """Parses IR spectrum file and generates embedding vector."""
    if ir_file_path.lower().endswith('.csv'):
        df = pd.read_csv(ir_file_path, header=None)
        wavenumbers, intensities = df.iloc[:, 0].values, df.iloc[:, 1].values
        ir_data = preprocess_universal_ir_spectrum(wavenumbers, intensities)
    elif ir_file_path.lower().endswith('.jdx'):
        data = jcamp.jcamp_readfile(ir_file_path)
        wavenumbers, intensities = np.array(data['x'], dtype=float), np.array(data['y'], dtype=float)
        ir_data = preprocess_universal_ir_spectrum(wavenumbers, intensities)
    else:
        raise ValueError("Unsupported file format. Please upload a .jdx or .csv file.")

    if ir_data is None:
        raise ValueError("Failed to process spectrum. Please ensure sufficient valid data points.")

    ir_spectra_tensor = torch.tensor(ir_data, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        ir_feature = model_infer_instance.ir_encode(ir_spectra_tensor)
    return ir_feature


# ==================== 2. Retrieval Engine ====================
class IR_Retrieval_Engine_100M:
    def __init__(self):
        self.tokenizer_path = os.path.join(PROJECT_ROOT, 'model', "tokenizer-smiles-roberta-1e_new")
        self.pretrain_smiles_path = os.path.join(PROJECT_ROOT, "check_points", "Multi-stage_training_Stage_III_EXP", "MD_DFT_EXP", "best_smiles_model.pth")
        self.pretrain_ir_path = os.path.join(PROJECT_ROOT, "check_points", "Multi-stage_training_Stage_III_EXP", "MD_DFT_EXP", "best_ir_model.pth")

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
                    "mw": os.path.join(part_folder, f'global_pool_features_100M_1024dim_fp16_part_{p_name}_mw_part_{p_name}_sub{sub_i}.txt')
                }
                self.lib_configs.append(config)

        self.lib_manager = UnifiedCombinedLibrary(self.lib_configs)

    def search(self, ir_file, mw, formula, top_k, search_range):
        try:
            if ir_file is None:
                return "Please upload an IR spectrum file.", None, gr.update(visible=False)

            ir_feature = process_ir(ir_file.name, self.model_infer)

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
            top1_score = float(df.iloc[0]['similarity'])

            # Compute calibrated confidence using NIST baseline parameters (scale=0.2187, bias=-0.3748)
            top1_calibrated_conf = calculate_calibrated_confidence(
                top1_score,
                self.sm_model,
                scale=0.2187,
                bias=-0.3748
            )

            df['Calibrated Confidence'] = df['similarity'].map(
                lambda s: f"{calculate_calibrated_confidence(float(s), self.sm_model, scale=0.2187, bias=-0.3748) * 100:.2f}%"
            )

            # Confidence Statement and Enhanced Domain Guidance Note
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

            df = df[['rank', 'similarity', 'Calibrated Confidence', 'formula', 'smiles']]
            df['similarity'] = df['similarity'].map(lambda x: f"{float(x):.4f}")
            df.columns = ['Rank', 'Cosine Similarity', 'Calibrated Confidence', 'Formula', 'SMILES']

            return "Search completed!", df, gr.update(value=conf_summary, visible=True)

        except Exception as e:
            return f"Error: {str(e)}", None, gr.update(visible=False)


engine = IR_Retrieval_Engine_100M()


def handle_range_visibility(mw, formula):
    """Automatically locks search range to Full Library when molecular weight or formula is provided."""
    if (mw and mw.strip()) or (formula and formula.strip()):
        return gr.update(visible=False, value="Full Library")
    return gr.update(visible=True)


# ==================== 3. Gradio User Interface ====================
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
    - **Adaptive Spectral Preprocessing**: Both **.jdx** and **.csv** files in either **absorbance** or **transmittance** mode are automatically parsed.
    - **Retrieval Priority**: For optimal performance, the search follows: Formula-based filtering > Exact Mass filtering > IR-only search.
    - **Search Time**: Searching across the full 100M library in IR-only mode takes approximately **8m 30s**. Providing molecular weight or chemical formula enables rapid screening (usually within 30 seconds).
    - **Molecular Weight Specification**: Exact masses in the library are calculated via RDKit's `rdMolDescriptors.CalcExactMolWt(mol)`, which determines the monoisotopic mass using the most abundant isotope by default (e.g., <sup>79</sup>Br instead of <sup>81</sup>Br). Users should follow this convention for isotopic compounds. In practice, enter the nearest integer within a ±0.5 tolerance of the calculated mass (e.g., input **169** for a mass of **169.1103**).
    """)

    with gr.Row():
        with gr.Column(scale=1):
            ir_input = gr.File(label="Upload IR Spectrum (.jdx, .csv)")

            gr.Markdown(
                "**🧪 Example Spectral Data:**\n"
                "- `4-(Methylthio)phenol`: Mass Input`140`, Formula Input`C7H8OS`"
            )

            gr.Examples(
                examples=[
                    [os.path.join(EXAMPLE_DIR, '4-(Methylthio)phenol.CSV')]
                ],
                inputs=[ir_input],
                label="Step 1: Load Example Spectrum",
                elem_id="white-text-example"
            )

            mw_input = gr.Textbox(
                label="Mass (Optional)", 
                placeholder="e.g. 140"
            )
            formula_input = gr.Textbox(
                label="Formula (Optional)", 
                placeholder="e.g. C7H8OS"
            )

            search_range = gr.Dropdown(
                choices=["1w", "10w", "100w", "1000w", "Full Library"],
                value="Full Library",
                label="Search Range (Only for IR-only search)"
            )

            mw_input.change(handle_range_visibility, inputs=[mw_input, formula_input], outputs=[search_range])
            formula_input.change(handle_range_visibility, inputs=[mw_input, formula_input], outputs=[search_range])

            top_k_slider = gr.Slider(minimum=10, maximum=100, value=50, step=10, label="Top-K Candidates")
            search_btn = gr.Button("🚀 Start Retrieval", variant="primary")

        with gr.Column(scale=2):
            status_output = gr.Textbox(label="System Status")
            conf_display = gr.HTML(visible=False, elem_classes="white-box")
            result_table = gr.DataFrame(label="Candidate Hits (Top-K)")

    search_btn.click(
        fn=engine.search,
        inputs=[ir_input, mw_input, formula_input, top_k_slider, search_range],
        outputs=[status_output, result_table, conf_display]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        share=False,
        allowed_paths=[EXAMPLE_DIR]
    )
