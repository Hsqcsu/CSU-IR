import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import pandas as pd
import numpy as np
import jcamp

# If there is a red underline below, don't worry, it will not affect the code running
from model.IR_encoder import IRModel
from model.SMILES_encoder import SmilesModel

from data_process.ir_process import preprocess_jdx_spectra_higer_500
from data_process.ir_process import preprocess_jdx_spectra_lower_500
from data_process.ir_process import preprocess_csv_spectra_higer_500
from data_process.ir_process import preprocess_csv_spectra_lower_500

from test_and_infer.infer import ModelInference
from test_and_infer.infer import get_feature_from_smiles
from test_and_infer.infer import get_topK_result
from test_and_infer.infer import draw_molecules

'''
You need to download the model weight file in hugging_face and save it in the check_points folder.
'''

TOKENIZER_PATH = os.path.join(PROJECT_ROOT,'model',"tokenizer-smiles-roberta-1e_new")
PRETRAIN_SMILES_MODEL_PATH = os.path.join(PROJECT_ROOT, "check_points", "best_smiles_model_0.9230379746835443.pth")
PRETRAIN_IR_MODEL_PATH = os.path.join(PROJECT_ROOT, "check_points", "best_ir_model_0.9230379746835443.pth")
LIBRARY_FILE_PATH = os.path.join(PROJECT_ROOT,'data', "example_library_and_ir_for_user_dinfined", "library.txt")
EXAMPLE_IR_FILE_PATH = os.path.join(PROJECT_ROOT, 'data',"example_library_and_ir_for_user_dinfined", "Ethanol.jdx")
OUTPUT_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "retrieved_molecules.png")


IR_model = IRModel()
SmilesModel = SmilesModel(roberta_model_path=None,
    roberta_tokenizer_path= TOKENIZER_PATH,
    smiles_maxlen=300,
    max_position_embeddings=505,
    vocab_size=181,
    feature_dim=768,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IR_model.to(device)
SmilesModel.to(device)


ModelInferenc = ModelInference(
        SmilesModel,
        IR_model,
        pretrain_model_path_sm= PRETRAIN_SMILES_MODEL_PATH,
        pretrain_model_path_ir= PRETRAIN_IR_MODEL_PATH,
        device=None
    )

'------------------------------------------------------------------------------------------------'
# load you library here
def load_smiles(smiles_path):
    with open(smiles_path, 'r') as f:
        smiles_list = f.read().splitlines()
    return smiles_list

your_library = load_smiles(LIBRARY_FILE_PATH)
'------------------------------------------------------------------------------------------------'

import gzip

# define retrieval funtions
def retrieval_library(ir_feature, your_library):
    your_library_smiles_features = get_feature_from_smiles(your_library, ModelInferenc)
    indices, scores = get_topK_result(ir_feature, your_library_smiles_features, 10)

    top_smiles = []
    top_scores = []

    for sco, idx in zip(scores[0], indices[0]):
        retrieved_smiles = your_library[idx.item()]
        top_smiles.append(retrieved_smiles)
        top_scores.append(sco.item())


    img = draw_molecules(top_smiles,top_scores)
    return img, top_scores, top_smiles

def retrieval(ir_spectra_file, spectrum_type, your_library):
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
            ir_data = preprocess_spectra_higer_500(wavenumbers, transmittances)
        else:
            ir_data = preprocess_spectra_lower_500(file_path, wavenumbers, transmittances)
    else:
        transmittances = transmittances / 100.0
        if wavenumbers[0] > 500:
            ir_data = preprocess_spectra_higer_500(wavenumbers, transmittances)
        else:
            ir_data = preprocess_spectra_lower_500(file_path, wavenumbers, transmittances)

    ir_spectra_tensor = torch.tensor(ir_data, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        ir_feature = ModelInferenc.ir_encode(ir_spectra_tensor)

    img, scores,top_smiles = retrieval_library(ir_feature, your_library)
    return img, scores,top_smiles



if __name__ == "__main__":
    ir_spectra_file = EXAMPLE_IR_FILE_PATH
    img, scores, top_smiles = retrieval(ir_spectra_file , spectrum_type="absorbance spectrum", your_library=your_library)
    if img:
        img.save(OUTPUT_IMAGE_PATH)

