import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import jcamp

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# If there is a red underline below, don't worry, it will not affect the code running
from model.IR_encoder import IRModel
from model.Classifier_model import classifymodel
from data_process.ir_process import preprocess_spectra_higer_500
from data_process.ir_process import preprocess_spectra_lower_500


IR_model = IRModel()
Classifier_model = classifymodel(dim=1024,num_classes=2)


PRETRAIN_IR_MODEL_PATH = os.path.join(PROJECT_ROOT, "check_points", "best_ir_model_650-4000.pth")
PRETRAIN_IClassifier_MODEL_PATH = os.path.join(PROJECT_ROOT, "check_points", "best_IR_Classifier_model.pth")
SINGLE_PS_IR_PATH = os.path.join(PROJECT_ROOT, "data", "test_data", "IR_Classifier", "#773 - N-desethyletonitazene HCl (Lot# 0603955-33).JDX")
SINGLE_NON_PS_PATH = os.path.join(PROJECT_ROOT, "data", "test_data", "IR_Classifier", "(3-MERCAPTOPROPYL)TRIMETHOXYSILANE, 95%.CSV")


IR_model.load_weights(PRETRAIN_IR_MODEL_PATH)
Classifier_model.load_weights(PRETRAIN_IClassifier_MODEL_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IR_model.to(device)
Classifier_model.to(device)


def predict_ir(ir_spectra_file, spectrum_type):
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

    ir_spectra_tensor = torch.tensor(ir_data, dtype=torch.float32).unsqueeze(0).to(device)[:,150:]
    ir_feature = IR_model(ir_spectra_tensor)
    logits = Classifier_model(ir_feature)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()

    return "psychotropic substances" if pred == 1 else "non-psychotic substance"


print(predict_ir(SINGLE_PS_IR_PATH , spectrum_type = 'transmittance spectrum'))
print(predict_ir(SINGLE_NON_PS_PATH , spectrum_type = 'absorbance spectrum'))

