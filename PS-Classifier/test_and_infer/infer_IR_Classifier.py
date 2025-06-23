from rdkit import Chem
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm



def normalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    normalized_smiles = Chem.MolToSmiles(mol, canonical=True)
    return normalized_smiles

def infer_internal_batch(ir_model, classify_model, test_loader, device='cuda'):
    ir_model.eval()
    classify_model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for ir_spectra_batch, label_batch in tqdm(test_loader, desc="Inferencing", unit="batch"):
            ir_spectra_tensor = ir_spectra_batch.to(device)
            ir_spectra_tensor = ir_spectra_tensor[:, 150:]
            label_batch = label_batch.to(device)
            with autocast():
                ir_features = ir_model(ir_spectra_tensor)
                logits = classify_model(ir_features)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
            predictions.extend(preds.cpu().numpy())
            labels.extend(label_batch.cpu().numpy())

    return predictions, labels

def infer_external_batch(ir_model, classify_model, test_loader, device='cuda'):
    ir_model.eval()
    classify_model.eval()
    predictions = []
    with torch.no_grad():
        for ir_spectra_batch, _ in tqdm(test_loader, desc="Inferencing", unit="batch"):
            ir_spectra_tensor = ir_spectra_batch.to(device)
            ir_spectra_tensor = ir_spectra_tensor[:, 150:]
            with autocast():
                ir_features = ir_model(ir_spectra_tensor)
                logits = classify_model(ir_features)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
            predictions.extend(preds.cpu().numpy())

    return predictions

