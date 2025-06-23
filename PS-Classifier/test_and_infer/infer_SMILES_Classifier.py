from rdkit import Chem
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast


def normalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    normalized_smiles = Chem.MolToSmiles(mol, canonical=True)
    return normalized_smiles

def infer_internal_batch(smiles_model, classify_model, test_loader, device='cuda'):
    smiles_model.eval()
    classify_model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for smiles_batch,label_batch in tqdm(test_loader, desc="Inferencing", unit="batch"):
            label_batch = label_batch.to(device)
            with autocast():
                tokenizer = smiles_model.smiles_tokenizer
                encoded_smiles = [tokenizer.encode_plus(
                    text=smiles,
                    max_length=300,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                ) for smiles in smiles_batch]
                input_ids = torch.cat([item['input_ids'] for item in encoded_smiles], dim=0).to(device)
                attention_mask = torch.cat([item['attention_mask'] for item in encoded_smiles], dim=0).to(device)
                lengths = attention_mask.sum(dim=1)
                smiles_features = smiles_model.encode((input_ids, attention_mask), lengths)
                logits = classify_model(smiles_features)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
            predictions.extend(preds.cpu().numpy())
            labels.extend(label_batch.cpu().numpy())

    return predictions, labels

def infer_external_batch(smiles_model, classify_model, test_loader, device='cuda'):
    smiles_model.eval()
    classify_model.eval()
    predictions = []
    with torch.no_grad():
        for smiles_batch in tqdm(test_loader, desc="Inferencing", unit="batch"):
            with autocast():
                tokenizer = smiles_model.smiles_tokenizer
                encoded_smiles = [tokenizer.encode_plus(
                    text=smiles,
                    max_length=300,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                ) for smiles in smiles_batch]
                input_ids = torch.cat([item['input_ids'] for item in encoded_smiles], dim=0).to(device)
                attention_mask = torch.cat([item['attention_mask'] for item in encoded_smiles], dim=0).to(device)
                lengths = attention_mask.sum(dim=1)
                smiles_features = smiles_model.encode((input_ids, attention_mask), lengths)
                logits = classify_model(smiles_features)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
            predictions.extend(preds.cpu().numpy())

    return predictions