import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelInference(object):
    def __init__(self, SmilesModel, IR_model, pretrain_model_path_sm, pretrain_model_path_ir, device):

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.smilesmodel = SmilesModel
        self.irmodel = IR_model

        self.smilesmodel.load_weights(path=pretrain_model_path_sm)
        self.irmodel.load_weights(path=pretrain_model_path_ir)

        self.smilesmodel = self.smilesmodel.to(self.device)
        self.irmodel = self.irmodel.to(self.device)

        self.smilesmodel.eval()
        self.irmodel.eval()

    def smiles_encode(self, smiles_str):
        with torch.no_grad():
            tokenizer = self.smilesmodel.smiles_tokenizer
            encoded_smiles = [tokenizer.encode_plus(
                text=smiles,
                max_length=300,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ) for smiles in smiles_str]

            input_ids = torch.cat([item['input_ids'] for item in encoded_smiles], dim=0).to(self.device)
            attention_mask = torch.cat([item['attention_mask'] for item in encoded_smiles], dim=0).to(self.device)
            lengths = attention_mask.sum(dim=1)

            smiles_feature = self.smilesmodel.encode((input_ids, attention_mask),lengths)
            return smiles_feature

    def ir_encode(self, ir):
        with torch.no_grad():
            ir_feature = self.irmodel(ir.view(1, -1)).to(self.device)
            return ir_feature



def get_feature_from_smiles(smiles_list, model_inference, batch_size=288):
    contexts = []
    print("start load batch")
    for i in range(0, len(smiles_list), batch_size):
        contexts.append(smiles_list[i:i + batch_size])
    print("start encode batch")
    result = [model_inference.smiles_encode(i).cpu() for i in tqdm(contexts)]
    result = torch.cat(result, 0)
    return result

def get_topK_result(ir_features, smiles_features, topK):
    indices = []
    scores = []
    with torch.no_grad():
        for i in ir_features:
            ir_smiles_distances_tmp = (
                    i.unsqueeze(0) @ smiles_features.to(device).t()).cpu()
        scores_, indices_ = ir_smiles_distances_tmp.topk(topK,
                                                          dim=1,
                                                          largest=True,
                                                          sorted=True)
        indices.append(indices_)
        scores.append(scores_)
    indices = torch.cat(indices, 0)
    scores = torch.cat(scores, 0)
    return indices, scores

def normalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    normalized_smiles = Chem.MolToSmiles(mol, canonical=True)
    return normalized_smiles


def draw_molecules(smiles_list, scores_list):
    if len(smiles_list) != len(scores_list):
        raise ValueError("smiles_list and scores_list must be the same length!")
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    legends = [f"Top {i + 1}: {round(score, 4)}" for i, score in enumerate(scores_list)]
    valid_mols_legends = [(mol, legend) for mol, legend in zip(mols, legends) if mol is not None]
    valid_mols, valid_legends = zip(*valid_mols_legends)
    img = Draw.MolsToGridImage(
        valid_mols,
        molsPerRow=5,
        subImgSize=(300, 300),
        legends=list(valid_legends)
    )
    return img



