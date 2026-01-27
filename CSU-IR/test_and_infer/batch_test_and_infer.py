
from matplotlib.style.core import library

from ir_CNN_VIT_TR import IRModel
from add_new_tokens_smiles_model import SmilesModel
from tqdm import tqdm
import torch.nn.functional as F
from rdkit import Chem
import gzip



import torch
from torch.utils.data import Dataset, DataLoader


# 定义自定义数据集类
class IRSmilesDataset(Dataset):
    def __init__(self, ir_spectra, smiles):
        self.ir_spectra = ir_spectra
        self.smiles = smiles

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.ir_spectra[idx], self.smiles[idx]


def load_data(smiles_path, ir_path):
    with open(smiles_path, 'r') as f:
        smiles_list = f.read().splitlines()
    ir_data = torch.load(ir_path)
    return smiles_list, ir_data






'--------------------------------------------------------------------------------------------'

Derivative_drug_path = r'F:\Spectrum\Drug\smirks\filtered_aug_smiles_smirks.txt'

def load_smiles(smiles_path):
    with open(smiles_path, 'r') as f:
        smiles_list = f.read().splitlines()
    return smiles_list
All_Drug_smiles_path_raw = r"E:\Spectrum\data\ENFSI_DWG_IR_Library_JCAMP-DX_20240508\data\1_1_1_1_data_drug_uni_dup\merged_smiles_unique.txt"
ALL_Drug_smiles_list_raw = load_smiles(All_Drug_smiles_path_raw)

#NPS_smiles_path = r'E:\Spectrum\data\USA_drug_IR\processed_data\with_smiles\60\continue_delete_5_smiles\continue_filter\final_filter\filtered_new_smiles_with_data_filtered.txt'
#NPS_ir_path = r'E:\Spectrum\data\USA_drug_IR\processed_data\with_smiles\60\continue_delete_5_smiles\continue_filter\final_filter\filtered_new_ir_with_data_filtered.pt'

NPS_smiles_path = r'E:\Spectrum\data\USA_drug_IR\processed_data\with_smiles\60\continue_delete_5_smiles\continue_delete_consider_ste\continue_filter\\filtered_final_NPS_smiles.txt'
NPS_ir_path = r'E:\Spectrum\data\USA_drug_IR\processed_data\with_smiles\60\continue_delete_5_smiles\continue_delete_consider_ste\continue_filter\\filtered_final_NPS_ir.pt'

#NPS_smiles_path = r'E:\Spectrum\data\USA_drug_IR\processed_data\with_smiles\60\continue_delete_5_smiles\continue_delete_consider_ste\final_NPS_smiles.txt'
#NPS_ir_path = r'E:\Spectrum\data\USA_drug_IR\processed_data\with_smiles\60\continue_delete_5_smiles\continue_delete_consider_ste\final_NPS_ir.pt'

Derivative_drug_smiles_list = load_smiles(Derivative_drug_path)

NPS_smiles, NPS_ir = load_data(NPS_smiles_path, NPS_ir_path)

NPS_dataset = IRSmilesDataset(NPS_ir, NPS_smiles)
batch_size = 208
NPS_loader = DataLoader(NPS_dataset, batch_size=batch_size, shuffle=False)


def normalize_smiles(smiles):
    """
    将输入的 SMILES 字符串转换为规范化的 SMILES。

    参数:
    smiles (str): 输入的 SMILES 字符串。

    返回:
    str: 规范化的 SMILES 字符串。如果分子对象无效，则返回原始的 SMILES 字符串。
    """
    # 使用 RDKit 将 SMILES 转换为分子对象
    mol = Chem.MolFromSmiles(smiles)

    # 检查分子对象是否有效
    if mol is None:
        return smiles

    # 使用 RDKit 将分子对象转换为规范化的 SMILES
    normalized_smiles = Chem.MolToSmiles(mol, isomericSmiles=False,canonical=True)

    return normalized_smiles



IR_model = IRModel()
SmilesModel = SmilesModel(roberta_model_path=None,
    roberta_tokenizer_path= "F:\\Spectrum\\models\\models\\tokenizer-smiles-roberta-1e_new",
    smiles_maxlen=300,
    max_position_embeddings=505,
    vocab_size=181,
    feature_dim=768,
)



# 训练过程
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SmilesModel.to(device)
IR_model.to(device)


# 定义模型路径
# 定义模型路径
model_pairs = [
    (
    r'F:\Spectrum\1122_after\model\ESA_model_sigmoid\20250530_esa_ir_CNN_transformer\Optimized\trained_lr_without_restart_new\lr_turning_10_Sigmoid\results_trial_0_lr_2.6e-04\best_smiles_model_0.9230379746835443.pth',
    r'F:\Spectrum\1122_after\model\ESA_model_sigmoid\20250530_esa_ir_CNN_transformer\Optimized\trained_lr_without_restart_new\lr_turning_10_Sigmoid\results_trial_0_lr_2.6e-04\best_ir_model_0.9230379746835443.pth'),
]

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
            # 编码 SMILES 数据
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

            smiles_features = self.smilesmodel.encode((input_ids, attention_mask),lengths)

            # 进行 L2 归一化
            I_e = F.normalize(smiles_features, p=2, dim=1)
            return I_e

    def ir_encode(self, ir_list):
        # print(nmr_list.shape)
        with torch.no_grad():
            ir_tensor = self.irmodel(ir_list.view(1, -1)).to(self.device)
            T_e = F.normalize(ir_tensor, p=2, dim=1)
            return T_e



def get_feature_from_smiles(smiles_list, model_inference, batch_size=288):
    contexts = []
    print("start load batch")
    for i in range(0, len(smiles_list), batch_size):
        contexts.append(smiles_list[i:i + batch_size])
    print("start encode batch")
    result = [model_inference.smiles_encode(i).cpu() for i in tqdm(contexts)]
    result = torch.cat(result, 0)
    return result

def get_topK_result(nmr_feature, smiles_feature, topK):
    indices = []
    scores = []
    with torch.no_grad():
        for i in nmr_feature:
            # 将 smiles_feature 移动到与 i 相同的设备
            nmr_smiles_distances_tmp = (
                    i.unsqueeze(0) @ smiles_feature.to(device).t()).cpu()
        scores_, indices_ = nmr_smiles_distances_tmp.topk(topK,
                                                          dim=1,
                                                          largest=True,
                                                          sorted=True)
        indices.append(indices_)
        scores.append(scores_)
    indices = torch.cat(indices, 0)
    scores = torch.cat(scores, 0)
    return indices, scores

def evaluate_loader(loader, combined_features, smiles_list, loader_name):
    top_1_matches = 0
    top_2_matches = 0
    top_3_matches = 0
    top_4_matches = 0
    top_5_matches = 0
    top_10_matches = 0
    total_samples = 0

    results = []

    test_loader_tqdm = tqdm(loader, unit="batch")
    for ir_spectra_batch, smiles_batch in test_loader_tqdm:
        ir_spectra_tensor = ir_spectra_batch.to(device)
        #ir_spectra_tensor = ir_spectra_tensor[:, 150:]
        for idx, ir_spectra in enumerate(ir_spectra_tensor):
            smiles = smiles_batch[idx]
            original_smiles = normalize_smiles(smiles)

            # 提取 IR 光谱特征向量
            ir_feature = ModelInferenc.ir_encode(ir_spectra)
            # 获取 top 10 候选项
            indices, scores = get_topK_result(ir_feature, combined_features, 10)

            # 打印结果
            top_smiles = []
            for (sco, idx) in zip(scores, indices):
                for ii, i in enumerate(idx):
                    if i < len(smiles_list):
                        top_smiles.append(smiles_list[i])

            # 保存结果
            results.append((top_smiles, scores.tolist()))

            # 检查 top 10 中是否有匹配
            for ii, i in enumerate(indices[0]):  # 只检查第一个候选
                if i < len(smiles_list) and original_smiles == normalize_smiles(smiles_list[i]):
                    if ii == 0:
                        top_1_matches += 1
                    if ii < 2:
                        top_2_matches += 1
                    if ii < 3:
                        top_3_matches += 1
                    if ii < 4:
                        top_4_matches += 1
                    if ii < 5:
                        top_5_matches += 1
                    if ii < 10:
                        top_10_matches += 1
                    break  # 找到匹配后可以退出循环

            total_samples += 1
            print(total_samples)

    # 计算top-k比率
    top_1_ratio = top_1_matches / total_samples if total_samples > 0 else 0
    top_2_ratio = top_2_matches / total_samples if total_samples > 0 else 0
    top_3_ratio = top_3_matches / total_samples if total_samples > 0 else 0
    top_4_ratio = top_4_matches / total_samples if total_samples > 0 else 0
    top_5_ratio = top_5_matches / total_samples if total_samples > 0 else 0
    top_10_ratio = top_10_matches / total_samples if total_samples > 0 else 0

    # 打印结果
    print(f"Results for {loader_name}:")
    print("Top-1 比率:", top_1_ratio)
    print("Top-2 比率:", top_2_ratio)
    print("Top-3 比率:", top_3_ratio)
    print("Top-4 比率:", top_4_ratio)
    print("Top-5 比率:", top_5_ratio)
    print("Top-10 比率:", top_10_ratio)


    # 保存结果到文件
    with open(f'{loader_name}_results.txt', 'w') as file:
        for top_smiles, scores in results:
            file.write(f'Top SMILES: {top_smiles}\n')
            file.write(f'Scores: {scores}\n\n')




# 迭代处理每对模型
for smiles_model_path, ir_model_path in model_pairs:
    print(f"Processing models: {smiles_model_path} and {ir_model_path}")

    ModelInferenc = ModelInference(
        SmilesModel,
        IR_model,
        pretrain_model_path_sm=smiles_model_path,
        pretrain_model_path_ir=ir_model_path,
        device=None
    )


    #NPS_smiles_features = get_feature_from_smiles(NPS_smiles,ModelInferenc)

    #Existed_PS_library = get_feature_from_smiles(ALL_Drug_smiles_list_raw,ModelInferenc)

    #Derivative_PS_library = get_feature_from_smiles(Derivative_drug_smiles_list,ModelInferenc)

    #library_Existed_PS = torch.cat((NPS_smiles_features, Existed_PS_library), dim=0)#.to(torch.float16)
    #library_Derivative_PS = torch.cat((NPS_smiles_features, Derivative_PS_library), dim=0)#.to(torch.float16)

    '''
    library_Existed_PS_file_path_gz = 'library_Existing_PS_smiles_features_fp16.gz'
    with gzip.open(library_Existed_PS_file_path_gz, 'wb') as f:
        torch.save(library_Existed_PS, f)
    
    library_Derivative_PS_file_path_gz = 'library_Derivative_PS_smiles_features_fp16.gz'
    with gzip.open(library_Derivative_PS_file_path_gz, 'wb') as f:
        torch.save(library_Derivative_PS, f)
    '''


    def load_gz_features(path, device):
        with gzip.open(path, 'rb') as f: features = torch.load(f)
        return features.to(device)
    library_Derivative_PS =  load_gz_features(r'F:\Spectrum\github\CSU-IR\data\processed_library\PS\library_Derivative_PS_smiles_features_fp16.gz',device='cuda')
    smiles_Derivative_PS = load_smiles(r'F:\Spectrum\github\CSU-IR\data\processed_library\PS\smiles_Derivative_PS.txt')

    #smiles_Derivative_PS = list(NPS_smiles) + list(Derivative_drug_smiles_list)


    #smiles_Existed_PS = list(NPS_smiles) + list(ALL_Drug_smiles_list_raw)
    #smiles_Derivative_PS = list(NPS_smiles) + list(Derivative_drug_smiles_list)

    def save_smiles_list(smiles_list, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for smi in smiles_list:
                f.write(smi + '\n')


    #save_smiles_list(smiles_Existed_PS, 'smiles_Existing_PS.txt')
    #save_smiles_list(smiles_Derivative_PS, 'smiles_Derivative_PS.txt')

    evaluate_loader(NPS_loader,library_Derivative_PS, smiles_Derivative_PS, '1')




