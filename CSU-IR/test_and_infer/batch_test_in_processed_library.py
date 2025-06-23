import sys
import os
import gzip
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# --- 1. Set up the project root directory, which is the base for all relative paths ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"Project Root Path: {PROJECT_ROOT}")
sys.path.insert(0, PROJECT_ROOT)

# --- 2. Import custom modules from within the project ---
# If there is a red underline below, don't worry, it will not affect the code running
from model.IR_encoder import IRModel
from model.SMILES_encoder import SmilesModel
from data_process.ir_process import preprocess_spectra_higer_500
from data_process.ir_process import preprocess_spectra_lower_500
from test_and_infer.infer import ModelInference
from test_and_infer.infer import get_feature_from_smiles
from test_and_infer.infer import get_topK_result
from test_and_infer.infer import draw_molecules
from test_and_infer.infer import normalize_smiles

# --- 3. Define dynamic paths for all files and models ---
# Paths for models and tokenizer
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "model", "tokenizer-smiles-roberta-1e_new")
PRETRAIN_SMILES_MODEL_PATH = os.path.join(PROJECT_ROOT, "check_points", "best_smiles_model_500-4000.pth")
PRETRAIN_IR_MODEL_PATH = os.path.join(PROJECT_ROOT, "check_points", "best_ir_model_500-4000.pth")

# Root directories for processed feature libraries and SMILES lists
# You need to download the corresponding data in huggingface and put it in the corresponding folder
LIB_GENERAL_PATH = os.path.join(PROJECT_ROOT, "data", "processed_library", "General")
LIB_PS_PATH = os.path.join(PROJECT_ROOT, "data", "processed_library", "PS")

# Root directory for test datasets
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "test_data")

# --- 4. Initialize the models ---
print("Initializing models...")
IR_model = IRModel()
SmilesModel = SmilesModel(
    roberta_model_path=None,
    roberta_tokenizer_path=TOKENIZER_PATH,
    smiles_maxlen=300,
    max_position_embeddings=505,
    vocab_size=181,
    feature_dim=768,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
IR_model.to(device)
SmilesModel.to(device)

ModelInferenc = ModelInference(
    SmilesModel,
    IR_model,
    pretrain_model_path_sm=PRETRAIN_SMILES_MODEL_PATH,
    pretrain_model_path_ir=PRETRAIN_IR_MODEL_PATH,
    device=device
)


# --- 5. Define data loading helper functions ---
def load_gz_features(path):
    """Loads a .gz compressed feature file and converts it to float32."""
    with gzip.open(path, 'rb') as f:
        features = torch.load(f)
    return features.to(torch.float32)


def load_smiles_list(path):
    """Loads a SMILES list from a text file."""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def load_test_data(smiles_path, ir_path):
    """Loads SMILES and IR data for a test set."""
    smiles_list = load_smiles_list(smiles_path)
    ir_data = torch.load(ir_path)
    return smiles_list, ir_data


class IRSmilesDataset(Dataset):
    def __init__(self, ir_spectra, smiles):
        self.ir_spectra = ir_spectra
        self.smiles = smiles

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.ir_spectra[idx], self.smiles[idx]

# You need to download the corresponding data in huggingface and put it in the corresponding folder
# --- 6. Load all feature libraries and SMILES lists ---
print("Loading feature libraries and SMILES lists...")
# General Libraries
library_nist_features = load_gz_features(os.path.join(LIB_GENERAL_PATH, 'library_nist_smiles_features_fp16.gz'))
library_nist_filter_by_CHONF_features = load_gz_features(os.path.join(LIB_GENERAL_PATH, 'library_nist_filter_by_CHONF_smiles_features_fp16.gz'))
library_4k_features = load_gz_features(os.path.join(LIB_GENERAL_PATH, 'library_4k_smiles_features_fp16.gz'))
library_200k_features = load_gz_features(os.path.join(LIB_GENERAL_PATH, 'library_200k_smiles_features_fp16.gz'))
library_1M_features = load_gz_features(os.path.join(LIB_GENERAL_PATH, 'library_1M_smiles_features_fp16.gz'))
library_2M_features = load_gz_features(os.path.join(LIB_GENERAL_PATH, 'library_2M_smiles_features_fp16.gz'))

general_smiles_nist = load_smiles_list(os.path.join(LIB_GENERAL_PATH, 'smiles_nist.txt'))
general_smiles_nist_filter_by_CHONF = load_smiles_list(os.path.join(LIB_GENERAL_PATH, 'smiles_nist_filtered_by_CHONF.txt'))
general_smiles_4k = load_smiles_list(os.path.join(LIB_GENERAL_PATH, 'smiles_4k.txt'))
general_smiles_200k = load_smiles_list(os.path.join(LIB_GENERAL_PATH, 'smiles_200k.txt'))
general_smiles_1M = load_smiles_list(os.path.join(LIB_GENERAL_PATH, 'smiles_1M.txt'))
general_smiles_2M = load_smiles_list(os.path.join(LIB_GENERAL_PATH, 'smiles_2M.txt'))

# Psychoactive Substances (PS) Libraries
library_existed_ps_features = load_gz_features(os.path.join(LIB_PS_PATH, 'library_Existed_PS_smiles_features_fp16.gz'))
library_derivative_ps_features = load_gz_features(
    os.path.join(LIB_PS_PATH, 'library_Derivative_PS_smiles_features_fp16.gz'))
library_combined_ps_features = load_gz_features(
    os.path.join(LIB_PS_PATH, 'library_combined_PS_smiles_features_fp16.gz'))

ps_smiles_existed = load_smiles_list(os.path.join(LIB_PS_PATH, 'smiles_Existed_PS.txt'))
ps_smiles_derivative = load_smiles_list(os.path.join(LIB_PS_PATH, 'smiles_Derivative_PS.txt'))
ps_smiles_combined = load_smiles_list(os.path.join(LIB_PS_PATH, 'smiles_combined.txt'))

# --- 7. Create DataLoaders for the test sets ---
print("Creating test DataLoaders...")
batch_size = 208

# NPS Test Set
NPS_smiles, NPS_ir = load_test_data(
    os.path.join(TEST_DATA_PATH, "NPS", "NPS_smiles.txt"),
    os.path.join(TEST_DATA_PATH, "NPS", "NPS_ir.pt")
)
NPS_loader = DataLoader(IRSmilesDataset(NPS_ir, NPS_smiles), batch_size=batch_size, shuffle=False)

# Filtered NPS Test Set
NPS_smiles_filter, NPS_ir_filter = load_test_data(
    os.path.join(TEST_DATA_PATH, "NPS", "Derivative_filter", "filter_NPS_smiles.txt"),
    os.path.join(TEST_DATA_PATH, "NPS", "Derivative_filter", "filter_NPS_ir.pt")
)
NPS_loader_filter = DataLoader(IRSmilesDataset(NPS_ir_filter, NPS_smiles_filter), batch_size=batch_size, shuffle=False)

# Internal Test Set
Internal_test_smiles, Internal_test_ir = load_test_data(
    os.path.join(TEST_DATA_PATH, "Internal_test", "Internal_test_smiles.txt"),
    os.path.join(TEST_DATA_PATH, "Internal_test", "Internal_test_ir.pt")
)
Internal_test_loader = DataLoader(IRSmilesDataset(Internal_test_ir, Internal_test_smiles), batch_size=batch_size,
                                  shuffle=False)

# External NIST Test Set
External_nist_test_smiles, External_nist_test_ir = load_test_data(
    os.path.join(TEST_DATA_PATH, "External_nist_test", "nist_smiles.txt"),
    os.path.join(TEST_DATA_PATH, "External_nist_test", "nist_ir.pt")
)
External_nist_test_loader = DataLoader(IRSmilesDataset(External_nist_test_ir, External_nist_test_smiles),
                                       batch_size=batch_size, shuffle=False)


# External NIST Test Set Filtered by CHONF
External_nist_filtered_by_CHONF_test_smiles, External_nist_filtered_by_CHONF_test_ir = load_test_data(
    os.path.join(TEST_DATA_PATH, "External_nist_test", "nist_smiles_filtered_by_CHONF.txt"),
    os.path.join(TEST_DATA_PATH, "External_nist_test", "nist_ir_filtered_by_CHONF.pt")
)
External_nist_filtered_by_CHONF_test_test_loader = DataLoader(IRSmilesDataset(External_nist_filtered_by_CHONF_test_ir, External_nist_filtered_by_CHONF_test_smiles),
                                       batch_size=batch_size, shuffle=False)

# --- 8. Define the evaluation function ---
def evaluate_loader(loader, combined_features, smiles_list, loader_name):
    top_1_matches, top_5_matches, top_10_matches, total_samples = 0, 0, 0, 0
    test_loader_tqdm = tqdm(loader, desc=f"Evaluating {loader_name}", unit="batch")

    # Disable gradient calculations during evaluation for speed and memory efficiency
    with torch.no_grad():
        for ir_spectra_batch, smiles_batch in test_loader_tqdm:
            ir_spectra_tensor = ir_spectra_batch.to(device)
            for i, ir_spectra in enumerate(ir_spectra_tensor):
                original_smiles = normalize_smiles(smiles_batch[i])
                ir_feature = ModelInferenc.ir_encode(ir_spectra.unsqueeze(0))  # Ensure input has a batch dimension
                indices, _ = get_topK_result(ir_feature, combined_features, 10)

                # Check for Top-K matches
                for rank, lib_idx in enumerate(indices[0]):
                    if lib_idx < len(smiles_list) and original_smiles == normalize_smiles(smiles_list[lib_idx]):
                        if rank == 0: top_1_matches += 1
                        if rank < 5: top_5_matches += 1
                        if rank < 10: top_10_matches += 1
                        break  # Exit the inner loop once a match is found
                total_samples += 1

    # Calculate and print the results
    top_1_ratio = (top_1_matches / total_samples) if total_samples > 0 else 0
    top_5_ratio = (top_5_matches / total_samples) if total_samples > 0 else 0
    top_10_ratio = (top_10_matches / total_samples) if total_samples > 0 else 0

    print(f"\n--- Results for {loader_name} ---")
    print(f"Total Samples: {total_samples}")
    print(f"Top-1 Accuracy: {top_1_ratio:.4f} ({top_1_matches}/{total_samples})")
    print(f"Top-5 Accuracy: {top_5_ratio:.4f} ({top_5_matches}/{total_samples})")
    print(f"Top-10 Accuracy: {top_10_ratio:.4f} ({top_10_matches}/{total_samples})")


# --- 9. Execute the evaluations ---
print("\nStarting evaluations...")
evaluate_loader(NPS_loader, library_existed_ps_features, ps_smiles_existed, "NPS in Existed_PS_Library")
evaluate_loader(NPS_loader_filter, library_derivative_ps_features, ps_smiles_derivative,
                "Filtered_NPS in Derivative_PS_Library")
#evaluate_loader(Internal_test_loader, library_4k_features, general_smiles_4k, "Internal_Test in General_4k_Library")
#evaluate_loader(Internal_test_loader, library_200k_features, general_smiles_200k,"Internal_Test in General_200k_Library")
#evaluate_loader(Internal_test_loader, library_1M_features, general_smiles_1M, "Internal_Test in General_1M_Library")
#evaluate_loader(Internal_test_loader, library_2M_features, general_smiles_2M, "Internal_Test in General_2M_Library")

evaluate_loader(External_nist_filtered_by_CHONF_test_test_loader, library_nist_filter_by_CHONF_features, general_smiles_nist_filter_by_CHONF, "External_Test in General_nist_filter_by_CHONF_Library")
evaluate_loader(External_nist_test_loader, library_nist_features, general_smiles_nist, "External_Test in General_nist_Library")
evaluate_loader(External_nist_test_loader, library_4k_features, general_smiles_4k, "External_Test in General_4k_Library")
evaluate_loader(External_nist_test_loader, library_200k_features, general_smiles_200k,
                "External_Test in General_200k_Library")
evaluate_loader(External_nist_test_loader, library_1M_features, general_smiles_1M, "External_Test in General_1M_Library")
evaluate_loader(External_nist_test_loader, library_2M_features, general_smiles_2M, "External_Test in General_2M_Library")