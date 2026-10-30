"""
SMILES Gradient and In-Distribution (ID) Data Extraction Tool

Partition criteria (based on maximum Tanimoto similarity Max_Sim with reference dataset 1):
1. In-Distribution / Near-ID : Max_Sim >= 0.7        (High similarity / Near In-Distribution)
2. Near-OOD                  : 0.5 <= Max_Sim < 0.7 (Near Out-of-Distribution, structurally close)
3. Moderate-OOD              : 0.3 <= Max_Sim < 0.5 (Moderate Out-of-Distribution, moderate structural difference)
4. Far-OOD                   : Max_Sim < 0.3        (Far Out-of-Distribution, significant structural difference)
"""

import os
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import torch

# Project path configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
print(f"Project Root: {PROJECT_ROOT}")


# ==================== Configuration ====================

# Pair 1 files (Reference distribution for Dataset 1: EB train + val + test)
PAIR1_FILES = [
    (
        os.path.join(PROJECT_ROOT, 'data', 'Multi-staged_training_data', 'Experimental_Benching_data', 'EB', 'data_with_NIST_IR', 'test_smiles.txt'),
       os.path.join(PROJECT_ROOT, 'data', 'Multi-staged_training_data', 'Experimental_Benching_data', 'EB', 'data_with_NIST_IR', 'test_ir.pt'),
    ),
    (
        os.path.join(PROJECT_ROOT, 'data', 'Multi-staged_training_data', 'Experimental_Benching_data', 'EB', 'data_with_NIST_IR', 'val_smiles.txt'),
       os.path.join(PROJECT_ROOT, 'data', 'Multi-staged_training_data', 'Experimental_Benching_data', 'EB', 'data_with_NIST_IR', 'val_ir.pt'),
    ),
    (
        os.path.join(PROJECT_ROOT, 'data', 'Multi-staged_training_data', 'Experimental_Benching_data', 'EB', 'data_with_NIST_IR', 'train_smiles.txt'),
       os.path.join(PROJECT_ROOT, 'data', 'Multi-staged_training_data', 'Experimental_Benching_data', 'EB', 'data_with_NIST_IR', 'train_ir.pt'),
    ),
]

# Pair 2 target data sources to be extracted (Merged file)
TARGET_SOURCE_FILE_TXT = os.path.join(PROJECT_ROOT, 'data', 'Multi-staged_training_data', 'Experimental_Benching_data', 'EB_TF', 'test_labels.txt')
TARGET_SMILES_TXT = os.path.join(PROJECT_ROOT, 'data', 'Multi-staged_training_data', 'Experimental_Benching_data', 'EB_TF', 'test_smiles.txt')
TARGET_PT_FILE = os.path.join(PROJECT_ROOT, 'data', 'Multi-staged_training_data', 'Experimental_Benching_data', 'EB_TF', 'test_ir.pt')

# Part II Target data identification tags (First column)
TARGET_TAGS = {"TF", "THERMOFISHER"}

# Output directory
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'OOD_data')
PAIR2_NAME = "thermo"

# ========================================================


def build_data1_fingerprints(pair1_files):
    """Build the Morgan fingerprint library for Dataset 1 (EB reference)."""
    data1_fps = []
    print("Building Morgan fingerprint library for Dataset 1 (train + val + test)...")

    for txt_path, _ in pair1_files:
        if not os.path.exists(txt_path):
            print(f"  Warning: File does not exist, skipping -> {txt_path}")
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            smiles_list = [line.strip() for line in f if line.strip()]

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            # Extract ECFP4 fingerprint (Radius=2, 2048 bits)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            data1_fps.append(fp)

    print(f"Dataset 1 reference fingerprint library built: Total fingerprints = {len(data1_fps)}")
    return data1_fps


def load_target_tf_data(source_path, smiles_path, pt_path):
    """Extract only target data belonging to TF / ThermoFisher from the merged file."""
    print(f"\nExtracting {PAIR2_NAME} target data from the merged dataset...")

    with open(source_path, "r", encoding="utf-8") as f:
        source_lines = [line.strip() for line in f if line.strip()]

    with open(smiles_path, "r", encoding="utf-8") as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    data_tensor = torch.load(pt_path, map_location="cpu")

    # Verify data lengths
    n_samples = len(smiles_list)
    assert len(source_lines) == n_samples, (
        f"Row count mismatch: source={len(source_lines)}, smiles={n_samples}"
    )
    assert data_tensor.shape[0] == n_samples, (
        f"Count mismatch: smiles={n_samples}, pt={data_tensor.shape[0]}"
    )

    # Filter indices belonging to TF / ThermoFisher
    target_indices = []
    for idx, line in enumerate(source_lines):
        first_col = line.split()[0]
        if first_col.upper() in TARGET_TAGS:
            target_indices.append(idx)

    target_smiles = [smiles_list[i] for i in target_indices]
    target_tensor = data_tensor[target_indices]

    print(f"  Total samples in merged data: {n_samples}")
    print(
        f"  Successfully filtered {PAIR2_NAME} data: {len(target_smiles)} samples "
        f"(Proportion: {len(target_smiles)/n_samples*100:.2f}%)"
    )

    return target_smiles, target_tensor


def extract_gradient_and_id_data(target_smiles, target_tensor, data1_fps):
    """Partition gradient OOD data based on Tanimoto similarity."""
    print("\nStarting similarity gradient partitioning for target data...")

    # Container definitions
    id_smiles, id_tensors = [], []  # Max_Sim >= 0.7
    near_smiles, near_tensors = [], []  # [0.5, 0.7)
    mod_smiles, mod_tensors = [], []  # [0.3, 0.5)
    far_smiles, far_tensors = [], []  # < 0.3

    total_scanned = 0
    valid_count = 0

    for idx, smi in enumerate(target_smiles):
        total_scanned += 1
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        valid_count += 1
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        sims = DataStructs.BulkTanimotoSimilarity(fp, data1_fps)
        max_sim = max(sims) if sims else 0.0

        # ---------------- Similarity Interval Conditions ----------------
        if max_sim >= 0.7:
            id_smiles.append(smi)
            id_tensors.append(target_tensor[idx])
        elif 0.5 <= max_sim < 0.7:
            near_smiles.append(smi)
            near_tensors.append(target_tensor[idx])
        elif 0.3 <= max_sim < 0.5:
            mod_smiles.append(smi)
            mod_tensors.append(target_tensor[idx])
        elif max_sim < 0.3:
            far_smiles.append(smi)
            far_tensors.append(target_tensor[idx])

    print(f"\nScan completed (Total scanned: {total_scanned}, Valid molecules: {valid_count}):")
    print(f"  1. High-Sim / ID   (Sim >= 0.7)      : {len(id_smiles)} samples")
    print(f"  2. Near-OOD        (0.5 <= Sim < 0.7): {len(near_smiles)} samples")
    print(f"  3. Moderate-OOD    (0.3 <= Sim < 0.5): {len(mod_smiles)} samples")
    print(f"  4. Far-OOD         (Sim < 0.3)       : {len(far_smiles)} samples")

    results = {
        "id_data": (
            id_smiles,
            torch.stack(id_tensors) if id_tensors else torch.empty(0),
        ),
        "near_ood": (
            near_smiles,
            torch.stack(near_tensors) if near_tensors else torch.empty(0),
        ),
        "moderate_ood": (
            mod_smiles,
            torch.stack(mod_tensors) if mod_tensors else torch.empty(0),
        ),
        "far_ood": (
            far_smiles,
            torch.stack(far_tensors) if far_tensors else torch.empty(0),
        ),
    }
    return results


def save_results(results, output_dir):
    """Save the partitioned groups of data."""
    os.makedirs(output_dir, exist_ok=True)

    file_mapping = [
        ("id_data", f"id_sim_gt0.7_{PAIR2_NAME}"),
        ("near_ood", f"ood_near_sim0.5-0.7_{PAIR2_NAME}"),
        ("moderate_ood", f"ood_moderate_sim0.3-0.5_{PAIR2_NAME}"),
        ("far_ood", f"ood_far_sim_lt0.3_{PAIR2_NAME}"),
    ]

    print(f"\nSaving partitioned data to: {output_dir}")
    for key, prefix in file_mapping:
        smiles, tensor_data = results[key]
        if len(smiles) > 0:
            txt_path = os.path.join(output_dir, f"{prefix}_smiles.txt")
            pt_path = os.path.join(output_dir, f"{prefix}_data.pt")

            with open(txt_path, "w", encoding="utf-8") as f:
                for smi in smiles:
                    f.write(smi + "\n")

            torch.save(tensor_data, pt_path)
            print(f"  Saved {key}: {len(smiles)} samples -> {txt_path}")
        else:
            print(f"  Note: No matching data found for {key}, skipping save.")


if __name__ == "__main__":
    # 1. Build the reference fingerprint library for Dataset 1 (EB data)
    data1_fps = build_data1_fingerprints(PAIR1_FILES)

    # 2. Extract TF / ThermoFisher data from the merged file
    target_smiles, target_tensor = load_target_tf_data(
        TARGET_SOURCE_FILE_TXT, TARGET_SMILES_TXT, TARGET_PT_FILE
    )

    # 3. Perform ID and gradient OOD similarity partitioning
    results = extract_gradient_and_id_data(
        target_smiles, target_tensor, data1_fps
    )

    # 4. Save results
    save_results(results, OUTPUT_DIR)
