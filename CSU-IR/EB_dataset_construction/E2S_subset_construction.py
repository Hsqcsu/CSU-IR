import os
import torch
import torch.nn.functional as F
from rdkit import Chem
from tqdm import tqdm

# =====================================================================
# 1. Path Configuration
# =====================================================================
PATH_CONFIG = {
    # Dataset 1 (contains train, val, and test splits)
    "ds1_train_smiles": r"F:\Spectrum\1122_after\model\ESA_model_sigmoid\20250530_esa_ir_CNN_transformer\_20260602_MG_training\data\processed_data\comparison\exp\1_2_3_4_5_8_9_splitted_and_augmented_data_absortion\train_description_augmented.txt",
    "ds1_train_ir": r"F:\Spectrum\1122_after\model\ESA_model_sigmoid\20250530_esa_ir_CNN_transformer\_20260602_MG_training\data\processed_data\comparison\exp\1_2_3_4_5_8_9_splitted_and_augmented_data_absortion\train_ir.pt",

    "ds1_val_smiles": r"F:\Spectrum\1122_after\model\ESA_model_sigmoid\20250530_esa_ir_CNN_transformer\_20260602_MG_training\data\processed_data\comparison\exp\1_2_3_4_5_8_9_splitted_and_augmented_data_absortion\val_description_augmented.txt",
    "ds1_val_ir": r"F:\Spectrum\1122_after\model\ESA_model_sigmoid\20250530_esa_ir_CNN_transformer\_20260602_MG_training\data\processed_data\comparison\exp\1_2_3_4_5_8_9_splitted_and_augmented_data_absortion\val_ir.pt",

    "ds1_test_smiles": r"F:\Spectrum\1122_after\model\ESA_model_sigmoid\20250530_esa_ir_CNN_transformer\_20260602_MG_training\data\processed_data\comparison\exp\1_2_3_4_5_8_9_splitted_and_augmented_data_absortion\test_description_augmented.txt",
    "ds1_test_ir": r"F:\Spectrum\1122_after\model\ESA_model_sigmoid\20250530_esa_ir_CNN_transformer\_20260602_MG_training\data\processed_data\comparison\exp\1_2_3_4_5_8_9_splitted_and_augmented_data_absortion\test_ir.pt",

    # Dataset 2 (combines QM9S parts 1-3 and QMe14S)
    "ds2_sources": [
        # QM9S Part 1-3
        (r'E:\Spectrum\data\DetaNet_QM9S_13w\raw_ir_and_raman_3500\processed_smiles_part1.txt',
         r'E:\Spectrum\data\DetaNet_QM9S_13w\raw_ir_and_raman_3500\processed_ir_part1.pt',
         "QM9S-Part1"),
        (r'E:\Spectrum\data\DetaNet_QM9S_13w\raw_ir_and_raman_3500\processed_smiles_part2.txt',
         r'E:\Spectrum\data\DetaNet_QM9S_13w\raw_ir_and_raman_3500\processed_ir_part2.pt',
         "QM9S-Part2"),
        (r'E:\Spectrum\data\DetaNet_QM9S_13w\raw_ir_and_raman_3500\processed_smiles_part3.txt',
         r'E:\Spectrum\data\DetaNet_QM9S_13w\raw_ir_and_raman_3500\processed_ir_part3.pt',
         "QM9S-Part3"),
        # QMe14S
        (r"E:\Spectrum\data\IR_broaden_process_and_data_QMe14S\raw_data\all_smiles_normalized_processed.txt",
         r"E:\Spectrum\data\IR_broaden_process_and_data_QMe14S\raw_data\all_spectra_processed.pt",
         "QMe14S")
    ],

    # Output directory for results
    "output_dir": r"F:\Spectrum\1122_after\model\ESA_model_sigmoid\20250530_esa_ir_CNN_transformer\_20260602_MG_training\data\processed_data\comparison\exp\7_E2S"
}


# =====================================================================
# 2. SMILES Canonicalization without Stereochemistry
# =====================================================================
def normalize_smiles_no_stereo(smiles):
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # isomericSmiles=False strips chiral centers (@, @@) and cis/trans isomerism (/, \)
            return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    except Exception:
        pass
    return None


# =====================================================================
# 3. Retrieval Evaluation Helper Function
# =====================================================================
def evaluate_ir_retrieval_with_distractors(ds1_ir, ds2_ir, query_indices, top_k=[1, 5, 10]):
    """
    Compute retrieval metrics, excluding seen/optimized molecules from acting as queries.
    ds1_ir: [N, 3500] Full intersection IR spectra of Dataset 1
    ds2_ir: [N, 3500] Full intersection IR spectra of Dataset 2 (including train and val sets as distractors)
    query_indices: Index list of molecules belonging strictly to the test set
    """
    num_queries = len(query_indices)
    if num_queries == 0:
        print("Error: No valid test query molecules found!")
        return {k: 0.0 for k in top_k}

    # L2 normalize features to support cosine similarity computation
    ds1_norm = F.normalize(ds1_ir, p=2, dim=-1)
    ds2_norm = F.normalize(ds2_ir, p=2, dim=-1)

    # Extract test set query feature vectors [Q, 3500]
    query_norm = ds1_norm[query_indices]

    # Compute cosine similarity matrix between queries and the entire gallery [Q, N]
    sim_matrix = torch.matmul(query_norm, ds2_norm.T)

    # Sort indices in descending order
    sorted_indices = torch.argsort(sim_matrix, dim=-1, descending=True)

    correct_counts = {k: 0 for k in top_k}

    # Evaluation phase
    for i, global_query_idx in enumerate(query_indices):
        retrieved_ranks = sorted_indices[i]  # Get ranking order for the i-th query in the full gallery

        # Target physical index is global_query_idx
        rank = (retrieved_ranks == global_query_idx).nonzero(as_tuple=True)[0].item() + 1

        for k in top_k:
            if rank <= k:
                correct_counts[k] += 1

    recalls = {k: correct_counts[k] / num_queries for k in top_k}
    return recalls


# =====================================================================
# 4. Main Data Loading and Processing Pipeline
# =====================================================================
def main():
    os.makedirs(PATH_CONFIG["output_dir"], exist_ok=True)

    # ------------------ Pre-load Train / Val SMILES of Dataset 1 (as known exclusion sets) ------------------
    print("\nExtracting known/optimized molecule sets from Dataset 1 Train and Val splits...")

    ds1_train_normalized_set = set()
    with open(PATH_CONFIG["ds1_train_smiles"], 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                norm_s = normalize_smiles_no_stereo(line.strip().split()[0])
                if norm_s:
                    ds1_train_normalized_set.add(norm_s)

    ds1_val_normalized_set = set()
    with open(PATH_CONFIG["ds1_val_smiles"], 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                norm_s = normalize_smiles_no_stereo(line.strip().split()[0])
                if norm_s:
                    ds1_val_normalized_set.add(norm_s)

    print(f"--> Dataset 1 Train unique molecules: {len(ds1_train_normalized_set)}")
    print(f"--> Dataset 1 Val unique molecules: {len(ds1_val_normalized_set)}")

    # ------------------ Process Dataset 1 (Merge Train, Val, Test) ------------------
    print("\n[Step 1/5] Loading and merging Dataset 1 (Train/Val/Test)...")
    ds1_parts = [
        (PATH_CONFIG["ds1_train_smiles"], PATH_CONFIG["ds1_train_ir"], "Train"),
        (PATH_CONFIG["ds1_val_smiles"], PATH_CONFIG["ds1_val_ir"], "Val"),
        (PATH_CONFIG["ds1_test_smiles"], PATH_CONFIG["ds1_test_ir"], "Test")
    ]

    db1 = {}  # Structure: {normalized_smiles: ir_tensor_slice}
    total_ds1_raw_count = 0

    for smiles_path, ir_path, part_name in ds1_parts:
        with open(smiles_path, 'r', encoding='utf-8') as f:
            part_raw_smiles = [line.strip().split()[0] for line in f if line.strip()]
        part_ir_tensor = torch.load(ir_path)

        assert len(part_raw_smiles) == part_ir_tensor.shape[0], f"Dataset 1 {part_name} rows and matrix dimensions do not match!"
        total_ds1_raw_count += len(part_raw_smiles)

        for idx, s in enumerate(tqdm(part_raw_smiles, desc=f"Processing Dataset 1-{part_name}")):
            norm_s = normalize_smiles_no_stereo(s)
            # If duplicated, retain the first encountered IR spectrum
            if norm_s and norm_s not in db1:
                db1[norm_s] = part_ir_tensor[idx]

    print(f"--> Dataset 1 processing complete. Total raw samples: {total_ds1_raw_count}, Deduplicated unique molecules: {len(db1)}")

    # ------------------ Process Dataset 2 (Merge QM9S 1-3 and QMe14S) ------------------
    print("\n[Step 2/5] Loading and merging Dataset 2 (QM9S + QMe14S)...")
    db2 = {}  # Structure: {normalized_smiles: ir_tensor_slice}
    total_ds2_raw_count = 0

    for smiles_path, ir_path, part_name in PATH_CONFIG["ds2_sources"]:
        with open(smiles_path, 'r', encoding='utf-8') as f:
            part_raw_smiles = [line.strip().split()[0] for line in f if line.strip()]
        part_ir_tensor = torch.load(ir_path)

        assert len(part_raw_smiles) == part_ir_tensor.shape[0], f"Dataset 2 {part_name} rows and matrix dimensions do not match!"
        total_ds2_raw_count += len(part_raw_smiles)

        for idx, s in enumerate(tqdm(part_raw_smiles, desc=f"Processing Dataset 2-{part_name}")):
            norm_s = normalize_smiles_no_stereo(s)
            if norm_s and norm_s not in db2:
                db2[norm_s] = part_ir_tensor[idx]

    print(f"--> Dataset 2 processing complete. Total raw samples: {total_ds2_raw_count}, Deduplicated unique molecules: {len(db2)}")

    # ------------------ Find and Extract Intersection ------------------
    print("\n[Step 3/5] Computing chemical structure intersection between both datasets...")
    intersection_smiles = sorted(list(set(db1.keys()) & set(db2.keys())))
    intersection_count = len(intersection_smiles)

    if intersection_count == 0:
        print("Warning: No chemical intersection found between the two datasets! Please check paths and data.")
        return

    print(f"--> Done. Extracted {intersection_count} intersecting molecules in total.")

    # Extract corresponding IR spectra
    intersection_ds1_ir_list = []
    intersection_ds2_ir_list = []

    for s in intersection_smiles:
        intersection_ds1_ir_list.append(db1[s])
        intersection_ds2_ir_list.append(db2[s])

    # Reconstruct into [N, 3500] dimensional Tensors
    final_ds1_ir = torch.stack(intersection_ds1_ir_list, dim=0)
    final_ds2_ir = torch.stack(intersection_ds2_ir_list, dim=0)

    # ------------------ Save Extracted Intersection Data ------------------
    print("\n[Step 4/5] Saving extracted intersection data...")

    out_smiles_path = os.path.join(PATH_CONFIG["output_dir"], "intersection_smiles.txt")
    out_ds1_ir_path = os.path.join(PATH_CONFIG["output_dir"], "intersection_ds1_ir.pt")
    out_ds2_ir_path = os.path.join(PATH_CONFIG["output_dir"], "intersection_ds2_ir.pt")

    # 1. Save global intersection data
    with open(out_smiles_path, 'w', encoding='utf-8') as f:
        for s in intersection_smiles:
            f.write(f"{s}\n")
    torch.save(final_ds1_ir, out_ds1_ir_path)
    torch.save(final_ds2_ir, out_ds2_ir_path)

    # 2. Split into three non-overlapping subsets (Test / Train / Val) and save separately
    test_smiles, test_ds1_list, test_ds2_list = [], [], []
    train_smiles, train_ds1_list, train_ds2_list = [], [], []
    val_smiles, val_ds1_list, val_ds2_list = [], [], []

    for idx, s in enumerate(intersection_smiles):
        if s in ds1_train_normalized_set:
            train_smiles.append(s)
            train_ds1_list.append(final_ds1_ir[idx])
            train_ds2_list.append(final_ds2_ir[idx])
        elif s in ds1_val_normalized_set:
            val_smiles.append(s)
            val_ds1_list.append(final_ds1_ir[idx])
            val_ds2_list.append(final_ds2_ir[idx])
        else:
            test_smiles.append(s)
            test_ds1_list.append(final_ds1_ir[idx])
            test_ds2_list.append(final_ds2_ir[idx])

    # A. Save Test subset
    if test_smiles:
        test_ds1_ir = torch.stack(test_ds1_list, dim=0)
        test_ds2_ir = torch.stack(test_ds2_list, dim=0)
        torch.save(test_ds1_ir, os.path.join(PATH_CONFIG["output_dir"], "test_intersection_ds1_ir.pt"))
        torch.save(test_ds2_ir, os.path.join(PATH_CONFIG["output_dir"], "test_intersection_ds2_ir.pt"))
        with open(os.path.join(PATH_CONFIG["output_dir"], "test_intersection_smiles.txt"), 'w', encoding='utf-8') as f:
            for s in test_smiles:
                f.write(f"{s}\n")

    # B. Save Train subset (distractors)
    if train_smiles:
        train_ds1_ir = torch.stack(train_ds1_list, dim=0)
        train_ds2_ir = torch.stack(train_ds2_list, dim=0)
        torch.save(train_ds1_ir, os.path.join(PATH_CONFIG["output_dir"], "train_intersection_ds1_ir.pt"))
        torch.save(train_ds2_ir, os.path.join(PATH_CONFIG["output_dir"], "train_intersection_ds2_ir.pt"))
        with open(os.path.join(PATH_CONFIG["output_dir"], "train_intersection_smiles.txt"), 'w', encoding='utf-8') as f:
            for s in train_smiles:
                f.write(f"{s}\n")

    # C. Save Val subset (distractors)
    if val_smiles:
        val_ds1_ir = torch.stack(val_ds1_list, dim=0)
        val_ds2_ir = torch.stack(val_ds2_list, dim=0)
        torch.save(val_ds1_ir, os.path.join(PATH_CONFIG["output_dir"], "val_intersection_ds1_ir.pt"))
        torch.save(val_ds2_ir, os.path.join(PATH_CONFIG["output_dir"], "val_intersection_ds2_ir.pt"))
        with open(os.path.join(PATH_CONFIG["output_dir"], "val_intersection_smiles.txt"), 'w', encoding='utf-8') as f:
            for s in val_smiles:
                f.write(f"{s}\n")

    print("\n" + "=" * 60)
    print(" Data extraction and saving complete!")
    print(f" -> Total intersection molecule count: {intersection_count}")
    print(f" -> [Saved] Test intersection molecules (Query evaluation): {len(test_smiles)}")
    print(f" -> [Saved] Train intersection molecules (Training distractors): {len(train_smiles)}")
    print(f" -> [Saved] Val intersection molecules (Validation distractors): {len(val_smiles)}")
    print("=" * 60 + "\n")

    # ------------------ Run IR Retrieval Evaluation ------------------
    print("[Step 5/5] Running Dataset 1 IR -> Dataset 2 IR cosine retrieval evaluation (excluding train/val queries, kept as gallery distractors)...")

    # Keep only molecules belonging to the unseen test set as queries
    query_indices = []
    for idx, s in enumerate(intersection_smiles):
        if s not in ds1_train_normalized_set and s not in ds1_val_normalized_set:
            query_indices.append(idx)

    num_seen_distractors = len(intersection_smiles) - len(query_indices)

    # Perform retrieval evaluation
    recalls = evaluate_ir_retrieval_with_distractors(
        final_ds1_ir, final_ds2_ir, query_indices, top_k=[1, 5, 10]
    )

    print("\n" + "=" * 20 + " IR Retrieval Evaluation with Distractors Report " + "=" * 20)
    print(f" Evaluation Query Size   : {len(query_indices)} (Excluded {num_seen_distractors} train and val molecules)")
    print(f" Gallery Size            : {intersection_count} (Train and val molecules retained as distractors)")
    print(f"   Recall@1  (Top-1  Accuracy): {recalls[1]:.4%}")
    print(f"   Recall@5  (Top-5  Accuracy): {recalls[5]:.4%}")
    print(f"   Recall@10 (Top-10 Accuracy): {recalls[10]:.4%}")
    print("=" * 86 + "\n")


if __name__ == "__main__":
    main()
