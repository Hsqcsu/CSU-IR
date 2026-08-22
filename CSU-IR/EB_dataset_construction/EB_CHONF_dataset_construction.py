import os
import torch

def extract_chonf_data():
    # ================= 1. Path Settings =================
    base_dir = r"F:\Spectrum\1122_after\model\ESA_model_sigmoid\20250530_esa_ir_CNN_transformer\_20260602_MG_training\data\data_process\optimization\exp_20260817\0_1_2_4_5_6_8_9_16_splitted_and_augmented_data_delete_299"
    
    sample_id_file = os.path.join(base_dir, "test_source_with_sample_ids_annotated.txt")
    smiles_file = os.path.join(base_dir, "test_description_augmented.txt")
    pt_file = os.path.join(base_dir, "test_ir.pt")

    # Output directory (will be created automatically)
    output_dir = os.path.join(base_dir, "CHONF_extracted")
    os.makedirs(output_dir, exist_ok=True)
    
    out_sample_id_file = os.path.join(output_dir, "test_source_chonf.txt")
    out_smiles_file = os.path.join(output_dir, "test_description_chonf.txt")
    out_pt_file = os.path.join(output_dir, "test_ir_chonf.pt")

    print("Loading data files...")
    
    # ================= 2. Load Original Files =================
    with open(sample_id_file, 'r', encoding='utf-8') as f:
        sample_lines = [line.strip() for line in f if line.strip()]
        
    with open(smiles_file, 'r', encoding='utf-8') as f:
        smiles_lines = [line.strip() for line in f if line.strip()]

    # Load spectral tensor (usually with shape [N, L] or [N, 1, L])
    ir_tensors = torch.load(pt_file)

    # Check data alignment
    total_samples = len(sample_lines)
    assert len(smiles_lines) == total_samples, f"SMILES count ({len(smiles_lines)}) does not match Sample ID count ({total_samples})!"
    assert len(ir_tensors) == total_samples, f"PT Tensor count ({len(ir_tensors)}) does not match Sample ID count ({total_samples})!"

    print(f"Loaded {total_samples} samples in total. Starting filtering and processing according to rules...")

    # ================= 3. Filtering and SMILES Replacement =================
    selected_indices = []
    processed_smiles = []
    processed_sample_lines = []

    for idx, (s_line, orig_smiles) in enumerate(zip(sample_lines, smiles_lines)):
        parts = s_line.split()
        if len(parts) < 4:
            continue
        
        # Extract relevant columns
        # Example: NIST B6000103_IR_0 CHONF_True None PS_False
        chonf_tag = parts[2]       # Column 3: CHONF_True / CHONF_False
        replace_tag = parts[3]     # Column 4: None / CHONF_replace_smiles_xxx

        # Rule 1: Only extract samples where column 3 is CHONF_True
        if chonf_tag == "CHONF_True":
            selected_indices.append(idx)
            processed_sample_lines.append(s_line)

            # Rule 2: If column 4 is not None, replace the SMILES string
            if replace_tag != "None" and "CHONF_replace_smiles_" in replace_tag:
                new_smiles = replace_tag.replace("CHONF_replace_smiles_", "")
                processed_smiles.append(new_smiles)
            else:
                processed_smiles.append(orig_smiles)

    # ================= 4. Slice Tensor Based on Selected Indices =================
    selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long)
    filtered_ir_tensors = ir_tensors[selected_indices_tensor]

    # ================= 5. Save Processed Files =================
    print("Saving processed files...")
    
    # Save the updated SMILES
    with open(out_smiles_file, 'w', encoding='utf-8') as f:
        for smi in processed_smiles:
            f.write(smi + '\n')

    # Save the filtered Sample ID information
    with open(out_sample_id_file, 'w', encoding='utf-8') as f:
        for s_info in processed_sample_lines:
            f.write(s_info + '\n')

    # Save the extracted spectral tensor
    torch.save(filtered_ir_tensors, out_pt_file)

    print("Processing complete!")
    print(f"Original samples count: {total_samples}")
    print(f"Extracted CHONF samples count: {len(selected_indices)}")
    print(f"Filtered spectral tensor shape: {filtered_ir_tensors.shape}")
    print(f"Output files saved to: {output_dir}")

if __name__ == "__main__":
    extract_chonf_data()
