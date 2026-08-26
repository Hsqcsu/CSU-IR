"""CHONF Subset Extraction & SMILES Replacement Script

Pipeline Overview:
  1. Loads reconstructed datasets from `CSU-IR/data/EB_dataset/data_with_NIST_IR`.
  2. Uses robust right-to-left splitting (`rsplit(None, 3)`) to parse tags:
     - Column -3: CHONF status (CHONF_True / CHONF_False)
     - Column -2: SMILES replacement status (None / CHONF_replace_smiles_xxx)
     - Column -1: PS status (PS_True / PS_False)
  3. Filters samples with `CHONF_True` and performs necessary SMILES substitutions.
  4. Slices corresponding IR spectral tensors and exports results to `CSU-IR/data/EB_dataset/EB_CHONF`.
"""

import os
import torch


def extract_chonf_data(splits=None):
    if splits is None:
        splits = ["train", "val", "test"]

    # ================= 1. Relative Path Settings =================
    # Locate current script directory: CSU-IR/EB_dataset_construction/
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up one level to CSU-IR/ and locate CSU-IR/data/EB_dataset/
    csu_ir_dir = os.path.dirname(script_dir)
    data_base_dir = os.path.join(csu_ir_dir, "data", "Multi-staged_training_data", "Experimental_Benching_data", "EB")

    # Input directory (data with reconstructed NIST IR)
    input_dir = os.path.join(data_base_dir, "data_with_NIST_IR")

    # Output directory for CHONF subset
    output_dir = os.path.join(data_base_dir, "EB_CHONF")
    os.makedirs(output_dir, exist_ok=True)

    print("🚀 Launching CHONF Subset Extraction Pipeline...")
    print(f"Input Directory  : {input_dir}")
    print(f"Output Directory : {output_dir}")

    for split in splits:
        print(f"\n{'=' * 25} Processing [{split.upper()} Set] {'=' * 25}")

        sample_id_file = os.path.join(input_dir, f"{split}_labels.txt")
        smiles_file = os.path.join(input_dir, f"{split}_smiles.txt")
        pt_file = os.path.join(input_dir, f"{split}_ir.pt")

        out_sample_id_file = os.path.join(output_dir, f"{split}_labels_CHONF.txt")
        out_smiles_file = os.path.join(output_dir, f"{split}_smiles_CHONF.txt")
        out_pt_file = os.path.join(output_dir, f"{split}_ir_CHONF.pt")

        # Check if source files exist
        if not (
            os.path.exists(sample_id_file)
            and os.path.exists(smiles_file)
            and os.path.exists(pt_file)
        ):
            print(f"⚠️ Warning: Required files for [{split}] split not found. Skipping...")
            continue

        print(f"Loading {split} data files...")

        # ================= 2. Load Original Files =================
        with open(sample_id_file, "r", encoding="utf-8") as f:
            sample_lines = [line.strip() for line in f if line.strip()]

        with open(smiles_file, "r", encoding="utf-8") as f:
            smiles_lines = [line.strip() for line in f if line.strip()]

        # Load spectral tensor ([N, L] or [N, 1, L])
        ir_tensors = torch.load(pt_file, map_location="cpu")

        # Verify data alignment
        total_samples = len(sample_lines)
        assert (
            len(smiles_lines) == total_samples
        ), f"SMILES count ({len(smiles_lines)}) does not match label count ({total_samples})!"
        assert (
            len(ir_tensors) == total_samples
        ), f"IR Tensor count ({len(ir_tensors)}) does not match label count ({total_samples})!"

        print(
            f"Loaded {total_samples} samples. Filtering and processing according to rules..."
        )

        # ================= 3. Filtering and SMILES Replacement =================
        selected_indices = []
        processed_smiles = []
        processed_sample_lines = []
        replaced_count = 0

        for idx, (s_line, orig_smiles) in enumerate(
            zip(sample_lines, smiles_lines)
        ):
            # Split exactly the last 3 columns from the right: <CHONF> <Replace> <PS>
            parts = s_line.rsplit(maxsplit=3)
            if len(parts) < 4:
                continue

            chonf_status = parts[1]     # Column -3
            replace_status = parts[2]   # Column -2
            # ps_status = parts[3]      # Column -1

            # Rule 1: Only extract samples where CHONF status is CHONF_True
            if chonf_status == "CHONF_True":
                selected_indices.append(idx)
                processed_sample_lines.append(s_line)

                # Rule 2: If replacement tag exists, substitute the SMILES string
                if replace_status.startswith("CHONF_replace_smiles_"):
                    new_smiles = replace_status[len("CHONF_replace_smiles_"):]
                    processed_smiles.append(new_smiles)
                    replaced_count += 1
                else:
                    processed_smiles.append(orig_smiles)

        # ================= 4. Slice Tensor Based on Selected Indices =================
        selected_indices_tensor = torch.tensor(
            selected_indices, dtype=torch.long
        )
        filtered_ir_tensors = ir_tensors[selected_indices_tensor]

        # ================= 5. Save Processed Files =================
        print(f"Saving processed {split} CHONF files...")

        # Save processed SMILES
        with open(out_smiles_file, "w", encoding="utf-8") as f:
            for smi in processed_smiles:
                f.write(smi + "\n")

        # Save filtered label information
        with open(out_sample_id_file, "w", encoding="utf-8") as f:
            for s_info in processed_sample_lines:
                f.write(s_info + "\n")

        # Save sliced spectral tensors
        torch.save(filtered_ir_tensors, out_pt_file)

        print(f"✅ [{split.upper()}] Processing complete!")
        print(f"    - Total Original Samples       : {total_samples}")
        print(f"    - Extracted CHONF Samples      : {len(selected_indices)} ({len(selected_indices)/total_samples:.2%})")
        print(f"    - Replaced SMILES Count        : {replaced_count}")
        print(f"    - Filtered IR Tensor Shape     : {filtered_ir_tensors.shape}")

    print("\n" + "=" * 75)
    print(f"🎉 All splits processed successfully! Output saved to: {output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    extract_chonf_data(splits=["train", "val", "test"])
