"""PS (Polystyrene / Phase-Selected) Subset Extraction Script

Pipeline Overview:
  1. Loads reconstructed datasets from `CSU-IR/data/EB_dataset/data_with_NIST_IR`.
  2. Reads `{split}_ps_subset_labels.txt`, where each line is either 'YES' or 'NO'.
  3. Filters samples whose PS label is 'YES'.
  4. Slices corresponding IR tensors, SMILES, and labels, then saves outputs to `PS_subset`.
"""

import os
from pathlib import Path
import torch


def extract_ps_data(splits=None):
    if splits is None:
        splits = ["train", "val", "test"]

    # ================= 1. Relative Path Settings =================
    script_dir = Path(__file__).resolve().parent
    csu_ir_dir = script_dir.parent
    data_base_dir = (
        csu_ir_dir
        / "data"
        / "Multi-staged_training_data"
        / "Experimental_Benching_data"
        / "EB"
    )

    # Input directory (data with reconstructed NIST IR)
    input_dir = data_base_dir / "data_with_NIST_IR"

    # Output directory for PS subset
    output_dir = data_base_dir / "PS_subset"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Launching PS Subset Extraction Pipeline...")
    print(f"Input Directory  : {input_dir}")
    print(f"Output Directory : {output_dir}\n" + "=" * 70)

    for split in splits:
        print(f"\n================ Processing [{split.upper()} Set] ================")

        ps_label_file = input_dir / f"{split}_ps_subset_labels.txt"
        sample_id_file = input_dir / f"{split}_labels.txt"
        smiles_file = input_dir / f"{split}_smiles.txt"
        pt_file = input_dir / f"{split}_ir.pt"

        out_sample_id_file = output_dir / f"{split}_labels_ps.txt"
        out_smiles_file = output_dir / f"{split}_smiles_ps.txt"
        out_pt_file = output_dir / f"{split}_ir_ps.pt"

        # Check if source files exist
        if not (
            ps_label_file.exists()
            and sample_id_file.exists()
            and smiles_file.exists()
            and pt_file.exists()
        ):
            print(f"Warning: Required files for [{split}] split not found. Skipping...")
            continue

        print(f"Loading {split} data files...")

        # ================= 2. Load Original Files =================
        with open(ps_label_file, "r", encoding="utf-8") as f:
            ps_flags = [line.strip().upper() for line in f if line.strip()]

        with open(sample_id_file, "r", encoding="utf-8") as f:
            sample_lines = [line.strip() for line in f if line.strip()]

        with open(smiles_file, "r", encoding="utf-8") as f:
            smiles_lines = [line.strip() for line in f if line.strip()]

        ir_tensors = torch.load(pt_file, map_location="cpu")

        # Verify data alignment
        total_samples = len(sample_lines)
        assert (
            len(ps_flags) == total_samples
        ), f"PS flags count ({len(ps_flags)}) does not match label count ({total_samples})!"
        assert (
            len(smiles_lines) == total_samples
        ), f"SMILES count ({len(smiles_lines)}) does not match label count ({total_samples})!"
        assert (
            len(ir_tensors) == total_samples
        ), f"IR Tensor count ({len(ir_tensors)}) does not match label count ({total_samples})!"

        print(f"Loaded {total_samples} samples. Filtering based on 'YES' flag...")

        # ================= 3. Filtering by 'YES' Flag =================
        selected_indices = [
            idx for idx, flag in enumerate(ps_flags) if flag == "YES"
        ]

        if not selected_indices:
            print(f"Warning: No samples marked 'YES' in [{split}] split. Skipping saving.")
            continue

        processed_sample_lines = [sample_lines[i] for i in selected_indices]
        processed_smiles = [smiles_lines[i] for i in selected_indices]

        # ================= 4. Slice Spectral Tensor =================
        selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long)
        filtered_ir_tensors = ir_tensors[selected_indices_tensor]

        # ================= 5. Save Extracted Files =================
        print(f"Saving extracted {split} PS subset files...")

        with open(out_smiles_file, "w", encoding="utf-8") as f:
            for smi in processed_smiles:
                f.write(smi + "\n")

        with open(out_sample_id_file, "w", encoding="utf-8") as f:
            for s_info in processed_sample_lines:
                f.write(s_info + "\n")

        torch.save(filtered_ir_tensors, out_pt_file)

        print(f"Successfully processed and wrote [{split.upper()}] dataset:")
        print(f"    - Total Initial Samples        : {total_samples}")
        print(f"    - Extracted PS Samples ('YES') : {len(selected_indices)} ({len(selected_indices)/total_samples:.2%})")
        print(f"    - Filtered IR Tensor Shape     : {filtered_ir_tensors.shape}")
        print(f"    - Output Directory             : {output_dir}")

    print("\n" + "=" * 70)
    print(f"All splits processed successfully! Saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    extract_ps_data(splits=["train", "val", "test"])
