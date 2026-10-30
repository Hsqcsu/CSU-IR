"""EB_CHONF Dataset Extraction and Multi-Stage Cleaning Script

Pipeline Overview:
  1. Base directory: CSU-IR relative path -> `data_with_NIST_IR`.
  2. Stage 1 (Elemental Filter): Validates molecules with RDKit and strictly retains CHONF elements.
  3. Stage 2 (Quality Filter): Drops multi-component/salt molecules (containing '.') and blacklist SMILES.
  4. Slices IR spectral tensors, SMILES, and labels synchronously, outputting clean files to `EB_CHONF`.
"""

import os
from pathlib import Path
import torch
from rdkit import Chem

# Allowed chemical element set
ALLOWED_ELEMENTS = {"C", "H", "O", "N", "F"}

# Specific blacklist SMILES to exclude per split (conformation generation failures or anomalous structures in SMEN and VibraCLIP data processing)
EXCLUDE_SMILES = {
    "train": {
        "CCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCC)OC(=O)CCCCCCCCCCC",
        "O=C(O)c1ccc(C(=O)O)c2c(C(=O)O)ccc(C(=O)O)c12",
        "CCCCCC[C@H](C/C=C\\CCCCCCCC(=O)OCC(COC(=O)CCCCCCC/C=C\\C[C@@H](CCCCCC)OC(C)=O)OC(=O)CCCCCCC/C=C\\C[C@@H](CCCCCC)OC(C)=O)OC(C)=O",
        "c1cc2ccc1CCc1ccc(cc1)CC2",
        "CCCCCCCCCCCCCCCCCC(=O)OC[C@@H](OC(=O)CCCCCCCCCCCCCCCCC)[C@H]1OC[C@H](O)[C@H]1OC(=O)CCCCCCCCCCCCCCCCC",
        "CCCCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCCCC",
        "CCCCCCCCCCCCN(CCCCCCCCCCCC)CCCCCCCCCCCC",
        "C1#CCCCCCCC#CCCCCC1",
        "CCCCCCCCCCCCCCCC(=O)OC[C@H]1O[C@H](O[C@]2(COC(=O)CCCCCCCCCCCCCCC)O[C@H](CO)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O",
        "CCCCCCCCC(=O)OC1C(C)(C)C(OC(=O)CCCCCCCC)C(C)(C)C(OC(=O)CCCCCCCC)C1(C)C",
        "c1(C)cccc2c1c1c(cc2)cccc1C",
    },
    "val": {
        "O=C(OC[C@H]1O[C@@H](OC(=O)c2cc(O)c(O)c(OC(=O)c3cc(O)c(O)c(O)c3)c2)[C@H](OC(=O)c2cc(O)c(O)c(OC(=O)c3cc(O)c(O)c(O)c3)c2)[C@@H](OC(=O)c2cc(O)c(O)c(OC(=O)c3cc(O)c(O)c(O)c3)c2)[C@@H]1OC(=O)c1cc(O)c(O)c(OC(=O)c2cc(O)c(O)c(O)c2)c1)c1cc(O)c(O)c(OC(=O)c2cc(O)c(O)c(O)c2)c1"
    },
    "test": {
        "CCCCCCCCCCOC(=O)c1ccc(C(=O)OCCCCCCCCCC)c(C(=O)OCCCCCCCCCC)c1",
        "CC(=O)OCCCCCCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCCCCCCOC(C)=O)OC(=O)CCCCCCCCCCCCCCCCCOC(C)=O",
    },
}


def is_chonf(smi):
    """Checks whether the molecule contains only C, H, O, N, and F elements."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ALLOWED_ELEMENTS:
            return False
    return True


def extract_chonf_dataset(splits=None):
    if splits is None:
        splits = ["train", "val", "test"]

    # ================= 1. Relative Path Configuration =================
    script_dir = Path(__file__).resolve().parent
    csu_ir_dir = script_dir.parent
    data_base_dir = (
        csu_ir_dir
        / "data"
        / "Multi-staged_training_data"
        / "Experimental_Benching_data"
        / "EB"
    )

    # Input directory (containing datasets with reconstructed NIST IR)
    input_dir = data_base_dir / "data_with_NIST_IR"
    # Output directory (cleaned EB_CHONF subset)
    output_dir = data_base_dir / "EB_CHONF"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Launching EB_CHONF Dataset Construction Pipeline...")
    print(f"Input Directory  : {input_dir}")
    print(f"Output Directory : {output_dir}\n" + "=" * 70)

    for split in splits:
        print(f"\n================ Processing [{split.upper()} Set] ================")

        labels_path = input_dir / f"{split}_labels.txt"
        smiles_path = input_dir / f"{split}_smiles.txt"
        ir_path = input_dir / f"{split}_ir.pt"

        if not (labels_path.exists() and smiles_path.exists() and ir_path.exists()):
            print(f"Warning: Missing required files for [{split}] split. Skipping...")
            continue

        # ----------------- Load Raw Source Data -----------------
        print(f"Loading raw files for [{split}]...")
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_list = [line.strip() for line in f if line.strip()]

        with open(smiles_path, "r", encoding="utf-8") as f:
            smiles_list = [line.strip() for line in f if line.strip()]

        ir_tensor = torch.load(ir_path, map_location="cpu")
        orig_count = len(smiles_list)

        assert (
            len(labels_list) == orig_count == len(ir_tensor)
        ), f"Sample count mismatch! Labels: {len(labels_list)}, SMILES: {orig_count}, IR: {len(ir_tensor)}"
        print(f"Initial total samples: {orig_count}")

        # ----------------- Step 1: Retain CHONF Elements Only -----------------
        print("-> Step 1: Filtering for valid CHONF molecules...")
        chonf_indices = []
        for idx, smi in enumerate(smiles_list):
            if is_chonf(smi):
                chonf_indices.append(idx)

        print(
            f"   Step 1 completed: Kept {len(chonf_indices)} / {orig_count} samples "
            f"(Dropped non-CHONF: {orig_count - len(chonf_indices)})"
        )

        # ----------------- Step 2: Remove Multi-Component & Blacklist Entries -----------------
        print("-> Step 2: Filtering out '.' entries and blacklist molecules...")
        current_exclude_set = EXCLUDE_SMILES.get(split, set())

        final_indices = []
        dot_drop_count = 0
        manual_drop_count = 0

        for idx in chonf_indices:
            smi = smiles_list[idx]
            if "." in smi:
                dot_drop_count += 1
            elif smi in current_exclude_set:
                manual_drop_count += 1
            else:
                final_indices.append(idx)

        print(f"   Entries dropped with '.': {dot_drop_count}")
        print(f"   Entries dropped via blacklist: {manual_drop_count}")
        print(
            f"   Final retained valid samples: {len(final_indices)} "
            f"({len(final_indices)/orig_count:.2%})"
        )

        # ----------------- Slice and Export Cleaned Files -----------------
        filtered_labels = [labels_list[i] for i in final_indices]
        filtered_smiles = [smiles_list[i] for i in final_indices]
        filtered_ir = ir_tensor[final_indices]

        out_labels_file = output_dir / f"{split}_labels.txt"
        out_smiles_file = output_dir / f"{split}_smiles.txt"
        out_ir_file = output_dir / f"{split}_ir.pt"

        with open(out_labels_file, "w", encoding="utf-8") as f:
            for lbl in filtered_labels:
                f.write(f"{lbl}\n")

        with open(out_smiles_file, "w", encoding="utf-8") as f:
            for smi in filtered_smiles:
                f.write(f"{smi}\n")

        torch.save(filtered_ir, out_ir_file)

        print(f"Successfully processed and wrote [{split.upper()}] dataset:")
        print(f"    - IR Tensor Shape : {filtered_ir.shape}")
        print(f"    - Output Directory: {output_dir}")

    print("\n" + "=" * 70)
    print(f"All datasets processed successfully! Saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    extract_chonf_dataset(splits=["train", "val", "test"])
