# This script is used to verify the correctness of the SMILES labels for the EB and EB_TF datasets.

# Before running the verification, please download the required data from CSU-IR/Multi-stage training data/Stage III- EXP at Hugging Face into their corresponding directories:
# EB Dataset: Download the smiles file and label_enriched file to:
# https://github.com/Hsqcsu/CSU-IR/tree/main/CSU-IR/data/Multi-staged_training_data/Experimental_Benching_data/EB/data_with_NIST_IR
# EB_TF Dataset: Download the smiles file and label_enriched file to:
# https://github.com/Hsqcsu/CSU-IR/tree/main/CSU-IR/data/Multi-staged_training_data/Experimental_Benching_data/EB_TF

import argparse
from datetime import datetime
import os
import sys  # Added missing sys import
import time
import urllib.parse
import pandas as pd
from rdkit import Chem, RDLogger
import requests
from tqdm import tqdm

# Project path configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
print(f"Project Root: {PROJECT_ROOT}")

# Suppress unnecessary RDKit warnings
RDLogger.DisableLog("rdApp.*")

# ==================== 1. Paths and Dataset Configurations ====================
DATA_PATHS = {
    "EB": os.path.join(
        PROJECT_ROOT,
        "data",
        "Multi-staged_training_data",
        "Experimental_Benching_data",
        "EB",
        "data_with_NIST_IR",
    ),
    "EB_TF": os.path.join(
        PROJECT_ROOT,
        "data",
        "Multi-staged_training_data",
        "Experimental_Benching_data",
        "EB_TF",
    ),
}

DATASETS = ["train", "val", "test"]


# ==================== 2. SMILES Retrieval Tools (Real-time requests per item) ====================
def fetch_smiles_pubchem(substance_name, retries=3):
    """Retrieve SMILES by substance name or CAS registry number from PubChem PUG REST API."""
    if not substance_name or substance_name == "None":
        return None
    encoded_name = urllib.parse.quote(substance_name.strip(), safe="")
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/property/IsomericSMILES/TXT"
    for _ in range(retries):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.text.strip().split("\n")[0].strip()
            elif resp.status_code == 404:
                return None
            elif resp.status_code in [503, 429]:  # Rate-limiting wait
                time.sleep(1.0)
        except Exception:
            time.sleep(0.5)
    return None


def fetch_smiles_opsin(substance_name, retries=3):
    """Retrieve SMILES for IUPAC names via the OPSIN Web Engine."""
    if not substance_name or substance_name == "None":
        return None
    encoded_name = urllib.parse.quote(substance_name.strip(), safe="")
    url = f"https://opsin.ch.cam.ac.uk/opsin/{encoded_name}.smi"
    for _ in range(retries):
        try:
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200 and resp.text.strip():
                return resp.text.strip().split()[0].strip()
            elif resp.status_code == 404:
                return None
        except Exception:
            time.sleep(0.5)
    return None


def query_smiles(name, use_iupac):
    """Send an independent network request based on whether use_iupac is 'YES', returning (smiles, tool_name)."""
    tool = (
        "OPSIN Web Engine"
        if str(use_iupac).strip().upper() == "YES"
        else "PubChem PUG REST"
    )
    if str(use_iupac).strip().upper() == "YES":
        return fetch_smiles_opsin(name), tool
    else:
        return fetch_smiles_pubchem(name), tool


# ==================== 3. Non-stereochemical SMILES Canonicalization ====================
def canonicalize_no_stereo(smiles):
    """Remove chiral and cis/trans stereochemical information via RDKit and output canonical SMILES."""
    if not smiles or smiles == "None":
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    except Exception:
        return None


# ==================== 4. Main Verification Workflow ====================
def run_verification(target_type):
    target_type = target_type.upper()
    if target_type not in DATA_PATHS:
        raise ValueError(
            f"Invalid target dataset: {target_type}. Must be 'EB' or 'EB_TF'."
        )

    data_dir = DATA_PATHS[target_type]
    print(f"\nTarget Dataset Selected: [{target_type}]")
    print(f"Directory: {data_dir}")

    # Output paths tailored to the selected dataset
    report_output_path = os.path.join(
        data_dir, f"{target_type}_smiles_verification_error_report.csv"
    )
    log_output_path = os.path.join(
        data_dir, f"{target_type}_verification_detailed_query_log.txt"
    )

    error_records = []

    # Open TXT log file (immediate flush, append records on-the-fly)
    with open(log_output_path, "w", encoding="utf-8") as log_file:
        log_file.write("====================================================\n")
        log_file.write(
            f"     SMILES Verification & Web Retrieval Log ({target_type})    \n"
        )
        log_file.write(
            f"     Execution Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        log_file.write(
            "     Language: English (All-inclusive: SUCCESS & FAILED)\n"
        )
        log_file.write(
            "====================================================\n\n"
        )
        log_file.flush()

        for split in DATASETS:
            enriched_file = os.path.join(
                data_dir, f"{split}_labels_enriched_for_verification.txt"
            )
            augmented_file = os.path.join(data_dir, f"{split}_smiles.txt")

            if not os.path.exists(enriched_file) or not os.path.exists(
                augmented_file
            ):
                tqdm.write(
                    f"[SKIP] Relevant files not found for dataset split: {split}"
                )
                continue

            # Use rstrip('\r\n') to keep physical line indices strictly aligned
            with open(enriched_file, "r", encoding="utf-8") as f:
                enriched_lines = [line.rstrip("\r\n") for line in f]

            with open(augmented_file, "r", encoding="utf-8") as f:
                augmented_smiles_lines = [line.rstrip("\r\n") for line in f]

            total_lines = len(enriched_lines)
            tqdm.write(
                f"\n{'=' * 25} Starting verification for: {target_type} [{split.upper()}] {'=' * 25}"
            )

            pbar = tqdm(
                enumerate(enriched_lines, start=1),
                total=total_lines,
                desc=f"Progress [{split.upper()}]",
                unit="item",
                dynamic_ncols=True,
            )

            for line_idx, line in pbar:
                if not line.strip():
                    continue

                parts = line.split("\t")
                if len(parts) < 6:
                    continue

                source = parts[0].strip()
                file_name = parts[1].strip()
                smiles_query = parts[2].strip()
                use_iupac = parts[3].strip()
                desalted_query = parts[4].strip()
                desalted_smiles = parts[5].strip()

                # 1. SDBS or empty substance name: silently skip without sending queries
                if (
                    source == "SDBS"
                    or smiles_query == "None"
                    or not smiles_query
                ):
                    continue

                # 2. Read the actual Target SMILES used in CSU-IR (matching line index from smiles file)
                if line_idx <= len(augmented_smiles_lines):
                    csu_target_smiles = augmented_smiles_lines[
                        line_idx - 1
                    ].strip()
                else:
                    csu_target_smiles = None

                # 3. Determine target structure to compare against web query result (eval_target_smiles)
                is_salt_handled = desalted_query.upper() == "YES"
                salt_status_str = (
                    "YES (Desalted/Salt-handled)"
                    if is_salt_handled
                    else "NO (As-is)"
                )

                if is_salt_handled:
                    eval_target_smiles = desalted_smiles
                else:
                    eval_target_smiles = csu_target_smiles

                # 4. Fire independent web request
                queried_smiles, tool_name = query_smiles(
                    smiles_query, use_iupac
                )

                # 5. Non-stereochemical canonicalization
                canon_queried = canonicalize_no_stereo(queried_smiles)
                canon_target = canonicalize_no_stereo(eval_target_smiles)

                # 6. Check consistency and determine status
                is_match = False
                fail_reason = ""
                en_status = ""
                en_consistency = ""

                if not queried_smiles:
                    fail_reason = (
                        "Web retrieval returned no SMILES (404 or request"
                        " timeout)"
                    )
                    en_status = "FAILED - SMILES Not Found"
                    en_consistency = "N/A (Query Failed)"
                elif canon_queried is None:
                    fail_reason = "Retrieved SMILES could not be parsed by RDKit"
                    en_status = "FAILED - Queried SMILES Parsing Error"
                    en_consistency = "N/A (Parse Error)"
                elif canon_target is None:
                    fail_reason = (
                        "Target comparison SMILES could not be parsed by RDKit"
                        " or is empty"
                    )
                    en_status = "FAILED - Target SMILES Invalid"
                    en_consistency = "N/A (Invalid Target)"
                elif canon_queried == canon_target:
                    is_match = True
                    en_status = "SUCCESS"
                    en_consistency = "YES (Consistent without stereo)"
                else:
                    fail_reason = (
                        "Canonical SMILES mismatch after stereochemistry"
                        " removal"
                    )
                    en_status = "FAILED - Inconsistent Canonical SMILES"
                    en_consistency = "NO (Mismatch)"

                # 7. Write to TXT log
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[
                    :-3
                ]

                log_lines = [
                    f"[{timestamp}] STATUS: {en_status}",
                    f"  Dataset               : {split}",
                    f"  Index                 : Line {line_idx}",
                    f"  Source                : {source}",
                    f"  File Name             : {file_name}",
                    f"  Tool Used             : {tool_name}",
                    f"  Query Term            : {smiles_query}",
                    (
                        "  Queried SMILES        :"
                        f" {queried_smiles if queried_smiles else 'None (Not Found)'}"
                    ),
                    f"  Salt Handled          : {salt_status_str}",
                ]

                # Detailed structure output
                if queried_smiles:
                    log_lines.append(
                        "  Target SMILES (Used in CSU-IR)   :"
                        f" {csu_target_smiles}"
                    )
                    if is_salt_handled:
                        log_lines.append(
                            "  Target SMILES (desalted)         :"
                            f" {desalted_smiles}"
                        )

                    log_lines.extend([
                        (
                            "  Canon Queried (No-Iso):"
                            f" {canon_queried if canon_queried else 'Invalid'}"
                        ),
                        (
                            "  Canon Target  (No-Iso):"
                            f" {canon_target if canon_target else 'Invalid'}"
                        ),
                        f"  Match                 : {en_consistency}",
                    ])
                else:
                    log_lines.append(f"  Match                 : {en_consistency}")

                log_lines.append("-" * 75 + "\n")

                # Flush to disk immediately
                log_file.write("\n".join(log_lines))
                log_file.flush()

                # 8. Append failure cases to CSV error report and highlight in console
                if not is_match:
                    err_msg = (
                        f"\n{'!' * 20} [MISMATCH DETECTED] Line: {line_idx}"
                        f" ({split}) {'!' * 20}\n[Raw Line Content]           "
                        f" : {line}\n  - Source                     : {source}\n"
                        f"  - File Name                  : {file_name}\n  -"
                        f" Query Term                 : {smiles_query}\n  -"
                        f" Tool Used                  : {tool_name}\n  -"
                        f" Queried SMILES             : {queried_smiles}\n  -"
                        f" Salt Handled               : {desalted_query}\n  -"
                        " Target SMILES in Dataset   :"
                        f" {csu_target_smiles}\n  - Target SMILES Evaluated   "
                        f" : {eval_target_smiles}\n  - Failure Reason          "
                        f"   : {fail_reason}\n{'!' * 65}\n"
                    )
                    tqdm.write(err_msg)

                    error_records.append({
                        "dataset": split,
                        "line_number": line_idx,
                        "source": source,
                        "file_name": file_name,
                        "smiles_query": smiles_query,
                        "use_IUPAC": use_iupac,
                        "desalted_query": desalted_query,
                        "desalted_smiles": desalted_smiles,
                        "target_smiles_used_in_csu": csu_target_smiles,
                        "target_smiles_evaluated": eval_target_smiles,
                        "queried_smiles": queried_smiles,
                        "canon_target": canon_target,
                        "canon_queried": canon_queried,
                        "fail_reason": fail_reason,
                        "raw_line": line,
                    })

                pbar.set_postfix({"failed_count": len(error_records)})

    # ==================== 5. Output Error Report (CSV) ====================
    tqdm.write("\n" + "=" * 50)
    tqdm.write(
        f"All query logs (including SUCCESS and FAILED) saved to:"
        f" {log_output_path}"
    )
    if error_records:
        df_errors = pd.DataFrame(error_records)
        df_errors.to_csv(report_output_path, index=False, encoding="utf-8-sig")
        tqdm.write(
            f"[!] Verification completed! Found {len(error_records)} mismatch"
            " record(s)."
        )
        tqdm.write(f"Error report saved to: {report_output_path}")
    else:
        tqdm.write(
            "[✓] Verification completed successfully! All items matched"
            " perfectly. No error report generated."
        )


def select_dataset_interactive():
    """Prompt user to select dataset via interactive terminal input if no CLI argument passed."""
    print("\n" + "=" * 45)
    print("      Select Dataset for SMILES Verification")
    print("=" * 45)
    print("1. EB    (Experimental_Benching_data/EB/data_with_NIST_IR)")
    print("2. EB_TF (Experimental_Benching_data/EB_TF)")
    print("=" * 45)

    choice = input("Please select [1/2 or EB/EB_TF] (Default is 1): ").strip()
    if choice in ["2", "EB_TF", "eb_tf"]:
        return "EB_TF"
    return "EB"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify SMILES labels for EB or EB_TF datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["EB", "EB_TF"],
        help="Target dataset to verify (EB or EB_TF)",
    )
    args = parser.parse_args()

    # Determine dataset selection: CLI argument takes precedence; fallback to prompt
    if args.dataset:
        selected_dataset = args.dataset
    else:
        selected_dataset = select_dataset_interactive()

    run_verification(selected_dataset)
