# This script is used to verify the correctness of the SMILES labels for the EB and EB_TF datasets.

# Before running the verification, please download the required data from CSU-IR/Multi-stage training data/Stage III- EXP at Hugging Face into their corresponding directories:
# EB Dataset: Download the smiles file and label_enriched file to:
# https://github.com/Hsqcsu/CSU-IR/tree/main/CSU-IR/data/Multi-staged_training_data/Experimental_Benching_data/EB/data_with_NIST_IR
# EB_TF Dataset: Download the smiles file and label_enriched file to:
# https://github.com/Hsqcsu/CSU-IR/tree/main/CSU-IR/data/Multi-staged_training_data/Experimental_Benching_data/EB_TF

import argparse
from datetime import datetime
import os
import sys
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


# ==================== 2. SMILES Retrieval Tools ====================
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
    """Send an independent network request based on whether use_iupac is 'YES'."""
    tool = (
        "OPSIN Web Engine"
        if str(use_iupac).strip().upper() == "YES"
        else "PubChem PUG REST"
    )
    if str(use_iupac).strip().upper() == "YES":
        return fetch_smiles_opsin(name), tool
    else:
        return fetch_smiles_pubchem(name), tool


# ==================== 3. SMILES Canonicalization Tools ====================
def canonicalize_stereo(smiles):
    """Canonicalize SMILES while PRESERVING stereochemical information."""
    if not smiles or smiles == "None":
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception:
        return None


def canonicalize_no_stereo(smiles):
    """Remove chiral and cis/trans stereochemical information and output canonical SMILES."""
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

    report_output_path = os.path.join(
        data_dir, f"{target_type}_smiles_verification_error_report.csv"
    )
    log_output_path = os.path.join(
        data_dir, f"{target_type}_verification_detailed_query_log.txt"
    )

    error_records = []

    with open(log_output_path, "w", encoding="utf-8") as log_file:
        # File header strictly matching respective dataset logs
        log_file.write("====================================================\n")
        if target_type == "EB_TF":
            log_file.write(
                "    TF SMILES Verification & Web Retrieval Log      \n"
            )
            log_file.write(
                f"    Execution Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            log_file.write(
                "    Datasets: train, val, test (TF/ThermoFisher only)\n"
            )
            log_file.write(
                "    Language: English (All-inclusive: SUCCESS & FAILED)\n"
            )
        else:
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
            # Candidate file paths covering diverse naming conventions
            candidate_enriched = [
                os.path.join(
                    data_dir,
                    f"{target_type}_{split}_labels_enriched_for_verification.txt",
                ),
                os.path.join(
                    data_dir, f"{split}_labels_enriched_for_verification.txt"
                ),
                os.path.join(data_dir, f"{split}_tf_source_extended.txt"),
            ]
            candidate_augmented = [
                os.path.join(data_dir, f"{target_type}_{split}_smiles.txt"),
                os.path.join(data_dir, f"{split}_smiles.txt"),
                os.path.join(data_dir, f"{split}_smiles_augmented.txt"),
            ]

            enriched_file = next(
                (f for f in candidate_enriched if os.path.exists(f)), None
            )
            augmented_file = next(
                (f for f in candidate_augmented if os.path.exists(f)), None
            )

            if not enriched_file or not augmented_file:
                tqdm.write(
                    f"[SKIP] Relevant files not found for dataset split: {split}"
                )
                continue

            tqdm.write(
                f"[INFO] Using enriched/TF file: {os.path.basename(enriched_file)}"
            )
            tqdm.write(
                f"[INFO] Using SMILES file:      {os.path.basename(augmented_file)}"
            )

            with open(enriched_file, "r", encoding="utf-8") as f:
                lines = [line.rstrip("\r\n") for line in f if line.strip()]

            with open(augmented_file, "r", encoding="utf-8") as f:
                split_augmented_smiles = [
                    line.rstrip("\r\n") for line in f if line.strip()
                ]

            # Remove header line if present
            if lines and (
                lines[0].startswith("split_line_idx")
                or lines[0].startswith("source\t")
            ):
                data_lines = lines[1:]
            else:
                data_lines = lines

            total_lines = len(data_lines)
            tqdm.write(
                f"\n{'=' * 25} Starting verification for: {target_type} [{split.upper()}] (Total: {total_lines}) {'=' * 25}"
            )

            pbar = tqdm(
                enumerate(data_lines, start=1),
                total=total_lines,
                desc=f"Progress [{split.upper()}]",
                unit="item",
                dynamic_ncols=True,
            )

            for cur_idx, line in pbar:
                if not line.strip():
                    continue

                parts = line.split("\t")

                # Adaptive parsing: 7-column EB_TF vs 6-column EB
                is_tf_format = len(parts) >= 7 or (
                    len(parts) >= 6 and parts[0].strip().isdigit()
                )

                if is_tf_format:
                    split_line_idx = int(parts[0].strip())
                    source = parts[1].strip()
                    file_name = parts[2].strip()
                    smiles_query = parts[3].strip()
                    use_iupac = parts[4].strip()
                    desalted_query = parts[5].strip()
                    desalted_smiles = (
                        parts[6].strip() if len(parts) > 6 else ""
                    )
                else:
                    split_line_idx = None
                    source = parts[0].strip()
                    file_name = parts[1].strip()
                    smiles_query = parts[2].strip()
                    use_iupac = parts[3].strip()
                    desalted_query = parts[4].strip()
                    desalted_smiles = parts[5].strip()

                # SDBS or empty query: silently skip
                if (
                    source == "SDBS"
                    or not smiles_query
                    or smiles_query == "None"
                ):
                    continue

                # Locate target ground-truth SMILES
                actual_lookup_idx = (
                    split_line_idx if split_line_idx is not None else cur_idx
                )
                if 1 <= actual_lookup_idx <= len(split_augmented_smiles):
                    csu_raw_target_smiles = split_augmented_smiles[
                        actual_lookup_idx - 1
                    ].strip()
                else:
                    csu_raw_target_smiles = None

                is_salt_handled = desalted_query.upper() == "YES"
                salt_status_str = (
                    "YES (Desalted/Salt-handled)"
                    if is_salt_handled
                    else "NO (As-is)"
                )

                # Determine the target SMILES used for evaluation
                if is_salt_handled:
                    eval_target_smiles = desalted_smiles
                else:
                    eval_target_smiles = csu_raw_target_smiles

                # Web retrieval
                queried_smiles, tool_name = query_smiles(
                    smiles_query, use_iupac
                )

                # Stereo and non-stereo canonicalization
                canon_queried_stereo = canonicalize_stereo(queried_smiles)
                canon_target_stereo = canonicalize_stereo(eval_target_smiles)
                canon_queried_no_stereo = canonicalize_no_stereo(queried_smiles)
                canon_target_no_stereo = canonicalize_no_stereo(
                    eval_target_smiles
                )

                # Evaluate consistency
                is_match = False
                is_stereo_match = False
                fail_reason = ""
                en_status = ""

                if not queried_smiles:
                    fail_reason = (
                        "Web retrieval returned no SMILES (404 or request"
                        " timeout)"
                    )
                    en_status = "FAILED - SMILES Not Found"
                    en_consistency = "N/A (Query Failed)"
                elif canon_queried_no_stereo is None:
                    fail_reason = "Retrieved SMILES could not be parsed by RDKit"
                    en_status = "FAILED - Queried SMILES Parsing Error"
                    en_consistency = "N/A (Parse Error)"
                elif canon_target_no_stereo is None:
                    fail_reason = (
                        "Target comparison SMILES could not be parsed by RDKit"
                        " or is empty"
                    )
                    en_status = "FAILED - Target SMILES Invalid"
                    en_consistency = "N/A (Invalid Target)"
                elif (
                    canon_queried_stereo is not None
                    and canon_queried_stereo == canon_target_stereo
                ):
                    is_match = True
                    is_stereo_match = True
                    en_status = "SUCCESS"
                    en_consistency = "YES"
                elif canon_queried_no_stereo == canon_target_no_stereo:
                    is_match = True
                    is_stereo_match = False
                    en_status = "SUCCESS"
                    en_consistency = "YES (Consistent without stereo)"
                else:
                    fail_reason = (
                        "Canonical SMILES mismatch after stereochemistry"
                        " removal"
                    )
                    en_status = "FAILED - Inconsistent Canonical SMILES"
                    en_consistency = "NO (Mismatch)"

                # Build log lines
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[
                    :-3
                ]
                log_lines = [
                    f"[{timestamp}] STATUS: {en_status}",
                    f"  Dataset               : {split}",
                ]

                # Reproduce line index format for respective datasets
                if is_tf_format:
                    log_lines.append(
                        "  Split File Line Index :"
                        f" Line {split_line_idx} (in"
                        f" {split}_source_file_name.txt)"
                    )
                else:
                    log_lines.append(
                        f"  Index                 : Line {cur_idx}"
                    )

                log_lines.extend([
                    f"  Source                : {source}",
                    f"  File Name             : {file_name}",
                    f"  Tool Used             : {tool_name}",
                    f"  Query Term            : {smiles_query}",
                    (
                        "  Queried SMILES        :"
                        f" {queried_smiles if queried_smiles else 'None (Not Found)'}"
                    ),
                    f"  Salt Handled          : {salt_status_str}",
                ])

                # Reproduce Target SMILES format for respective datasets
                if queried_smiles:
                    if is_tf_format:
                        # EB_TF layout: no space before colon, no separate desalted line
                        log_lines.append(
                            f"  Target SMILES (Used in CSU-IR):"
                            f" {eval_target_smiles}"
                        )
                    else:
                        # EB layout: 3 spaces before colon, separate desalted line if applicable
                        log_lines.append(
                            "  Target SMILES (Used in CSU-IR)   :"
                            f" {csu_raw_target_smiles}"
                        )
                        if is_salt_handled:
                            log_lines.append(
                                "  Target SMILES (desalted)         :"
                                f" {desalted_smiles}"
                            )

                    # Output the final three lines matching log update script format
                    if is_stereo_match:
                        log_lines.extend([
                            f"  Canon Queried : {canon_queried_stereo}",
                            f"  Canon Target  : {canon_target_stereo}",
                            "  Match                 : YES",
                        ])
                    else:
                        log_lines.extend([
                            (
                                "  Canon Queried (No-Iso):"
                                f" {canon_queried_no_stereo if canon_queried_no_stereo else 'Invalid'}"
                            ),
                            (
                                "  Canon Target  (No-Iso):"
                                f" {canon_target_no_stereo if canon_target_no_stereo else 'Invalid'}"
                            ),
                            f"  Match                 : {en_consistency}",
                        ])
                else:
                    log_lines.append(f"  Match                 : {en_consistency}")

                log_lines.append("-" * 75 + "\n")

                log_file.write("\n".join(log_lines))
                log_file.flush()

                # Record failures
                if not is_match:
                    display_idx = (
                        split_line_idx if is_tf_format else cur_idx
                    )
                    err_msg = (
                        f"\n{'!' * 20} [MISMATCH DETECTED] Line: {display_idx}"
                        f" ({split}) {'!' * 20}\n[Raw Line Content]           "
                        f" : {line}\n  - Source                     : {source}\n"
                        f"  - File Name                  : {file_name}\n  -"
                        f" Query Term                 : {smiles_query}\n  -"
                        f" Tool Used                  : {tool_name}\n  -"
                        f" Queried SMILES             : {queried_smiles}\n  -"
                        f" Salt Handled               : {desalted_query}\n  -"
                        " Target SMILES in Dataset   :"
                        f" {eval_target_smiles}\n  - Failure Reason          "
                        f"   : {fail_reason}\n{'!' * 65}\n"
                    )
                    tqdm.write(err_msg)

                    error_records.append({
                        "dataset": split,
                        "line_number": display_idx,
                        "source": source,
                        "file_name": file_name,
                        "smiles_query": smiles_query,
                        "use_IUPAC": use_iupac,
                        "desalted_query": desalted_query,
                        "desalted_smiles": desalted_smiles,
                        "target_smiles_used_in_csu": eval_target_smiles,
                        "queried_smiles": queried_smiles,
                        "canon_target": canon_target_no_stereo,
                        "canon_queried": canon_queried_no_stereo,
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
    """Prompt user to select dataset interactively."""
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

    selected_dataset = (
        args.dataset if args.dataset else select_dataset_interactive()
    )
    run_verification(selected_dataset)
