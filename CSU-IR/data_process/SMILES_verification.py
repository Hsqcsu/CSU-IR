from datetime import datetime
import os
import time
import urllib.parse
import pandas as pd
import requests
from rdkit import Chem, RDLogger
from tqdm import tqdm

# Suppress unnecessary RDKit warnings
RDLogger.DisableLog("rdApp.*")

# ==================== 1. Paths and Configurations ====================
DATA_DIR = r"F:\Spectrum\1122_after\CSU_IR_20261007_data_reprocess\EB_data\merged_splitted_dataset"
DATASETS = ["train", "val", "test"]

# Verification error report output path (CSV)
REPORT_OUTPUT_PATH = os.path.join(
    DATA_DIR, "smiles_verification_error_report.csv"
)

# Real-time detailed query log output path (TXT Log)
LOG_OUTPUT_PATH = os.path.join(
    DATA_DIR, "verification_detailed_query_log.txt"
)


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
def run_verification():
    error_records = []

    # Open TXT log file (immediate flush, append records on-the-fly)
    with open(LOG_OUTPUT_PATH, "w", encoding="utf-8") as log_file:
        log_file.write("====================================================\n")
        log_file.write("     SMILES Verification & Web Retrieval Log       \n")
        log_file.write(
            f"     Execution Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        log_file.write("     Language: English (All-inclusive: SUCCESS & FAILED)\n")
        log_file.write("====================================================\n\n")
        log_file.flush()

        for split in DATASETS:
            enriched_file = os.path.join(DATA_DIR, f"{split}_source_file_name_enriched.txt")
            augmented_file = os.path.join(DATA_DIR, f"{split}_smiles_augmented.txt")

            if not os.path.exists(enriched_file) or not os.path.exists(augmented_file):
                tqdm.write(f"[SKIP] Relevant files not found for dataset split: {split}")
                continue

            # Use rstrip('\r\n') to keep physical line indices strictly aligned
            with open(enriched_file, "r", encoding="utf-8") as f:
                enriched_lines = [line.rstrip("\r\n") for line in f]

            with open(augmented_file, "r", encoding="utf-8") as f:
                augmented_smiles_lines = [line.rstrip("\r\n") for line in f]

            total_lines = len(enriched_lines)
            tqdm.write(f"\n{'=' * 25} Starting verification for dataset: {split.upper()} {'=' * 25}")

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
                if source == "SDBS" or smiles_query == "None" or not smiles_query:
                    continue

                # 2. Read the actual Target SMILES used in CSU-IR (matching line index from smiles file)
                if line_idx <= len(augmented_smiles_lines):
                    csu_target_smiles = augmented_smiles_lines[line_idx - 1].strip()
                else:
                    csu_target_smiles = None

                # 3. Determine target structure to compare against web query result (eval_target_smiles)
                is_salt_handled = (desalted_query.upper() == "YES")
                salt_status_str = "YES (Desalted/Salt-handled)" if is_salt_handled else "NO (As-is)"

                if is_salt_handled:
                    eval_target_smiles = desalted_smiles
                else:
                    eval_target_smiles = csu_target_smiles

                # 4. Fire independent web request
                queried_smiles, tool_name = query_smiles(smiles_query, use_iupac)

                # 5. Non-stereochemical canonicalization
                canon_queried = canonicalize_no_stereo(queried_smiles)
                canon_target = canonicalize_no_stereo(eval_target_smiles)

                # 6. Check consistency and determine status
                is_match = False
                fail_reason = ""
                en_status = ""
                en_consistency = ""

                if not queried_smiles:
                    fail_reason = "Web retrieval returned no SMILES (404 or request timeout)"
                    en_status = "FAILED - SMILES Not Found"
                    en_consistency = "N/A (Query Failed)"
                elif canon_queried is None:
                    fail_reason = "Retrieved SMILES could not be parsed by RDKit"
                    en_status = "FAILED - Queried SMILES Parsing Error"
                    en_consistency = "N/A (Parse Error)"
                elif canon_target is None:
                    fail_reason = "Target comparison SMILES could not be parsed by RDKit or is empty"
                    en_status = "FAILED - Target SMILES Invalid"
                    en_consistency = "N/A (Invalid Target)"
                elif canon_queried == canon_target:
                    is_match = True
                    en_status = "SUCCESS"
                    en_consistency = "YES (Consistent without stereo)"
                else:
                    fail_reason = "Canonical SMILES mismatch after stereochemistry removal"
                    en_status = "FAILED - Inconsistent Canonical SMILES"
                    en_consistency = "NO (Mismatch)"

                # 7. Write to TXT log
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                log_lines = [
                    f"[{timestamp}] STATUS: {en_status}",
                    f"  Dataset               : {split}",
                    f"  Index                 : Line {line_idx}",
                    f"  Source                : {source}",
                    f"  File Name             : {file_name}",
                    f"  Tool Used             : {tool_name}",
                    f"  Query Term            : {smiles_query}",
                    f"  Queried SMILES        : {queried_smiles if queried_smiles else 'None (Not Found)'}",
                    f"  Salt Handled          : {salt_status_str}",
                ]

                # Detailed structure output
                if queried_smiles:
                    log_lines.append(f"  Target SMILES (Used in CSU-IR)   : {csu_target_smiles}")
                    # If desalted/salt-handled is YES, log an extra line for desalted Target SMILES
                    if is_salt_handled:
                        log_lines.append(f"  Target SMILES (desalted)         : {desalted_smiles}")

                    log_lines.extend([
                        f"  Canon Queried (No-Iso): {canon_queried if canon_queried else 'Invalid'}",
                        f"  Canon Target  (No-Iso): {canon_target if canon_target else 'Invalid'}",
                        f"  Match                 : {en_consistency}"
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
                        f"\n{'!' * 20} [MISMATCH DETECTED] Line: {line_idx} ({split}) {'!' * 20}\n"
                        f"[Raw Line Content]            : {line}\n"
                        f"  - Source                     : {source}\n"
                        f"  - File Name                  : {file_name}\n"
                        f"  - Query Term                 : {smiles_query}\n"
                        f"  - Tool Used                  : {tool_name}\n"
                        f"  - Queried SMILES             : {queried_smiles}\n"
                        f"  - Salt Handled               : {desalted_query}\n"
                        f"  - Target SMILES in Dataset   : {csu_target_smiles}\n"
                        f"  - Target SMILES Evaluated    : {eval_target_smiles}\n"
                        f"  - Failure Reason             : {fail_reason}\n"
                        f"{'!' * 65}\n"
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
                        "raw_line": line
                    })

                pbar.set_postfix({"failed_count": len(error_records)})

    # ==================== 5. Output Error Report (CSV) ====================
    tqdm.write("\n" + "=" * 50)
    tqdm.write(f"All query logs (including SUCCESS and FAILED) saved to: {LOG_OUTPUT_PATH}")
    if error_records:
        df_errors = pd.DataFrame(error_records)
        df_errors.to_csv(REPORT_OUTPUT_PATH, index=False, encoding="utf-8-sig")
        tqdm.write(f"[!] Verification completed! Found {len(error_records)} mismatch record(s).")
        tqdm.write(f"Error report saved to: {REPORT_OUTPUT_PATH}")
    else:
        tqdm.write("[✓] Verification completed successfully! All items matched perfectly. No error report generated.")


if __name__ == "__main__":
    run_verification()
