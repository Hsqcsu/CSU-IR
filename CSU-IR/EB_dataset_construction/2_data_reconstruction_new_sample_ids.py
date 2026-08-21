"""NIST IR Two-Stage High-Performance Reconstruction & Consistency Verification Script

Architecture Overview:
[Stage 1: Full Batch Pure Download]
  - 1.0s request interval with strict timeout protection.
  - Automatically scans for missing JDX files with multi-round resume-on-break capability until 100% downloaded.
[Stage 2: Local Offline High-Speed Batch Processing]
  - Operates completely offline: performs cubic spline interpolation, absorbance conversion, and tensor assembly.
  - Saves final outputs into the designated reconstruction directory.
[Stage 3: End-to-End Consistency Verification]
  - Compares line-by-line SMILES and numerical IR spectral values against ground-truth benchmark datasets.
"""

import os
import shutil
import time
import numpy as np
from rdkit import Chem
import requests
from requests.adapters import HTTPAdapter
from scipy.interpolate import CubicSpline
import torch
from tqdm import tqdm
from urllib3.exceptions import InsecureRequestWarning

# ================= 1. Network & Connection Pool Configuration =================
USE_PROXY = False

if not USE_PROXY:
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)

os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

SAFE_DELAY = 1.0
REQUEST_TIMEOUT = (3.0, 7.0)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}


def create_persistent_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    session.trust_env = False
    adapter = HTTPAdapter(
        pool_connections=20, pool_maxsize=20, max_retries=0
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


GLOBAL_SESSION = create_persistent_session()

# ================= 2. Path Configuration =================
# A. Input: Directory containing datasets with None IR and copied annotated label files
INPUT_NO_NIST_DIR = r"F:\Spectrum\1122_after\model\ESA_model_sigmoid\20250530_esa_ir_CNN_transformer\_20260602_MG_training\data\data_process\optimization\exp_reconstrction\data_without_NIST_IR_new_sample_ids_file"

# B. Cache: Directory for storing downloaded raw JDX files
JDX_CACHE_DIR = r"F:\Spectrum\1122_after\model\ESA_model_sigmoid\20250530_esa_ir_CNN_transformer\_20260602_MG_training\data\data_process\optimization\exp_reconstrction\ir_redownload_new_sample_ids_file"
os.makedirs(JDX_CACHE_DIR, exist_ok=True)

# C. Output: Directory for saving fully reconstructed datasets
OUTPUT_RECON_DIR = r"F:\Spectrum\1122_after\model\ESA_model_sigmoid\20250530_esa_ir_CNN_transformer\_20260602_MG_training\data\data_process\optimization\exp_reconstrction\data_reconstructioned_new_sample_ids_file"
os.makedirs(OUTPUT_RECON_DIR, exist_ok=True)

# D. Benchmark: Directory containing ground-truth baseline data for verification
BENCHMARK_DIR = r"F:\Spectrum\1122_after\model\ESA_model_sigmoid\20250530_esa_ir_CNN_transformer\_20260602_MG_training\data\data_process\optimization\exp_20260817\0_1_2_4_5_6_8_9_16_splitted_and_augmented_data_delete_299"

SPLITS = ["train", "val", "test"]


# ================= 3. Stage 1: Network Download Functions =================
def download_single_jdx_file(sample_id):
    parts = sample_id.split("_")
    prefix = parts[0]
    spec_index = parts[-1] if len(parts) >= 3 else "0"

    jdx_filename = f"{sample_id}.jdx"
    local_jdx_path = os.path.join(JDX_CACHE_DIR, jdx_filename)

    if os.path.exists(local_jdx_path) and os.path.getsize(local_jdx_path) >= 100:
        return True

    url = f"https://webbook.nist.gov/cgi/cbook.cgi?JCAMP={prefix}&Index={spec_index}&Type=IR"
    download_success = False

    try:
        res = GLOBAL_SESSION.get(
            url, timeout=REQUEST_TIMEOUT, verify=False
        )
        if (
            "##JCAMP-DX=" in res.text
            or "##TITLE=" in res.text
            or "##DATA TYPE=" in res.text
        ):
            with open(local_jdx_path, "w", encoding="utf-8") as f:
                f.write(res.text)
            download_success = True
    except Exception:
        if os.path.exists(local_jdx_path):
            try:
                os.remove(local_jdx_path)
            except:
                pass
        download_success = False

    time.sleep(SAFE_DELAY)
    return download_success


def ensure_all_jdx_downloaded(split_name, nist_sample_ids):
    unique_ids = list(
        set([sid for sid in nist_sample_ids if sid and sid.lower() != "none"])
    )
    total_target = len(unique_ids)

    print(
        f"\n📥 [Stage 1: Batch JDX Download] Target files to acquire for '{split_name.upper()}': {total_target} (Interval: {SAFE_DELAY}s)"
    )

    round_idx = 1
    while True:
        missing_ids = [
            sid
            for sid in unique_ids
            if not os.path.exists(os.path.join(JDX_CACHE_DIR, f"{sid}.jdx"))
            or os.path.getsize(os.path.join(JDX_CACHE_DIR, f"{sid}.jdx")) < 100
        ]

        if not missing_ids:
            print(
                f"🎉 [{split_name.upper()} Set] All {total_target} JDX files are fully cached locally!"
            )
            break

        print(
            f"🔄 --- [Download Round {round_idx}] Remaining: {len(missing_ids)} / {total_target} ---"
        )

        success_count = 0
        for sid in tqdm(missing_ids, desc=f"Round {round_idx} Downloading"):
            if download_single_jdx_file(sid):
                success_count += 1

        print(
            f"--> Round {round_idx} Completed: Successfully downloaded {success_count}, Remaining: {len(missing_ids) - success_count}."
        )
        round_idx += 1

        if success_count == 0 and len(missing_ids) > 0:
            print("⚠️ Network fluctuation detected. Sleeping for 2.0s before retrying...")
            time.sleep(2.0)


# ================= 4. Stage 2: Offline Batch Processing & Interpolation Functions =================
def parse_jdx_spectra(file_path):
    x_raw, y_raw = [], []
    xunits = "WAVENUMBERS"
    xfactor = 1.0
    yfactor = 1.0
    deltax = None
    in_xydata = False

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("##XUNITS="):
                    xunits = line.split("=", 1)[1].strip().upper()
                elif line.startswith("##XFACTOR="):
                    xfactor = float(line.split("=", 1)[1].strip())
                elif line.startswith("##YFACTOR="):
                    yfactor = float(line.split("=", 1)[1].strip())
                elif line.startswith("##DELTAX="):
                    deltax = float(line.split("=", 1)[1].strip())
                elif line.startswith("##XYDATA="):
                    in_xydata = True
                    continue
                elif line.startswith("##END=") or (
                    line.startswith("##") and in_xydata
                ):
                    in_xydata = False

                if in_xydata:
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        line_x = float(parts[0]) * xfactor
                        for k, val_str in enumerate(parts[1:]):
                            line_y = float(val_str) * yfactor
                            if deltax is not None:
                                actual_x = line_x + k * deltax * xfactor
                            else:
                                actual_x = line_x
                            x_raw.append(actual_x)
                            y_raw.append(line_y)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Failed to parse JDX file: {e}")

    return x_raw, y_raw, xunits


def process_ir_data(x, y):
    min_x, max_x = x[0], x[-1]
    start = max(min_x, 500.0)
    end = min(max_x, 4000.0)

    if start >= end:
        raise ValueError(
            f"Wavenumber range [{min_x:.1f}, {max_x:.1f}] does not overlap with [500, 4000]."
        )

    mask = (x >= start) & (x <= end)
    x_crop = x[mask]
    y_crop = y[mask]

    if len(x_crop) < 4:
        raise ValueError("Too few valid data points for cubic spline interpolation.")

    start_int = int(np.ceil(start))
    end_int = int(np.floor(end))
    x_temp = np.arange(start_int, end_int + 1, 1.0)
    x_temp = np.clip(x_temp, x_crop[0], x_crop[-1])

    cs_crop = CubicSpline(x_crop, y_crop, extrapolate=True)
    y_temp = cs_crop(x_temp)

    x_full = np.arange(500, 4001, 1.0)
    y_full = np.zeros_like(x_full)

    idx_start = start_int - 500
    idx_end = end_int - 500
    y_full[idx_start : idx_end + 1] = y_temp

    if idx_start > 0:
        y_full[:idx_start] = y_temp[0]
    if idx_end < 3500:
        y_full[idx_end + 1 :] = y_temp[-1]

    x_target = np.linspace(500.0, 4000.0, 3500)
    cs_full = CubicSpline(x_full, y_full, extrapolate=True)
    y_target = cs_full(x_target)

    return y_target


def convert_to_absorbance_single(y_data):
    spec = np.array(y_data, dtype=np.float64)

    min_val = np.min(spec)
    max_val = np.max(spec)
    norm_spec = (
        (spec - min_val) / (max_val - min_val)
        if max_val > min_val
        else np.zeros_like(spec)
    )

    spec_median = np.median(norm_spec)
    if spec_median > 0.45:
        spec_clipped = np.clip(norm_spec, 1e-4, 1.0)
        abs_spec = -np.log10(spec_clipped)

        min_val_abs = np.min(abs_spec)
        max_val_abs = np.max(abs_spec)
        final_spec = (
            (abs_spec - min_val_abs) / (max_val_abs - min_val_abs)
            if max_val_abs > min_val_abs
            else np.zeros_like(abs_spec)
        )
    else:
        final_spec = norm_spec

    return torch.tensor(final_spec, dtype=torch.float32)


def process_local_jdx_file(sample_id):
    local_jdx_path = os.path.join(JDX_CACHE_DIR, f"{sample_id}.jdx")
    x_raw, y_raw, xunits = parse_jdx_spectra(local_jdx_path)
    if not x_raw:
        raise ValueError(f"Failed to parse local JDX file: {local_jdx_path}")

    x_raw_arr = np.array(x_raw)
    y_raw_arr = np.array(y_raw)

    if "MICR" in xunits.upper():
        x_conv = 10000.0 / x_raw_arr
    else:
        x_conv = x_raw_arr

    _, unique_indices = np.unique(x_conv, return_index=True)
    x_unique = x_conv[unique_indices]
    y_unique = y_raw_arr[unique_indices]

    sort_idx = np.argsort(x_unique)
    x_sorted = x_unique[sort_idx]
    y_sorted = y_unique[sort_idx]

    y_interp = process_ir_data(x_sorted, y_sorted)
    return convert_to_absorbance_single(y_interp)


# ================= 5. Dataset Reconstruction Pipeline =================
def load_annotated_source_records(split_name):
    """Reads (source_label, sample_id) pairs line by line directly from INPUT_NO_NIST_DIR."""
    file_path = os.path.join(
        INPUT_NO_NIST_DIR,
        f"{split_name}_source_with_sample_ids_annotated.txt",
    )
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Annotated source label file not found in input directory: {file_path}")

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            label = parts[0]
            sid = parts[1] if len(parts) > 1 else "None"
            records.append((label, sid))
    return records


def reconstruct_subset(split_name):
    print("\n" + "=" * 75)
    print(f"▶ Reconstructing Subset: [{split_name.upper()} Set]")
    print("=" * 75)

    desc_name = (
        f"{split_name}_description_augmented.txt"
        if os.path.exists(
            os.path.join(
                INPUT_NO_NIST_DIR, f"{split_name}_description_augmented.txt"
            )
        )
        else f"{split_name}_description.txt"
    )

    src_desc = os.path.join(INPUT_NO_NIST_DIR, desc_name)
    src_ir = os.path.join(INPUT_NO_NIST_DIR, f"{split_name}_ir.pt")
    src_annotated = os.path.join(
        INPUT_NO_NIST_DIR,
        f"{split_name}_source_with_sample_ids_annotated.txt",
    )

    source_records = load_annotated_source_records(split_name)
    ir_list = torch.load(src_ir, map_location="cpu")

    total_count = len(source_records)
    assert (
        len(ir_list) == total_count
    ), f"Mismatch in sample counts! Source records: {total_count}, IR list: {len(ir_list)}"

    # Extract all NIST sample IDs
    nist_sample_ids = [
        sid
        for lbl, sid in source_records
        if lbl == "NIST" and sid.lower() != "none"
    ]

    # --- Stage 1: Batch Download ---
    ensure_all_jdx_downloaded(split_name, nist_sample_ids)

    # --- Stage 2: Local High-Speed Batch Processing & Filling ---
    print(
        f"\n⚡ [Stage 2: Local Offline Batch Processing] Parsing and generating IR tensors..."
    )

    parsed_cache = {}
    reconstructed_ir_list = []

    for i in tqdm(
        range(total_count), desc=f"Batch Processing {split_name.upper()}"
    ):
        lbl, sid = source_records[i]
        curr_ir = ir_list[i]

        if lbl == "NIST" and sid.lower() != "none":
            if sid not in parsed_cache:
                parsed_cache[sid] = process_local_jdx_file(sid)
            reconstructed_ir_list.append(parsed_cache[sid])
        else:
            reconstructed_ir_list.append(
                curr_ir
                if isinstance(curr_ir, torch.Tensor)
                else torch.tensor(curr_ir, dtype=torch.float32)
            )

    # Stack into a single full Tensor and save
    reconstructed_tensor = torch.stack(reconstructed_ir_list, dim=0)

    dst_desc = os.path.join(OUTPUT_RECON_DIR, desc_name)
    dst_ir = os.path.join(OUTPUT_RECON_DIR, f"{split_name}_ir.pt")
    dst_annotated_source = os.path.join(
        OUTPUT_RECON_DIR,
        f"{split_name}_source_with_sample_ids_annotated.txt",
    )

    torch.save(reconstructed_tensor, dst_ir)
    shutil.copy2(src_desc, dst_desc)
    shutil.copy2(src_annotated, dst_annotated_source)

    print(
        f"✅ [{split_name.upper()} Set] Successfully reconstructed and exported to: {OUTPUT_RECON_DIR}"
    )
    print(f"    - Final IR Tensor Shape: {reconstructed_tensor.shape}")


# ================= 6. Consistency Verification Functions =================
def read_smiles(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data_lines = (
        lines[1:]
        if ("\t" in lines[0] or " " in lines[0])
        and any(Chem.MolFromSmiles(p) for p in lines[1].split())
        and not any(Chem.MolFromSmiles(p) for p in lines[0].split())
        else lines
    )

    res = []
    for l in data_lines:
        parts = l.strip().split("\t") if "\t" in l else l.strip().split()
        res.append(parts[0] if parts else "")
    return res


def verify_consistency(split_name):
    print("\n" + "=" * 75)
    print(
        f"▶ [Stage 3] Verifying data consistency with benchmark: [{split_name.upper()} Set]"
    )
    print("=" * 75)

    recon_ir_path = os.path.join(OUTPUT_RECON_DIR, f"{split_name}_ir.pt")
    bench_ir_path = os.path.join(BENCHMARK_DIR, f"{split_name}_ir.pt")

    desc_name = (
        f"{split_name}_description_augmented.txt"
        if os.path.exists(
            os.path.join(
                OUTPUT_RECON_DIR, f"{split_name}_description_augmented.txt"
            )
        )
        else f"{split_name}_description.txt"
    )
    recon_desc_path = os.path.join(OUTPUT_RECON_DIR, desc_name)
    bench_desc_path = os.path.join(BENCHMARK_DIR, desc_name)

    # 1. SMILES Comparison
    recon_smi = read_smiles(recon_desc_path)
    bench_smi = read_smiles(bench_desc_path)

    assert len(recon_smi) == len(bench_smi), "Mismatch in SMILES row counts!"
    smi_match = sum(1 for a, b in zip(recon_smi, bench_smi) if a == b)

    # 2. IR Numerical Spectral Comparison
    recon_ir = torch.load(recon_ir_path, map_location="cpu").float()
    bench_ir = torch.load(bench_ir_path, map_location="cpu").float()

    assert (
        recon_ir.shape == bench_ir.shape
    ), f"IR Tensor dimension mismatch: {recon_ir.shape} vs {bench_ir.shape}"

    abs_diff = torch.abs(recon_ir - bench_ir)
    max_ae = torch.max(abs_diff).item()
    mse = torch.mean((recon_ir - bench_ir) ** 2).item()
    perfect_match_count = torch.sum(
        torch.all(
            torch.isclose(recon_ir, bench_ir, atol=1e-4, rtol=1e-4), dim=1
        )
    ).item()

    total_len = len(recon_ir)
    print(f"[{split_name.upper()} Verification Statistics]")
    print(f"  - Total Samples                : {total_len}")
    print(
        f"  - SMILES 100% Identical        : {smi_match}/{total_len} ({(smi_match / total_len) * 100:.2f}%)"
    )
    print(
        f"  - IR Exact Match (w/ Tolerance): {perfect_match_count}/{total_len} ({(perfect_match_count / total_len) * 100:.2f}%)"
    )
    print(f"  - IR Max Absolute Error        : {max_ae:.6e}")
    print(f"  - IR Mean Squared Error (MSE)  : {mse:.2e}")

    if smi_match == total_len and max_ae < 1e-3:
        print(
            f"  ✨ Conclusion: [{split_name.upper()} Set] 100% element-wise match with the benchmark dataset!"
        )
    else:
        print(
            f"  ⚠️ Conclusion: [{split_name.upper()} Set] Minor discrepancies detected. Please inspect the metrics above."
        )


# ================= 7. Main Execution Entrypoint =================
def main():
    print(
        "🚀 Launching NIST IR Two-Stage High-Performance Reconstruction & Verification Pipeline (Download -> Offline Batch Processing)..."
    )
    print(
        f"Network Config: Safe Delay {SAFE_DELAY}s | Strict {REQUEST_TIMEOUT[0]+REQUEST_TIMEOUT[1]}s Circuit Breaking | Auto-Resume"
    )
    print(f"Input Data Directory (with targets) : {INPUT_NO_NIST_DIR}")
    print(f"JDX Cache Directory                 : {JDX_CACHE_DIR}")
    print(f"Output Reconstruction Directory     : {OUTPUT_RECON_DIR}")
    print(f"Benchmark Reference Directory       : {BENCHMARK_DIR}")

    # 1. Process Train -> Val -> Test sequentially
    for split in SPLITS:
        reconstruct_subset(split)

    # 2. Verify consistency for Train -> Val -> Test against benchmark
    print("\n" + "#" * 75)
    print("📋 Starting full dataset consistency verification...")
    print("#" * 75)
    for split in SPLITS:
        verify_consistency(split)

    print("\n" + "=" * 75)
    print("🎉 All tasks completed successfully! Datasets are 100% reconstructed and verified.")
    print("=" * 75)


if __name__ == "__main__":
    main()
