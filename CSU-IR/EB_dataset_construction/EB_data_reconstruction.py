"""NIST IR Two-Stage High-Performance Reconstruction Script

Architecture Overview:
[Stage 1: Batch Download NIST JDX Files]
  - 2.0s request interval with relaxed timeout protection and proxy support.
  - Automatically scans for missing JDX files with multi-round resume capability until all required files are downloaded.
[Stage 2: Local Offline Batch Processing & None Replacement]
  - Operates completely offline: applies unified physical preprocessing pipeline, absorbance conversion, and tensor assembly.
  - Replaces placeholders in the original dataset and saves reconstructed outputs ({split}_ir.pt, {split}_labels.txt, {split}_smiles.txt) into the designated directory.
"""

import os
import shutil
import time
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from scipy.interpolate import interp1d
import torch
from tqdm import tqdm
from urllib3.exceptions import InsecureRequestWarning

# ================= 1. Network & Connection Pool Configuration =================
# Disable SSL certificate verification warnings
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

# Request delay and relaxed timeout configuration
SAFE_DELAY = 1.3
REQUEST_TIMEOUT = (10.0, 25.0)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}


def create_persistent_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    session.trust_env = True
    adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


GLOBAL_SESSION = create_persistent_session()

# ================= 2. Relative Path Configuration =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSU_IR_DIR = os.path.dirname(SCRIPT_DIR)
DATA_BASE_DIR = os.path.join(
    CSU_IR_DIR, "data", "Multi-staged_training_data", "Experimental_Benching_data", "EB"
)

# A. Input: Directory containing datasets with placeholder None IR
INPUT_NO_NIST_DIR = os.path.join(DATA_BASE_DIR, "data_without_NIST_IR")

# B. Cache: Directory for caching downloaded raw JDX files
JDX_CACHE_DIR = os.path.join(DATA_BASE_DIR, "NIST_IR_raw_file_download")
os.makedirs(JDX_CACHE_DIR, exist_ok=True)

# C. Output: Directory for saving reconstructed datasets
OUTPUT_RECON_DIR = os.path.join(DATA_BASE_DIR, "data_with_NIST_IR")
os.makedirs(OUTPUT_RECON_DIR, exist_ok=True)

SPLITS = ["train", "val", "test"]


# ================= 3. Unified Standard Preprocessing Pipeline =================
def unified_preprocess_spectrum(
    wavenumbers,
    intensities,
    declared_yunits="",
    target_start=500.0,
    target_end=4000.0,
    num_points=3500,
):
    """Unified standard physical preprocessing pipeline for IR spectra.

    Strictly applies: Truncation of out-of-range ends, flat-padding with edge values 
    for missing ends, linear interpolation within the measured region, and normalization to [0, 1].
    """
    try:
        wn = np.array(wavenumbers, dtype=np.float64)
        spec = np.array(intensities, dtype=np.float64)

        # 1. Filter out invalid/non-finite data points
        valid_mask = np.isfinite(wn) & np.isfinite(spec)
        wn = wn[valid_mask]
        spec = spec[valid_mask]
        if len(wn) < 10:
            return None

        # 2. Sort in ascending order
        sort_idx = np.argsort(wn)
        wn = wn[sort_idx]
        spec = spec[sort_idx]

        # 3. Determine spectrum type (Transmittance vs. Absorbance)
        y_max = np.max(spec)
        y_median = np.median(spec)
        declared_upper = declared_yunits.upper()

        is_transmittance = False
        if any(t in declared_upper for t in ["TRANSMITTANCE", "%T", "PERCENT"]):
            is_transmittance = True
        elif any(a in declared_upper for a in ["ABSORBANCE", "ABS", "OD"]):
            is_transmittance = False
        else:
            if (y_max > 10.0 and y_median >= 40.0) or (
                y_median < 40.0 and y_max >= 95.0
            ):
                is_transmittance = True
            elif y_max <= 1.5 and y_median >= 0.6:
                is_transmittance = True

        # 4. Convert to absorbance (including removal of zero-transmittance dead zones)
        if is_transmittance:
            pos_mask = spec > 1e-5
            wn = wn[pos_mask]
            spec = spec[pos_mask]
            if len(wn) < 10:
                return None

            if np.max(spec) > 2.0:
                spec = spec / 100.0

            spec_clipped = np.clip(spec, 1e-4, 1.0)
            abs_spec = -np.log10(spec_clipped)
        else:
            abs_spec = spec - np.min(spec) if np.min(spec) < 0 else spec

        # Remove duplicates to ensure strictly monotonically increasing wavenumbers
        unique_wn, unique_idx = np.unique(wn, return_index=True)
        unique_abs = abs_spec[unique_idx]

        # 5. Enforce target range truncation and boundary padding (500–4000 cm^-1)
        target_grid = np.linspace(target_start, target_end, num_points)
        orig_min = unique_wn[0]
        orig_max = unique_wn[-1]

        resampled_abs = np.zeros(num_points, dtype=np.float64)

        # 5.1 Below the actual start wavenumber: flat-pad using the first valid measured value
        left_mask = target_grid < orig_min
        resampled_abs[left_mask] = unique_abs[0]

        # 5.2 Above the actual end wavenumber: flat-pad using the last valid measured value
        right_mask = target_grid > orig_max
        resampled_abs[right_mask] = unique_abs[-1]

        # 5.3 Points within the measured range: linear interpolation
        overlap_mask = (~left_mask) & (~right_mask)
        if np.any(overlap_mask):
            interp_func = interp1d(unique_wn, unique_abs, kind="linear")
            resampled_abs[overlap_mask] = interp_func(target_grid[overlap_mask])

        # 6. Min-Max normalization to [0, 1]
        min_v = np.min(resampled_abs)
        max_v = np.max(resampled_abs)
        if max_v > min_v:
            norm_abs = (resampled_abs - min_v) / (max_v - min_v)
        else:
            norm_abs = np.zeros_like(resampled_abs)

        return norm_abs

    except Exception:
        return None


# ================= 4. Stage 1: Network Download Functions =================
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
        res = GLOBAL_SESSION.get(url, timeout=REQUEST_TIMEOUT, verify=False)
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


# ================= 5. Stage 2: Offline Batch Processing =================
def parse_jdx_spectra(file_path):
    x_raw, y_raw = [], []
    xunits = "WAVENUMBERS"
    yunits = ""
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
                elif line.startswith("##YUNITS="):
                    yunits = line.split("=", 1)[1].strip().upper()
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

    return x_raw, y_raw, xunits, yunits


def process_local_jdx_file(sample_id):
    local_jdx_path = os.path.join(JDX_CACHE_DIR, f"{sample_id}.jdx")
    x_raw, y_raw, xunits, yunits = parse_jdx_spectra(local_jdx_path)
    if not x_raw:
        raise ValueError(f"Failed to parse local JDX file: {local_jdx_path}")

    x_arr = np.array(x_raw, dtype=np.float64)
    y_arr = np.array(y_raw, dtype=np.float64)

    # Convert Micrometers/Microns to Wavenumber (cm^-1) if needed
    if "MICR" in xunits:
        x_arr = 10000.0 / x_arr

    # Invoke unified preprocessing function
    processed_spec = unified_preprocess_spectrum(
        wavenumbers=x_arr,
        intensities=y_arr,
        declared_yunits=yunits,
        target_start=500.0,
        target_end=4000.0,
        num_points=3500,
    )

    if processed_spec is None:
        raise ValueError(f"Unified preprocessing failed for sample: {sample_id}")

    return torch.tensor(processed_spec, dtype=torch.float32)


# ================= 6. Dataset Reconstruction Pipeline =================
def load_annotated_labels(split_name):
    """Reads 2-column format (source_label, file_name) line by line from {split}_labels.txt."""
    file_path = os.path.join(INPUT_NO_NIST_DIR, f"{split_name}_labels.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Label file not found: {file_path}")

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            # 支持以制表符 \t 分隔或空格分隔两列
            if "\t" in line_str:
                parts = line_str.split("\t", 1)
            else:
                parts = line_str.split(None, 1)

            label = parts[0].strip()
            file_name = parts[1].strip() if len(parts) > 1 else "None"

            # 对第二列进行提取处理：若为 NIST 则剥离后缀保留基础 sample_id
            if label.upper() == "NIST" and file_name.lower() != "none":
                # 去除可能的 .jdx / .JDX 后缀以及多余的路径前缀
                base_name = os.path.basename(file_name)
                sid = os.path.splitext(base_name)[0]
            else:
                sid = file_name

            records.append((label, sid))
    return records


def reconstruct_subset(split_name):
    print("\n" + "=" * 75)
    print(f"▶ Reconstructing Subset: [{split_name.upper()} Set]")
    print("=" * 75)

    src_ir = os.path.join(INPUT_NO_NIST_DIR, f"{split_name}_ir.pt")
    src_labels = os.path.join(INPUT_NO_NIST_DIR, f"{split_name}_labels.txt")
    src_smiles = os.path.join(INPUT_NO_NIST_DIR, f"{split_name}_smiles.txt")

    source_records = load_annotated_labels(split_name)
    ir_list = torch.load(src_ir, map_location="cpu")

    total_count = len(source_records)
    assert (
        len(ir_list) == total_count
    ), f"Mismatch in sample counts! Labels: {total_count}, IR entries: {len(ir_list)}"

    # Extract all NIST sample IDs
    nist_sample_ids = [
        sid
        for lbl, sid in source_records
        if lbl.upper() == "NIST" and sid.lower() != "none"
    ]

    # --- Stage 1: Batch Download ---
    ensure_all_jdx_downloaded(split_name, nist_sample_ids)

    # --- Stage 2: Local Batch Processing & Replacement ---
    print(
        f"\n⚡ [Stage 2: Local Offline Batch Processing] Applying unified preprocess and assembling tensors..."
    )

    parsed_cache = {}
    reconstructed_tensors = []

    for i in tqdm(range(total_count), desc=f"Processing {split_name.upper()}"):
        lbl, sid = source_records[i]
        curr_ir = ir_list[i]

        if lbl.upper() == "NIST" and sid.lower() != "none":
            if sid not in parsed_cache:
                parsed_cache[sid] = process_local_jdx_file(sid)
            reconstructed_tensors.append(parsed_cache[sid])
        else:
            if isinstance(curr_ir, torch.Tensor):
                reconstructed_tensors.append(curr_ir.float())
            else:
                reconstructed_tensors.append(
                    torch.tensor(curr_ir, dtype=torch.float32)
                )

    # ③ 将列表堆叠为一整个二维 torch.Tensor: shape [N, 3500]
    reconstructed_tensor = torch.stack(reconstructed_tensors, dim=0)

    # Define output file paths
    dst_ir = os.path.join(OUTPUT_RECON_DIR, f"{split_name}_ir.pt")
    dst_labels = os.path.join(OUTPUT_RECON_DIR, f"{split_name}_labels.txt")
    dst_smiles = os.path.join(OUTPUT_RECON_DIR, f"{split_name}_smiles.txt")

    # Save reconstructed tensor and copy label/smiles files
    torch.save(reconstructed_tensor, dst_ir)
    shutil.copy2(src_labels, dst_labels)
    if os.path.exists(src_smiles):
        shutil.copy2(src_smiles, dst_smiles)

    print(
        f"✅ [{split_name.upper()} Set] Successfully reconstructed and saved to: {OUTPUT_RECON_DIR}"
    )
    print(f"    - Final IR Tensor Type : {type(reconstructed_tensor)}")
    print(f"    - Final IR Tensor Shape: {reconstructed_tensor.shape}")
    print(f"    - Final IR Tensor Dtype: {reconstructed_tensor.dtype}")


# ================= 7. Main Execution Entrypoint =================
def main():
    print(
        "🚀 Launching NIST IR Reconstruction Pipeline (Batch Download -> Processing -> Tensor Replacement)..."
    )
    print(
        f"Network Config: Safe Delay {SAFE_DELAY}s | Strict {REQUEST_TIMEOUT[0]+REQUEST_TIMEOUT[1]}s Circuit Breaking | Auto-Resume"
    )
    print(f"Input Data Directory            : {INPUT_NO_NIST_DIR}")
    print(f"JDX Cache Directory             : {JDX_CACHE_DIR}")
    print(f"Output Reconstruction Directory : {OUTPUT_RECON_DIR}")

    for split in SPLITS:
        reconstruct_subset(split)

    print("\n" + "=" * 75)
    print("🎉 All tasks successfully completed!")
    print("=" * 75)


if __name__ == "__main__":
    main()
