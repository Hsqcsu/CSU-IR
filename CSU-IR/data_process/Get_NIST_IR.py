import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import re
import time
import pickle
import numpy as np
import requests
from scipy.interpolate import CubicSpline
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
from tqdm import tqdm


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

INPUT_DIR = os.path.join(PROJECT_ROOT, "data",  "Multi-staged_training_data", "Experimental_Benching_data")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data",  "Multi-staged_training_data", "Experimental_Benching_data","Complete_EB")



RAW_JDX_CACHE_DIR = os.path.join(OUTPUT_DIR, "cached_jdx_files")
FAILED_LOG_PATH = os.path.join(OUTPUT_DIR, "failed_nist_ids.txt")

PKL_FILES = ["eb_train.pkl", "eb_val.pkl", "eb_test.pkl"]

REQUEST_INTERVAL = 0.8
TIMEOUT = (10, 25)
MAX_RETRIES = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_JDX_CACHE_DIR, exist_ok=True)


def create_robust_session():
    session = requests.Session()
    retries = Retry(
        total=MAX_RETRIES,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Academic-Fetcher',
        'Connection': 'keep-alive'
    })
    return session


http_session = create_robust_session()


def download_nist_jdx_by_id(nist_id, save_path):
    direct_url = f"https://webbook.nist.gov/cgi/cbook.cgi?JCAMP={nist_id}&Type=IR"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = http_session.get(direct_url, timeout=TIMEOUT, verify=False)
            if response.status_code == 200:
                text = response.text
                if "##TITLE=" in text and ("##JCAMP-DX=" in text or "##DATA TYPE=" in text):
                    with open(save_path, 'w', encoding='utf-8', errors='ignore') as f:
                        f.write(text)
                    return True, "Direct download successful"
        except Exception:
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
                continue

    page_url = f"https://webbook.nist.gov/cgi/cbook.cgi?ID={nist_id}&Units=SI&Type=IR"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = http_session.get(page_url, timeout=TIMEOUT, verify=False)
            if response.status_code == 200:
                html = response.text
                match = re.search(r'href="(/cgi/cbook\.cgi\?JCAMP=[^"]+)"', html)
                if match:
                    full_jcamp_url = f"https://webbook.nist.gov{match.group(1)}"
                    jdx_resp = http_session.get(full_jcamp_url, timeout=TIMEOUT, verify=False)
                    if jdx_resp.status_code == 200 and "##TITLE=" in jdx_resp.text:
                        with open(save_path, 'w', encoding='utf-8', errors='ignore') as f:
                            f.write(jdx_resp.text)
                        return True, "HTML parsing and download successful"
                elif "No spectrum available" in html:
                    return False, "No digital data available from NIST"
        except Exception:
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
                continue

    return False, "Download timed out or not found."


def parse_jdx_spectra(file_path):
    x_raw, y_raw = [], []
    xunits = "WAVENUMBERS"
    xfactor, yfactor = 1.0, 1.0
    deltax = None
    in_xydata = False

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('##XUNITS='):
                    xunits = line.split('=', 1)[1].strip().upper()
                elif line.startswith('##XFACTOR='):
                    xfactor = float(line.split('=', 1)[1].strip())
                elif line.startswith('##YFACTOR='):
                    yfactor = float(line.split('=', 1)[1].strip())
                elif line.startswith('##DELTAX='):
                    deltax = float(line.split('=', 1)[1].strip())
                elif line.startswith('##XYDATA='):
                    in_xydata = True
                    continue
                elif line.startswith('##END=') or (line.startswith('##') and in_xydata):
                    in_xydata = False

                if in_xydata:
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        line_x = float(parts[0]) * xfactor
                        for k, val_str in enumerate(parts[1:]):
                            line_y = float(val_str) * yfactor
                            actual_x = line_x + k * deltax * xfactor if deltax is not None else line_x
                            x_raw.append(actual_x)
                            y_raw.append(line_y)
                    except ValueError:
                        continue
    except Exception:
        pass

    return x_raw, y_raw, xunits


def process_ir_data(x, y):
    min_x, max_x = x[0], x[-1]
    start, end = max(min_x, 500.0), min(max_x, 4000.0)

    if start >= end:
        raise ValueError("There is no overlap in the spectral wavenumber ranges.")

    mask = (x >= start) & (x <= end)
    x_crop, y_crop = x[mask], y[mask]

    if len(x_crop) < 4:
        raise ValueError("Too few valid points.")

    start_int, end_int = int(np.ceil(start)), int(np.floor(end))
    x_temp = np.arange(start_int, end_int + 1, 1.0)
    x_temp = np.clip(x_temp, x_crop[0], x_crop[-1])

    cs_crop = CubicSpline(x_crop, y_crop, extrapolate=True)
    y_temp = cs_crop(x_temp)

    x_full = np.arange(500, 4001, 1.0)
    y_full = np.zeros_like(x_full)

    idx_start, idx_end = start_int - 500, end_int - 500
    y_full[idx_start:idx_end + 1] = y_temp

    if idx_start > 0:
        y_full[:idx_start] = y_temp[0]
    if idx_end < 3500:
        y_full[idx_end + 1:] = y_temp[-1]

    x_target = np.linspace(500.0, 4000.0, 3500)
    cs_full = CubicSpline(x_full, y_full, extrapolate=True)
    return cs_full(x_target)


def convert_to_absorbance_single(y_data):
    spec = np.array(y_data, dtype=np.float64)

    min_val, max_val = np.min(spec), np.max(spec)
    norm_spec = (spec - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(spec)

    if np.median(norm_spec) > 0.45:
        spec_clipped = np.clip(norm_spec, 1e-4, 1.0)
        abs_spec = -np.log10(spec_clipped)
        min_val_abs, max_val_abs = np.min(abs_spec), np.max(abs_spec)
        final_spec = (abs_spec - min_val_abs) / (
                    max_val_abs - min_val_abs) if max_val_abs > min_val_abs else np.zeros_like(abs_spec)
    else:
        final_spec = norm_spec

    return final_spec.tolist()


def convert_jdx_to_processed_ir(jdx_file_path):
    x_raw, y_raw, xunits = parse_jdx_spectra(jdx_file_path)
    if not x_raw or not y_raw:
        return None

    x_raw_arr, y_raw_arr = np.array(x_raw), np.array(y_raw)
    x_conv = 10000.0 / x_raw_arr if "MICR" in xunits.upper() else x_raw_arr

    _, unique_indices = np.unique(x_conv, return_index=True)
    x_unique, y_unique = x_conv[unique_indices], y_raw_arr[unique_indices]

    sort_idx = np.argsort(x_unique)
    x_sorted, y_sorted = x_unique[sort_idx], y_unique[sort_idx]

    processed_y = process_ir_data(x_sorted, y_sorted)
    return convert_to_absorbance_single(processed_y)


def main():
    print("=" * 70)
    print("      EB Auto Reconstitution")
    print("=" * 70)

    failed_nist_ids = set()

    for pkl_filename in PKL_FILES:
        input_pkl_path = os.path.join(INPUT_DIR, pkl_filename)
        output_pkl_path = os.path.join(OUTPUT_DIR, pkl_filename)

        if os.path.exists(output_pkl_path):
            read_path = output_pkl_path
            print(f"\nReading existing output progress file (resume mode): {read_path} ...")
        elif os.path.exists(input_pkl_path):
            read_path = input_pkl_path
            print(f"\nReading de-identified source file (original file is read-only): {read_path} ...")
        else:
            print(f"⚠️ Skip files not found: {pkl_filename}")
            continue

        with open(read_path, 'rb') as f:
            data_dict = pickle.load(f)

        smiles_list = data_dict["smiles"]
        ir_list = data_dict["IR"]
        nist_id_list = data_dict["NIST_ID"]

        total_num = len(smiles_list)
        updated_cnt = 0
        skipped_already_filled = 0
        skipped_public_data = 0
        failed_cnt = 0

        processed_ir_cache = {}

        for i in tqdm(range(total_num), desc=f"Check and complete {pkl_filename}"):
            nist_id = nist_id_list[i]
            current_ir = ir_list[i]

            if nist_id is None:
                skipped_public_data += 1
                continue
            if current_ir is not None:
                skipped_already_filled += 1
                continue

            if nist_id in processed_ir_cache:
                ir_list[i] = processed_ir_cache[nist_id]
                updated_cnt += 1
                continue

            jdx_path = os.path.join(RAW_JDX_CACHE_DIR, f"{nist_id}.jdx")

            if not os.path.exists(jdx_path) or os.path.getsize(jdx_path) < 200:
                success, msg = download_nist_jdx_by_id(nist_id, jdx_path)
                time.sleep(REQUEST_INTERVAL)
                if not success:
                    failed_cnt += 1
                    failed_nist_ids.add(nist_id)
                    continue
            try:
                processed_spec = convert_jdx_to_processed_ir(jdx_path)
                if processed_spec is not None:
                    ir_list[i] = processed_spec
                    processed_ir_cache[nist_id] = processed_spec
                    updated_cnt += 1
                else:
                    failed_cnt += 1
                    failed_nist_ids.add(nist_id)
            except Exception:
                failed_cnt += 1
                failed_nist_ids.add(nist_id)

        data_dict["IR"] = ir_list
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[{pkl_filename}] Refactoring Completion Statistics:")
        print(f"  - New file save location: {output_pkl_path}")
        print(f"  - Total number of entries: {total_num}")
        print(f"  - Data requiring no processing: {skipped_public_data} items")
        print(f"  - Check and mark as processed (automatically skipped): {skipped_already_filled} items")
        print(f"  - Newly added this time: {updated_cnt} items")
        print(f"  - Failure/Missing Data: {failed_cnt} items")

    if failed_nist_ids:
        with open(FAILED_LOG_PATH, 'a', encoding='utf-8') as f:
            for fid in failed_nist_ids:
                f.write(f"{fid}\n")

    print("\n" + "=" * 70)
    print("🎉 Automated reconstruction and filling completed successfully! No modifications were made to the original file.")
    print(f"Reconstructed PKL files have been securely stored.:\n{OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
