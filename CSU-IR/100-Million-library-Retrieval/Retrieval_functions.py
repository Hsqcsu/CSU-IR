'''
Retrieval functions, spectral preprocessing, and chunked database management
for 100-Million-Scale IR spectrum retrieval.
Supports IR-Only, MW-Filtered (Exact Mass with tolerance), and Formula-Filtered modes.
'''

import os
import hashlib
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.interpolate import interp1d

FEATURE_DIM = 1024
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== 1. Unified Spectral Preprocessing ====================
def read_native_jdx_file(file_path):
    """
    Robust native JDX spectrum parser aligned with training dataset preprocessing.
    Extracts wavenumbers, intensities, and declared Y units with full factor scaling.
    """
    x_raw = []
    y_raw = []
    xunits = "WAVENUMBERS"
    yunits = ""
    xfactor = 1.0
    yfactor = 1.0
    deltax = None
    in_xydata = False

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('##XUNITS='):
                xunits = line.split('=', 1)[1].strip().upper()
            elif line.startswith('##YUNITS='):
                yunits = line.split('=', 1)[1].strip().upper()
            elif line.startswith('##XFACTOR='):
                try:
                    xfactor = float(line.split('=', 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith('##YFACTOR='):
                try:
                    yfactor = float(line.split('=', 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith('##DELTAX='):
                try:
                    deltax = float(line.split('=', 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith('##XYDATA=') or line.startswith('##XYPOINTS='):
                in_xydata = True
                continue
            elif line.startswith('##END=') or (line.startswith('##') and in_xydata):
                in_xydata = False

            if in_xydata:
                parts = line.replace(',', ' ').split()
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

    if len(x_raw) == 0:
        raise ValueError("No valid spectral XY data points found in JDX file.")

    x_arr = np.array(x_raw, dtype=float)
    y_arr = np.array(y_raw, dtype=float)

    # Automatic micrometer-to-wavenumber conversion
    if "MICR" in xunits or np.nanmax(x_arr) < 100.0:
        x_arr = 10000.0 / x_arr

    return x_arr, y_arr, yunits




def unified_preprocess_spectrum(
    wavenumbers,
    intensities,
    declared_yunits="",
    target_start=500.0,
    target_end=4000.0,
    num_points=3500,
    bypass_domain_conversion=False,
    interp_kind="linear",
):
    """Unified standard physical preprocessing pipeline for IR spectra.

    Strictly applies: Data cleaning/sorting, unification into the absorbance domain 
    (bypassed for pre-converted SDBS data and natively generated simulated absorbance spectra), 
    truncation of out-of-range ends, flat-padding with edge values for missing ends, 
    interpolation within the measured region, and Min-Max normalization to [0, 1].
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

        # Note: Natively generated simulated absorbance spectra and pre-converted SDBS 
        # absorbance spectra bypass the transmittance-to-absorbance conversion steps (Steps 3 & 4)
        # and proceed directly to grid resampling and normalization (Steps 5 & 6).
        if bypass_domain_conversion:
            abs_spec = spec
        else:
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

        # 5. Enforce target range truncation and boundary padding (e.g., 500–4000 cm^-1)
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

        # 5.3 Points within the measured range: interpolate onto the uniform target grid
        # Default 'linear' for experimental spectra to prevent overshooting on noisy baselines;
        # higher-order (e.g., 'cubic') is also supported for smooth theoretical/simulated profiles.
        overlap_mask = (~left_mask) & (~right_mask)
        if np.any(overlap_mask):
            interp_func = interp1d(unique_wn, unique_abs, kind=interp_kind)
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


def process_ir(ir_file_path, model_infer_instance):
    """Parses IR spectrum file and generates normalized embedding vector."""
    declared_yunits = ""
    if ir_file_path.lower().endswith('.csv'):
        df = pd.read_csv(ir_file_path, header=None)
        wavenumbers = df.iloc[:, 0].values
        intensities = df.iloc[:, 1].values

        if np.nanmax(wavenumbers) < 100.0 and np.nanmax(wavenumbers) > 0:
            wavenumbers = 10000.0 / wavenumbers

    elif ir_file_path.lower().endswith('.jdx'):
        wavenumbers, intensities, declared_yunits = read_native_jdx_file(ir_file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a .jdx or .csv file.")

    ir_data = unified_preprocess_spectrum(wavenumbers, intensities, declared_yunits=declared_yunits)
    if ir_data is None:
        raise ValueError("Failed to process spectrum. Please ensure sufficient valid data points.")

    ir_spectra_tensor = torch.tensor(ir_data, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        ir_feature = model_infer_instance.ir_encode(ir_spectra_tensor)
        ir_feature = F.normalize(ir_feature, p=2, dim=-1)
    return ir_feature


# ==================== 2. Fast Chunk & Meta Utilities ====================
def formula_to_hash_id(formula_str):
    return int(hashlib.md5(formula_str.encode('utf-8')).hexdigest()[:15], 16)


def load_chunk_meta_fast(tsv_path):
    cache_mw = tsv_path.replace(".tsv", "_mw.npy")
    cache_hash = tsv_path.replace(".tsv", "_hash.npy")
    if os.path.exists(cache_mw) and os.path.exists(cache_hash):
        return np.load(cache_mw), np.load(cache_hash)

    df = pd.read_csv(tsv_path, sep='\t', header=None, usecols=[2, 3], names=['mw', 'form'], dtype={'mw': np.float32, 'form': str})
    df['mw'] = df['mw'].fillna(0.0)
    df['form'] = df['form'].fillna("N/A")
    bg_mws = df['mw'].values.astype(np.float32)
    bg_formulas_hash = np.array([formula_to_hash_id(f) for f in df['form']], dtype=np.int64)
    np.save(cache_mw, bg_mws)
    np.save(cache_hash, bg_formulas_hash)
    return bg_mws, bg_formulas_hash

# ==================== 3. 100M Library Manager with Inverted Index ====================
class Chunked100MLibraryManager:
    """
    High-performance indexed library manager for 100M chunks.
    Builds in-memory bucket indices on startup for instant (sub-second) MW and Formula lookups.
    """
    def __init__(self, db_dir, num_chunks=54):
        import time
        from tqdm import tqdm

        self.db_dir = db_dir
        self.chunk_specs = []
        self.mmaps = []
        self.mw_arrays = []
        self.form_hash_arrays = []
        self.chunk_sizes = []

        self.mw_bucket_to_records = defaultdict(list)
        self.form_hash_to_records = defaultdict(list)

        print("\n" + "=" * 70)
        print("[Init] Starting 100M Library Inverted Index Construction...")
        print("=" * 70)
        overall_start_time = time.time()

        valid_chunk_indices = []
        for i in range(num_chunks):
            tsv_path = os.path.join(db_dir, f"meta_chunk_{i}.tsv")
            dat_path = os.path.join(db_dir, f"features_chunk_{i}.dat")
            if os.path.exists(tsv_path) and os.path.exists(dat_path):
                valid_chunk_indices.append(i)

        total_valid = len(valid_chunk_indices)
        print(f"[Init] Detected {total_valid} valid chunks in database directory.")

        with tqdm(total=total_valid, desc="[Indexing Chunks]", unit="chunk", ncols=85) as pbar:
            for i in valid_chunk_indices:
                tsv_path = os.path.join(db_dir, f"meta_chunk_{i}.tsv")
                dat_path = os.path.join(db_dir, f"features_chunk_{i}.dat")

                bg_mws, bg_form_hashes = load_chunk_meta_fast(tsv_path)
                c_size = len(bg_mws)
                self.chunk_sizes.append(c_size)

                mmap = np.memmap(dat_path, dtype=np.float16, mode='r', shape=(c_size, FEATURE_DIM))
                self.mmaps.append(mmap)
                self.mw_arrays.append(bg_mws)
                self.form_hash_arrays.append(bg_form_hashes)

                self.chunk_specs.append({
                    "chunk_idx": i,
                    "tsv_path": tsv_path,
                    "dat_path": dat_path,
                    "size": c_size
                })

                # High-speed O(N log N) vectorized bucket indexing via sorting
                # 1. MW Buckets
                rounded_mws = np.rint(bg_mws).astype(np.int32)
                sort_idx_mw = np.argsort(rounded_mws)
                sorted_mw = rounded_mws[sort_idx_mw]
                unique_mws, split_indices_mw = np.unique(sorted_mw, return_index=True)
                grouped_mw_locs = np.split(sort_idx_mw, split_indices_mw[1:])
                for m_val, locs in zip(unique_mws, grouped_mw_locs):
                    self.mw_bucket_to_records[int(m_val)].append((i, locs))

                # 2. Formula Hash Buckets
                sort_idx_form = np.argsort(bg_form_hashes)
                sorted_form = bg_form_hashes[sort_idx_form]
                unique_forms, split_indices_form = np.unique(sorted_form, return_index=True)
                grouped_form_locs = np.split(sort_idx_form, split_indices_form[1:])
                for h_val, locs in zip(unique_forms, grouped_form_locs):
                    self.form_hash_to_records[int(h_val)].append((i, locs))

                pbar.update(1)

        total_elapsed = time.time() - overall_start_time
        self.total_size = sum(self.chunk_sizes)
        print("=" * 70)
        print(f"[Init] Indexing Complete!")
        print(f"       Total Compounds Indexed : {self.total_size:,}")
        print(f"       Unique MW Buckets       : {len(self.mw_bucket_to_records):,}")
        print(f"       Unique Formula Buckets  : {len(self.form_hash_to_records):,}")
        print(f"       Total Time Elapsed      : {total_elapsed:.2f} seconds ({total_elapsed/60.0:.2f} mins)")
        print("=" * 70 + "\n")


def unified_retrieval_100M(q_ir, lib_manager, mw=None, formula=None, top_k=100, search_range=None, mw_tolerance=0.5):
    """
    Sub-second candidate retrieval function utilizing inverted index for MW/Formula filtering,
    and streaming memmap scans for IR-Only queries.
    """
    q_ir = q_ir.to(device).half()
    if q_ir.dim() == 1:
        q_ir = q_ir.unsqueeze(0)

    has_formula = bool(formula and str(formula).strip())
    has_mw = bool(mw and str(mw).strip())

    # ================= 1. Instant Formula Filtered Mode =================
    if has_formula:
        target_hash = formula_to_hash_id(str(formula).strip())
        matched_records = lib_manager.form_hash_to_records.get(target_hash, [])
        if not matched_records:
            return []

        cand_feats_list = []
        cand_meta_coords = []  # (chunk_idx, local_idx)
        for c_idx, locs in matched_records:
            cand_feats_list.append(np.array(lib_manager.mmaps[c_idx][locs], dtype=np.float16))
            for l in locs:
                cand_meta_coords.append((c_idx, l))

        all_cand_feats = np.concatenate(cand_feats_list, axis=0)
        cand_feats_gpu = torch.from_numpy(all_cand_feats).to(device)

        sims = torch.matmul(q_ir, cand_feats_gpu.t()).squeeze(0)
        actual_k = min(top_k, len(sims))
        top_scores, top_idx = torch.topk(sims, k=actual_k)

        top_scores = top_scores.cpu().numpy()
        top_idx = top_idx.cpu().numpy()

        results = []
        for rank_pos in range(actual_k):
            c_idx, local_l = cand_meta_coords[top_idx[rank_pos]]
            tsv_path = lib_manager.chunk_specs[c_idx]["tsv_path"]
            df_row = pd.read_csv(tsv_path, sep='\t', header=None, skiprows=local_l, nrows=1, names=['cid', 'smiles', 'mw', 'form']).iloc[0]
            results.append({
                "rank": rank_pos + 1,
                "similarity": float(top_scores[rank_pos]),
                "smiles": str(df_row['smiles']),
                "formula": str(df_row['form']),
                "mw": float(df_row['mw'])
            })
        return results

    # ================= 2. Instant MW Filtered Mode =================
    if has_mw:
        target_mw = float(str(mw).strip())
        min_mw = target_mw - mw_tolerance
        max_mw = target_mw + mw_tolerance
        min_b = int(np.floor(min_mw))
        max_b = int(np.ceil(max_mw))

        cand_feats_list = []
        cand_meta_coords = []

        for b in range(min_b, max_b + 1):
            if b in lib_manager.mw_bucket_to_records:
                for c_idx, locs in lib_manager.mw_bucket_to_records[b]:
                    actual_mws = lib_manager.mw_arrays[c_idx][locs]
                    valid = np.abs(actual_mws - target_mw) <= mw_tolerance
                    valid_locs = locs[valid]
                    if len(valid_locs) > 0:
                        cand_feats_list.append(np.array(lib_manager.mmaps[c_idx][valid_locs], dtype=np.float16))
                        for l in valid_locs:
                            cand_meta_coords.append((c_idx, l))

        if not cand_feats_list:
            return []

        all_cand_feats = np.concatenate(cand_feats_list, axis=0)
        cand_feats_gpu = torch.from_numpy(all_cand_feats).to(device)

        sims = torch.matmul(q_ir, cand_feats_gpu.t()).squeeze(0)
        actual_k = min(top_k, len(sims))
        top_scores, top_idx = torch.topk(sims, k=actual_k)

        top_scores = top_scores.cpu().numpy()
        top_idx = top_idx.cpu().numpy()

        results = []
        for rank_pos in range(actual_k):
            c_idx, local_l = cand_meta_coords[top_idx[rank_pos]]
            tsv_path = lib_manager.chunk_specs[c_idx]["tsv_path"]
            df_row = pd.read_csv(tsv_path, sep='\t', header=None, skiprows=local_l, nrows=1, names=['cid', 'smiles', 'mw', 'form']).iloc[0]
            results.append({
                "rank": rank_pos + 1,
                "similarity": float(top_scores[rank_pos]),
                "smiles": str(df_row['smiles']),
                "formula": str(df_row['form']),
                "mw": float(df_row['mw'])
            })
        return results

    # ================= 3. Streaming IR-Only Mode =================
    running_vals = torch.full((1, top_k), -float('inf'), device=device, dtype=torch.float16)
    running_idxs = torch.full((1, top_k), -1, dtype=torch.long, device=device)

    total_scanned = 0
    max_scan = search_range if search_range is not None else lib_manager.total_size

    for c_idx, spec in enumerate(lib_manager.chunk_specs):
        if total_scanned >= max_scan:
            break

        c_size = spec["size"]
        remaining = max_scan - total_scanned
        limit = min(c_size, remaining)

        bg_feats = np.fromfile(spec["dat_path"], dtype=np.float16).reshape(-1, FEATURE_DIM)[:limit]
        bg_feats_gpu = torch.from_numpy(bg_feats).to(device)

        sims = torch.matmul(q_ir, bg_feats_gpu.t())
        k = min(top_k, sims.shape[1])
        vals, idxs = torch.topk(sims, k=k, dim=1)
        global_idxs = idxs + total_scanned

        comb_vals = torch.cat([running_vals, vals], dim=1)
        comb_idxs = torch.cat([running_idxs, global_idxs], dim=1)
        new_vals, topk_idx = torch.topk(comb_vals, k=top_k, dim=1)
        new_idxs = torch.gather(comb_idxs, dim=1, index=topk_idx)
        running_vals.copy_(new_vals)
        running_idxs.copy_(new_idxs)

        total_scanned += limit
        del bg_feats_gpu, bg_feats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_vals = running_vals[0].cpu().numpy()
    final_idxs = running_idxs[0].cpu().numpy()

    valid_mask = (final_idxs != -1) & np.isfinite(final_vals)
    final_vals = final_vals[valid_mask]
    final_idxs = final_idxs[valid_mask]

    if len(final_idxs) == 0:
        return []

    offsets = [0]
    for s in lib_manager.chunk_specs:
        offsets.append(offsets[-1] + s["size"])

    chunk_to_local = {i: [] for i in range(len(lib_manager.chunk_specs))}
    for rank_pos, g_idx in enumerate(final_idxs):
        c_i = np.searchsorted(offsets, g_idx, side='right') - 1
        local_l = g_idx - offsets[c_i]
        chunk_to_local[c_i].append((local_l, rank_pos))

    retrieved_meta = {}
    for c_i, records in chunk_to_local.items():
        if not records:
            continue
        records.sort(key=lambda x: x[0])
        tsv_path = lib_manager.chunk_specs[c_i]["tsv_path"]
        lines = [r[0] for r in records]
        df_meta = pd.read_csv(tsv_path, sep='\t', header=None, names=['cid', 'smiles', 'mw', 'form'])
        sub = df_meta.iloc[lines]
        for (local_l, rank_p), (_, row) in zip(records, sub.iterrows()):
            retrieved_meta[rank_p] = {
                "smiles": str(row['smiles']),
                "formula": str(row['form']),
                "mw": float(row['mw'])
            }

    results = []
    for rank_pos in range(len(final_idxs)):
        meta = retrieved_meta.get(rank_pos, {"smiles": "N/A", "formula": "N/A", "mw": 0.0})
        results.append({
            "rank": rank_pos + 1,
            "similarity": float(final_vals[rank_pos]),
            "smiles": meta["smiles"],
            "formula": meta["formula"],
            "mw": meta["mw"]
        })
    return results


# ==================== 4. Platt-Scaling Confidence Calibration ====================
def calculate_calibrated_confidence(cosine_sim: float, smiles_model_instance, scale: float = 0.2306, bias: float = -0.4529) -> float:
    with torch.no_grad():
        if hasattr(smiles_model_instance, 't_prime') and hasattr(smiles_model_instance, 'bias'):
            t = torch.exp(smiles_model_instance.t_prime).cpu().item()
            b = smiles_model_instance.bias.cpu().item()
        elif hasattr(smiles_model_instance, 'model_sm'):
            t = torch.exp(smiles_model_instance.model_sm.t_prime).cpu().item()
            b = smiles_model_instance.model_sm.bias.cpu().item()
        else:
            t = 10.0
            b = -10.0

    raw_logit = cosine_sim * t + b
    calibrated_logit = scale * raw_logit + bias
    confidence = 1.0 / (1.0 + np.exp(-calibrated_logit))
    return float(np.clip(confidence, 0.0, 1.0))
