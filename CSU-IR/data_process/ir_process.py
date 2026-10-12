import numpy as np
from scipy.interpolate import interp1d


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

        # 5.3 Points within the measured range: linear interpolation (out-of-bounds regions are truncated)
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
