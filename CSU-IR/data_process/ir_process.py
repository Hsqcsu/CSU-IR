import numpy as np
from scipy.interpolate import CubicSpline

def _resample_and_pad_spectrum(wavenumbers, intensities):
    x = np.array(wavenumbers, dtype=float)
    y = np.array(intensities, dtype=float)
    valid_mask = (~np.isnan(x)) & (~np.isnan(y)) & (y != 0)
    x = x[valid_mask]
    y = y[valid_mask]
    if len(x) < 4:
        return None
    if np.nanmax(x) < 100.0:
        x = 10000.0 / x
    _, unique_indices = np.unique(x, return_index=True)
    x = x[unique_indices]
    y = y[unique_indices]
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    min_x, max_x = x[0], x[-1]
    start = max(min_x, 500.0)
    end = min(max_x, 4000.0)
    if start >= end:
        return None
    mask = (x >= start) & (x <= end)
    x_crop = x[mask]
    y_crop = y[mask]
    if len(x_crop) < 4:
        return None
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
    y_full[idx_start:idx_end + 1] = y_temp
    if idx_start > 0:
        y_full[:idx_start] = y_temp[0]
    if idx_end < 3500:
        y_full[idx_end + 1:] = y_temp[-1]
    x_target = np.linspace(500.0, 4000.0, 3500)
    cs_full = CubicSpline(x_full, y_full, extrapolate=True)
    return cs_full(x_target)

def preprocess_csv_spectra_higer_500(wavenumbers, transmittances, method='cubic'):
    try:
        t = np.array(transmittances, dtype=float)
        if np.nanmax(t) > 1.5:
            t = t / 100.0
        t = np.clip(t, 1e-4, 1.0)
        abs_raw = -np.log10(t)
        processed_y = _resample_and_pad_spectrum(wavenumbers, abs_raw)
        if processed_y is None:
            return None
        min_val = np.min(processed_y)
        max_val = np.max(processed_y)
        if max_val > min_val:
            normalized = (processed_y - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(processed_y)
        if np.any(np.isnan(normalized)):
            return None
        return normalized
    except Exception:
        return None

def preprocess_csv_spectra_lower_than_500(wavenumbers, transmittances, method='cubic'):
    return preprocess_csv_spectra_higer_500(wavenumbers, transmittances, method=method)

def preprocess_jdx_spectra_higer_500(wavenumbers, intensities, method='cubic'):
    try:
        processed_y = _resample_and_pad_spectrum(wavenumbers, intensities)
        if processed_y is None:
            return None
        min_val = np.min(processed_y)
        max_val = np.max(processed_y)
        if max_val > min_val:
            norm_spec = (processed_y - min_val) / (max_val - min_val)
        else:
            norm_spec = np.zeros_like(processed_y)
        spec_median = np.median(norm_spec)
        if spec_median > 0.45:
            spec_clipped = np.clip(norm_spec, 1e-4, 1.0)
            abs_spec = -np.log10(spec_clipped)
            min_val_abs = np.min(abs_spec)
            max_val_abs = np.max(abs_spec)
            if max_val_abs > min_val_abs:
                final_spec = (abs_spec - min_val_abs) / (max_val_abs - min_val_abs)
            else:
                final_spec = np.zeros_like(abs_spec)
        else:
            final_spec = norm_spec
        if np.any(np.isnan(final_spec)):
            return None
        return final_spec
    except Exception:
        return None

def preprocess_jdx_spectra_lower_500(wavenumbers, intensities, method='cubic'):
    return preprocess_jdx_spectra_higer_500(wavenumbers, intensities, method=method)
