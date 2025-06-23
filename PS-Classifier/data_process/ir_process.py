import numpy as np
from scipy.interpolate import interp1d


def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# define ir process functions
def preprocess_spectra_higer_500(wavenumbers, transmittances, method='cubic'):
    wavenumbers = np.array(wavenumbers, dtype=float)
    transmittances = np.array(transmittances, dtype=float)
    valid_indices = np.where(transmittances != 0)[0]
    wavenumbers = wavenumbers[valid_indices]
    transmittances = transmittances[valid_indices]
    num_points = int(4000 - wavenumbers[0])
    target_wavenumbers = np.linspace(wavenumbers[0], wavenumbers[-1], num_points)
    interpolator = interp1d(wavenumbers, transmittances, kind=method)
    interpolated_transmittances = interpolator(target_wavenumbers)
    baseline_value = find_baseline(transmittances)
    num_zeros = max(int((wavenumbers[0] - 0.000000001) - 500) + 1, 0)
    y_points = [interpolated_transmittances[0], baseline_value]
    x_new = np.linspace(0, num_zeros - 1, num_zeros)
    a = y_points[0] - y_points[1]
    b = 0.03
    c = y_points[1]
    y_fit = exp_func(x_new, a, b, c)
    y_fit_flipped = np.flip(y_fit)
    padded_transmittances = np.concatenate((y_fit_flipped, interpolated_transmittances))
    padded_transmittances = (padded_transmittances - np.min(padded_transmittances)) / (
            np.max(padded_transmittances) - np.min(padded_transmittances))
    if np.any(np.isnan(padded_transmittances)):
        print("NaN values found in normalized absorbances")
    if len(padded_transmittances) != 3500:
        raise ValueError("The number of filled data points is not equal to 3500")
    return padded_transmittances

def preprocess_spectra_lower_500(filename, wavenumbers, transmittances, method='cubic'):
    wavenumbers = np.array(wavenumbers, dtype=float)
    transmittances = np.array(transmittances, dtype=float)
    valid_indices = np.where(transmittances != 0)[0]
    wavenumbers = wavenumbers[valid_indices]
    transmittances = transmittances[valid_indices]
    closest_index = np.argmin(np.abs(wavenumbers - 500))
    wavenumbers = wavenumbers[closest_index:]
    transmittances = transmittances[closest_index:]
    num_points = 3500
    target_wavenumbers = np.linspace(wavenumbers[0], wavenumbers[-1], num_points)
    interpolator = interp1d(wavenumbers, transmittances, kind=method)
    interpolated_transmittances = interpolator(target_wavenumbers)
    padded_transmittances = (interpolated_transmittances - np.min(interpolated_transmittances)) / (
            np.max(interpolated_transmittances) - np.min(interpolated_transmittances))
    if np.any(np.isnan(padded_transmittances)):
        print("NaN values found in normalized absorbances")
        print(filename)
    if len(padded_transmittances) != 3500:
        raise ValueError("The number of filled data points is not equal to 3500")
    return padded_transmittances

def find_baseline(spectrum, window_size=50):
    local_minima = [np.min(spectrum[i:i + window_size]) for i in range(0, len(spectrum), window_size)]
    baseline_value = np.median(local_minima)
    return baseline_value