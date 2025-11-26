import numpy as np
from scipy.signal import savgol_filter

from . import metadata

def split_by_jump(df, drop_factor=0.5):
    """
    Split IV segments when the current drops sharply within a file.
    drop_factor : float
        Fraction of the full in-file current range that defines a 'big drop'.
        0.5 means: if current decreases by >50% of total span so far, start new IV.
    """
    I = df["Current [A]"].to_numpy()
    F = df["File"].to_numpy()

    diffs = np.diff(I)
    split_idx = []

    running_max = I[0]
    running_min = I[0]
    for i in range(1, len(I)):
        running_max = max(running_max, I[i])
        running_min = min(running_min, I[i])
        full_range = running_max - running_min

        # detect a reset drop of more than 'drop_factor' of current span so far
        if full_range > 0 and (running_max - I[i]) > drop_factor * full_range:
            split_idx.append(i)
            running_max = I[i]
            running_min = I[i]

        if F[i] != F[i - 1]:
            split_idx.append(i)
            running_max = I[i]
            running_min = I[i]

    split_idx = np.unique(split_idx)
    segments = np.split(df.reset_index(drop=True), split_idx)
    return segments#, split_idx



def find_ROI(iv_file, index, flat_threshold=0.0005, center_fraction=0.9):
    """
    Determine the region of interest (ROI) around a specified index in the data.

    Parameters:
        iv_file: An object with .t and .I attributes, and .tops and .troths index arrays.
        index (int): Index around which to find the ROI (should be in iv_file.tops or iv_file.troths).
        option (str): ROI detection mode:
                      'smart'     – derivative-based edge detection (default),
                      'dirty'     – fixed ROI width from index,
                      'alt_dirty' – smaller fixed ROI width.
        flat_threshold (float): Threshold for |dI/dt| to consider a region flat (used in 'smart').
        center_fraction (float): Fraction of the flat region to keep around its center (used in 'smart').

    Returns:
        tuple: (lROI, hROI) indicating the bounds of the ROI.
    """
    t = iv_file.t
    I = iv_file.I

    tops = iv_file.tops
    troths = iv_file.troths

    # Define window boundaries
    window = tops[0]-troths[0]


    left = max(0, index - window)
    right = min(len(t) - 1, index + window)


    I_segment = I[left:right]
    t_segment = t[left:right]


    if len(I_segment) > 1024:
        # Smooth I_segment before differentiation for dense datasets
        I_segment = savgol_filter(I_segment, window_length=501, polyorder=1)

    if len(I_segment) < 2:
        return left, right

    dI = np.diff(I_segment)
    dt = np.diff(t_segment)
    dIdt = dI / dt

    rel_point = index - left
    rel_point = np.clip(rel_point, 0, len(dIdt) - 1)

    # Steepest left edge
    left_region = np.abs(dIdt[:rel_point])
    rel_start = np.argmax(left_region) if len(left_region) > 0 else 0
    start_idx = left + rel_start

    # Steepest right edge
    right_region = np.abs(dIdt[rel_point:])
    rel_end = np.argmax(right_region) if len(right_region) > 0 else 0
    end_idx = left + rel_point + rel_end

    # Flat region between edges
    flat_slice_start = max(start_idx - left, 0)
    flat_slice_end = max(end_idx - left, 0)
    flat_dIdt = dIdt[flat_slice_start:flat_slice_end]

    flat_mask = np.abs(flat_dIdt) < flat_threshold
    if np.any(flat_mask):
        flat_indices = np.where(flat_mask)[0]
        flat_start = left + flat_slice_start + flat_indices[0]
        flat_end = left + flat_slice_start + flat_indices[-1]
    else:
        flat_start = start_idx
        flat_end = end_idx

    # Region around flat center
    flat_width = flat_end - flat_start
    region_half_width = int((flat_width * center_fraction) // 2)
    flat_center = flat_start + flat_width // 2

    region_start = max(flat_center - region_half_width, 0)
    region_end = min(flat_center + region_half_width, len(t) - 1)

    return region_start, region_end # standard # TODO plotting
        # return flat_center, region_end # left only
        # return region_start, flat_center # rifht only


# def safe_mean(V):
#     if len(V) == 0:
#         mean = np.nan  #
#     if len(V) < 5:      # too short for SavGol
#         mean = np.mean(V)

#     # build a valid odd window length
#     w = min(len(V) - 1, int(len(V)//2 + 3) | 1)
#     if w <= 2:  # must be > polyorder
#         mean = np.mean(V)

#     p = min(1, w - 1)  # ensure polyorder < window_length

#     try:
#         volts = savgol_filter(V, window_length=w, polyorder=p)
#         mean = np.mean(volts)
#     except Exception:
#         volts = V
#         mean = np.mean(V)
#     return mean, volts

def safe_mean(V): # Avoid extra math
    volts = V
    mean = np.mean(V)
    return mean, volts




def process_IV_pulse(iv_file,top,left,right,excelname='250519_houston_log.xlsx'):
    '''
    Extracts the voltage, current, field and temperature for a single IV pulse iv_file class.
    '''
    lROI,hROI = find_ROI(iv_file,top)

    error_std = iv_file.noise_std
    error_rms = iv_file.noise_rms

    I = np.mean(iv_file.I[lROI:hROI])
    B = np.mean(iv_file.B[lROI:hROI])
    Bdiff = np.mean(np.diff(iv_file.B[lROI:hROI]))
    tdiff = np.mean(np.diff(iv_file.t[lROI:hROI]))
    dBdt = Bdiff/tdiff
    # dBdt = np.mean(np.diff(np.array([datafile.t[lROI:hROI],datafile.B[lROI:hROI]])))
    
    T = metadata.extract_numeric_temperature(iv_file.T)
    
    left_lROI,left_hROI = find_ROI(iv_file,left)
    right_lROI, right_hROI = find_ROI(iv_file,right)


    V, processed_V = safe_mean(iv_file.V[lROI:hROI])
    leftV, _ = safe_mean(iv_file.V[left_lROI:left_hROI])
    rightV, _ = safe_mean(iv_file.V[right_lROI:right_hROI])
    # print(leftV,V,rightV)
    V = V - (leftV+rightV)/2 # 1st option
    # V = V - leftV # 2nd option
    # V = V - rightV # 3rd option
    # if option == 'alt_dirty' or excelname=='250519_houston_log.xlsx':
    #     V += I*(0.00014286)

    result = {
        'Current [A]': I,
        'Voltage [V]': V,
        'Field [T]': np.round(B,4),
        'noise_std':error_std,
        'noise_rms':error_rms,
        'dBdt [T/s]': dBdt,
        'Temperature [K]': T,
        'lROI' : lROI,
        'rROI' : hROI,
        'Denoised Voltage Array [V]': processed_V
    }
    # print(leftV,V,rightV)
    return result,I,V,B,dBdt,T


def fit_power_law(x, y):
    """
    Fit y = a * x^b using linear regression in log-log space.
    
    Parameters:
        x (array-like): Independent variable (must be > 0). 
        y (array-like): Dependent variable (must be > 0).
    
    Returns:
        a (float): Prefactor in power law.
        b (float): Exponent.
        r2 (float): R² of the fit in log-log space.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Filter out invalid values
    mask = (x > 1e-9) & (y > 0)
    if np.count_nonzero(mask) < 2:
        raise ValueError("Not enough valid data points for log-log fit.")
    
    log_x = np.log(x[mask])
    log_y = np.log(y[mask])

    b, log_a = np.polyfit(log_x, log_y, 1)
    a = np.exp(log_a)

    return a, b

import statsmodels.api as sm

def fit_power_law_wls(x, y,weight_power=3):
    mask = (x > 0) & (y > 0)
    log_x = np.log(x[mask])
    log_y = np.log(y[mask])

    # Filter out invalid values
    if np.count_nonzero(mask) < 2:
        raise ValueError("Not enough valid data points for log-log fit.")
    
    # Weights: proportional to actual current
    # (low voltages have larger fractional noise)
    w = (x[mask]**weight_power) 

    X = sm.add_constant(log_x)
    model = sm.WLS(log_y, X, weights=w)
    results = model.fit()

    log_a = results.params[0]
    b = results.params[1]
    a = np.exp(log_a)
    return a, b#, results

import numpy as np
from scipy.optimize import curve_fit

def powerlaw(I, k, n):
    return k * I**n

def fit_power_law_curvefit(x, y, sigma=None):
    x = np.asarray(x)
    y = np.asarray(y)

    # Mask invalid
    mask = (x > 0) & (y >= 0)
    x = x[mask]
    y = y[mask]

    # Initial guesses
    n0 = 10
    k0 = y[np.argmax(x)] / x[np.argmax(x)]**n0

    popt, pcov = curve_fit(powerlaw, x, y, p0=[k0, n0], sigma=sigma, absolute_sigma=True)

    k, n = popt
    return k, n#, pcov

def compute_R2(x, y, a, b):
    """
    Compute R² for the power-law fit y = a * x^b.
    
    Parameters:
        x (array-like): Independent variable (must be > 0).
        y (array-like): Dependent variable (must be > 0).
        a (float): Prefactor from the fit.
        b (float): Exponent
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Filter out invalid values
    mask = (x > 1e-9) & (y > 0)
    if np.count_nonzero(mask) < 2:
        raise ValueError("Not enough valid data points for log-log fit.")
    
    log_x = np.log(x[mask])
    log_y = np.log(y[mask])

    # Compute R² in log-log space
    y_pred = np.log(a) + b * log_x
    ss_res = np.sum((log_y - y_pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return r2

import numpy as np

def compute_R2_weighted(x, y, a, b, weight_power=10):
    """
    Compute weighted R² for the power-law fit y = a * x^b.

    Parameters:
        x (array-like): Independent variable (must be > 0).
        y (array-like): Dependent variable (must be > 0).
        a (float): Prefactor from the fit.
        b (float): Exponent.
        weight_power (float): Power for weighting: weights = x**weight_power

    Returns:
        float: weighted R² in log-log space
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Filter out invalid values
    mask = (x > 1e-9) & (y > 0)
    if np.count_nonzero(mask) < 2:
        raise ValueError("Not enough valid data points for log-log fit.")
    
    log_x = np.log(x[mask])
    log_y = np.log(y[mask])

    # Weights proportional to x**weight_power
    w = log_x**0  # fallback
    w = x[mask]**weight_power

    # Predicted log-values
    y_pred = np.log(a) + b * log_x

    # Weighted sums of squares
    ss_res = np.sum(w * (log_y - y_pred)**2)
    ss_tot = np.sum(w * (log_y - np.sum(w*log_y)/np.sum(w))**2)

    r2_weighted = 1 - ss_res / ss_tot
    return r2_weighted


import numpy as np
from scipy.odr import ODR, Model, RealData

def fit_power_law_odr(x, y):
    mask = (x>0) & (y>0)
    log_x = np.log(x[mask])
    log_y = np.log(y[mask])

    def f(B, x):
        a, b = B
        return a + b * x

    # assume constant std in log(V) and log(I)
    data = RealData(log_x, log_y)
    model = Model(f)

    odr = ODR(data, model, beta0=[0.0, 1.0])
    out = odr.run()

    log_a, b = out.beta
    a = np.exp(log_a)
    return a, b

from scipy.stats import theilslopes

def fit_power_law_TheilSen(x, y):
    mask = (x>0)&(y>0)
    log_x = np.log(x[mask])
    log_y = np.log(y[mask])

    slope, intercept, _, _ = theilslopes(log_y, log_x)
    a = np.exp(intercept)
    b = slope
    return a, b

def try_fit_power_law(x, y):
    """
    Try fitting power law; return (a, b) or (None, None) on failure.
    
    Parameters:
        x (np.ndarray): Current values.
        y (np.ndarray): Voltage values.
    
    Returns:
        tuple: (a, b), or (None, None) if fit fails.
    """
    try:
        # return fit_power_law(x, y)
        return fit_power_law_wls(x,y)
        # return fit_power_law_odr(x,y)
        # return fit_power_law_TheilSen(x,y)
        # return fit_power_law_curvefit(x,y)
    except Exception:
        return None, None
    