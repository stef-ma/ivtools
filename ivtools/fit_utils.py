import numpy as np
from scipy.signal import savgol_filter

from . import iv_io

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
    # return segments#, split_idx
    return [seg.copy(deep=True) for seg in segments] #safer



def find_ROI(iv_file, index, flat_threshold=0.0005, center_fraction=0.5):
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
    
    T = iv_io.extract_numeric_temperature(iv_file.T)
    
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



def powerlaw(I, k, n):
    return k * I**n
def powerlaw_inverted(vc,k,n):
    return (vc / k) ** (1 / n)

import statsmodels.api as sm

# def fit_power_law_wls(x, y, voltage_criterion = None, weight_power=3): 
#     """
#     Find parameters to describe non-linear IV behavior.
#     Fit V/Vc = (I/Ic)^n in log-log space using weighted least squares. 
#     Fit on y′ = n*x + c, for: 
#         x = log I
#         y′ = logV
#         c = k

#     Parameters
#     ----------
#     x : array-like
#         Independent variable (must be > 0).
#     y : array-like
#         Dependent variable (must be > 0).
#     voltage_criterion : float
#         Voltage criterion for Ic calulation. Included for compatability with alternative function.
#     weight_power : float
#         Exponent used when constructing weights.

#     Returns
#     -------
#     k : float
#         Pre exponent expected downstream
#     Ic : float
#         Critical current
#     n : float
#         Power-law exponent
#     """
#     # log fit on log V = log k + n*log I
#     mask = (x > 0) & (y > 0)
#     log_x = np.log(x[mask])
#     log_y = np.log(y[mask])

#     # Filter out invalid values
#     if np.count_nonzero(mask) < 2:
#         raise ValueError("Not enough valid data points for log-log fit.")
    
#     # Weights: proportional to actual current
#     # (low voltages have larger fractional noise)
#     w = (x[mask]**weight_power) 

#     X = sm.add_constant(log_x)
#     model = sm.WLS(log_y, X, weights=w)
#     results = model.fit()

#     log_a = results.params[0]
#     n = results.params[1]
#     k = np.exp(log_a)
#     ic = None
#     return k, n, ic#, results

# def fit_power_law_wls(x, y, voltage_criterion = None, weight_power=3,weight_mode='index'): 
#     """
#     Find parameters to describe non-linear IV behavior.
#     Fit V/Vc = (I/Ic)^n in log-log space using weighted least squares. 
#     Fit on y′ = n*x + c, for: 
#         x = log I
#         y′ = logV − logVc 
#         c = −n log Ic

#     Parameters
#     ----------
#     x : array-like
#         Independent variable (must be > 0).
#     y : array-like
#         Dependent variable (must be > 0).
#     voltage_criterion : float
#         Voltage criterion for Ic calulation.
#     weight_power : float
#         Exponent used when constructing weights.
#     weight_mode : str
#         "x"      -> weights = x**weight_power
#         "index"  -> weights = (1..N)**weight_power

#     Returns
#     -------
#     k : float
#         Pre exponent expected downstream
#     Ic : float
#         Critical current
#     n : float
#         Power-law exponent
#     """
    
#     mask = (x > 0) & (y > 0)
#     log_x = np.log(x[mask])
#     log_y = np.log(y[mask])
#     log_vc = np.log(voltage_criterion)

#     # Filter out invalid values
#     if np.count_nonzero(mask) < 2:
#         raise ValueError("Not enough valid data points for log-log fit.")
    
#     fit_y = log_y - log_vc
    
#     # # Weights: proportional to actual current
#     # # (low voltages have larger fra ctional noise)
#     # w = (x[mask]**weight_power) 


#     # ----------------------------------------------------------------------
#     # Weight construction
#     # ----------------------------------------------------------------------
#     if weight_mode == "x":
#         # w = log_x ** weight_power # maybe x?
#         w = x[mask] ** weight_power # maybe x?

#     elif weight_mode == "index":
#         # Weight by progression in the *filtered* dataset
#         idx = np.arange(1, len(log_x) + 1)
#         w = idx ** weight_power

#     X = sm.add_constant(log_x)
#     model = sm.WLS(fit_y, X, weights=w)
#     results = model.fit()

#     c = results.params[0]

#     n = results.params[1]
#     ic = np.exp(-c / n)
#     k = voltage_criterion / (ic**n)

#     return k, n, ic#, results


def fit_power_law_wls(x, y, voltage_criterion = None, weight_power=3,weight_mode='index'): 
    """
    Find parameters to describe non-linear IV behavior.
    Fit V/Vc = (I/Ic)^n in log-log space using weighted least squares. Centers log V with the weights for better fit stability. 
    Fit on y′ = n*x + c, for: 
        x = log I
        y′ = logV − logVc 
        c = −n log Ic

    Parameters
    ----------
    x : array-like
        Independent variable (must be > 0).
    y : array-like
        Dependent variable (must be > 0).
    voltage_criterion : float
        Voltage criterion for Ic calulation.
    weight_power : float
        Exponent used when constructing weights.
    weight_mode : str
        "x"      -> weights = x**weight_power
        "index"  -> weights = (1..N)**weight_power

    Returns
    -------
    k : float
        Pre exponent expected downstream
    Ic : float
        Critical current
    n : float
        Power-law exponent
    """
    
    mask = (x > 0) & (y > 0)
    log_x = np.log(x[mask])
    log_y = np.log(y[mask])
    log_vc = np.log(voltage_criterion)

    # Filter out invalid values
    if np.count_nonzero(mask) < 2:
        raise ValueError("Not enough valid data points for log-log fit.")
    
    fit_y = log_y - log_vc
    
    # ----------------------------------------------------------------------
    # Weight construction
    # ----------------------------------------------------------------------
    if weight_mode == "x":
        # w = log_x ** weight_power # maybe x?
        w = x[mask] ** weight_power # maybe x?

    elif weight_mode == "index":
        # Weight by progression in the *filtered* dataset
        idx = np.arange(1, len(log_x) + 1)
        w = idx ** weight_power

    # ----------------------------------------------------------------------
    # Weighted centering of log_x
    # ----------------------------------------------------------------------
    w_sum = np.sum(w)
    x_bar = np.sum(w * log_x) / w_sum     # weighted mean
    log_xc = log_x - x_bar                # centered coordinates


    X = sm.add_constant(log_xc)
    model = sm.WLS(fit_y, X, weights=w)
    results = model.fit()

    n = results.params[1]

    c = results.params[0]
    c = c - n * x_bar        # original intercept
    ic = np.exp(-c / n)

    k = voltage_criterion / (ic**n)

    cov = results.cov_params()

    # var_ct = cov[0, 0]        # Var(ĉ)
    # var_n  = cov[1, 1]        # Var(n)
    # cov_cn = cov[0, 1]        # Cov(ĉ, n)

    # # Variance of log(Ic)
    # var_log_ic = (
    #     var_ct / n**2
    #     + (c**2 / n**4) * var_n
    #     - 2 * c / n**3 * cov_cn
    # )

    # sigma_ic = ic * np.sqrt(var_log_ic)


    return k, n, ic#, results

def try_fit_power_law(x, y, voltage_criterion=None):
    """
    Try fitting power law; return (a, b) or (None, None) on failure.
    
    Parameters:
        x (np.ndarray): Current values.
        y (np.ndarray): Voltage values.
    
    Returns:
        tuple: (a, b), or (None, None) if fit fails.
    """
    return fit_power_law_wls(x,y,voltage_criterion) 
    # try:
    #     return fit_power_law_wls(x,y,voltage_criterion) 
    # except Exception:
    #     return None, None, None
    

def compute_R2_weighted(
    x, 
    y, 
    a, 
    b, 
    weight_power=5, 
    weight_mode="index"      # "x" or "index"
):
    """
    Compute weighted R² for the power-law fit y = a * x^b in log-log space.

    Parameters
    ----------
    x : array-like
        Independent variable (must be > 0).
    y : array-like
        Dependent variable (must be > 0).
    a : float
        Fit prefactor.
    b : float
        Fit exponent.
    weight_power : float
        Exponent used when constructing weights.
    weight_mode : str
        "x"      -> weights = x**weight_power
        "index"  -> weights = (1..N)**weight_power

    Returns
    -------
    float
        Weighted R² in log-log space.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    # Filter valid values
    mask = (x > 1e-12) & (y > 0)
    if np.count_nonzero(mask) < 2:
        raise ValueError("Not enough valid data points after filtering.")

    xg = x[mask]
    yg = y[mask]

    log_x = np.log(xg)
    log_y = np.log(yg)

    # ----------------------------------------------------------------------
    # Weight construction
    # ----------------------------------------------------------------------
    if weight_mode == "x":
        w = xg ** weight_power

    elif weight_mode == "index":
        # Weight by progression in the *filtered* dataset
        idx = np.arange(1, len(xg) + 1)
        w = idx ** weight_power

    else:
        raise ValueError(
            f"Invalid weight_mode '{weight_mode}'. Must be 'x' or 'index'."
        )

    # Prevent division by zero
    if np.any(w <= 0):
        raise ValueError("Non-positive weights encountered.")

    # ----------------------------------------------------------------------
    # Weighted regression quality evaluation
    # ----------------------------------------------------------------------
    y_pred = np.log(a) + b * log_x

    # Weighted residual and total variance
    y_mean_w = np.sum(w * log_y) / np.sum(w)

    ss_res = np.sum(w * (log_y - y_pred)**2)
    ss_tot = np.sum(w * (log_y - y_mean_w)**2)

    # Guard against pathological degeneracy
    if ss_tot == 0:
        return 1.0

    return 1 - ss_res / ss_tot


def lin_subtraction(x,y,cutoff,linear_sub_criterion):
    best_lin_r2 = 0
    best_p = None

    fit_check_y = y[y<cutoff]
    fit_check_x = x[y<cutoff]

    # if len(fit_check_x)>=3: # better behavior for large IVs
    #     fit_check_y = y[y<cutoff*.5]
    #     fit_check_x = x[y<cutoff*.5]

    for start in range(0,len(y)//3):
        if y[start] >= cutoff and start !=0:
            continue
        for end in range(1,len(y)):
        # for end in [len(y)-1]:
            if end - start < 2:
                continue
            else:
                x_fit = x[start:end]
                y_fit = y[start:end]
                # Step 2. Fit linear and compute R2
                try:
                    p,_, _, _, _ = np.polyfit(x_fit, y_fit, 1, full=True)
                    y_pred = np.polyval(p, fit_check_x)
                    ss_res = np.sum((fit_check_y - y_pred) ** 2)
                    ss_tot = np.sum((fit_check_y - np.mean(fit_check_y)) ** 2)
                    lin_r2 = 1 - ss_res/ss_tot# if ss_tot > 0 else -np.inf
                except:
                    lin_r2 = 0
                if lin_r2 > best_lin_r2:
                    best_lin_r2 = lin_r2
                    best_p = p

    if best_lin_r2 and best_lin_r2>linear_sub_criterion:
        # plt.clf()
        # print('\n\n\n\nSubtracted!\n\n\n\n')
        # plt.plot(np.linspace(1,len(y),len(y)),y)
        lin_fit_full_y = np.polyval(best_p, x)
        y = y - lin_fit_full_y
        # plt.plot(np.linspace(1,len(y),len(y)),lin_fit_full_y)
        # plt.plot(np.linspace(1,len(y),len(y)),y)
        # plt.gca().axhspan(0,0.01e-6)
        # plt.gca().axhspan(24.9e-6,25.01e-6)
        # plt.gca().axhspan(24.9e-6*.66,25.01e-6*.66)
        # plt.gca().grid()
        # plt.show()
    return y

def masking(x,y,noise_level):

    keep_mask = np.ones(len(y), dtype=bool)
    application_mask = np.ones(len(y), dtype=bool)


    # # mask non monotonically increasing points
    # monotonic_mask = np.concatenate(([True], np.diff(y) >= 0))
    # y = y[monotonic_mask]
    # x = x[monotonic_mask]

    monotonic_mask = np.concatenate([[True], np.diff(y) >= 0]) 

    # monotonic_mask = [True]
    # for idx in range(1, len(y)):
    #     if y[idx] >= y[idx - 1]:
    #         monotonic_mask.append(True)
    #     else:
    #         monotonic_mask.append(False)

    # y = y[monotonic_mask]
    # x = x[monotonic_mask]
    # monotonic_mask = np.ones(len(y),dtype=bool)

    # Update global mask:
    keep_mask = keep_mask & monotonic_mask
    application_mask = application_mask & monotonic_mask

    # # Apply local mask:
    # x = x[monotonic_mask]
    # y = y[monotonic_mask]

    # Suppress near-zero noise
    if noise_level is not None and noise_level > 0:
        # y = np.where(y < 0, 0.0, y)
        # # y = np.clip(y, 0, None)
        # # print(y)
        # y = np.where(np.abs(y) < noise_level, None, y)
        zero_mask = y >= 0
        keep_mask = keep_mask & zero_mask
        application_mask = application_mask & zero_mask

        # x[~zero_mask] = 0 
        # y[~zero_mask] = 0 

        noise_mask = np.abs(y) >= noise_level
        keep_mask = keep_mask & noise_mask
        application_mask = application_mask & noise_mask

        # x[~noise_mask] = 0 
        # y[~noise_mask] = 0 

    x = x[application_mask]
    y = y[application_mask] 

    return x,y,keep_mask,application_mask

def anchor_low_voltage(x, y, noise_level):
    """
    Add a synthetic low-current / low-voltage point to stabilize power-law fits.
    """
    if noise_level is None or noise_level <= 0:
        return x, y

    # Smallest positive current in the dataset
    pos_x = x[x > 0]
    if len(pos_x) == 0:
        return x, y  # nothing meaningful to anchor

    I_min = np.min(pos_x)

    # Create a synthetic anchor point
    I_anchor = 0.1 * I_min      # one order of magnitude lower
    V_anchor = 0      # baseline measurable voltage
    # V_anchor = 0.001 * noise_level      # baseline measurable voltage
    
    # # Append
    # x_aug = np.append(x, I_anchor)
    # y_aug = np.append(y, V_anchor)


    # # Doubling down?
    # I_anchor = I_min*0.001
    # V_anchor = 0


    # Append and re-sort
    x_aug = np.append(x, I_anchor)
    y_aug = np.append(y, V_anchor)
    order = np.argsort(x_aug)

    return x_aug[order], y_aug[order]


# def fit_power_law(x, y):
#     """
#     Fit y = a * x^b using linear regression in log-log space.
    
#     Parameters:
#         x (array-like): Independent variable (must be > 0). 
#         y (array-like): Dependent variable (must be > 0).
    
#     Returns:
#         a (float): Prefactor in power law.
#         b (float): Exponent.
#         r2 (float): R² of the fit in log-log space.
#     """
#     x = np.asarray(x)
#     y = np.asarray(y)
    
#     # Filter out invalid values
#     mask = (x > 1e-9) & (y > 0)
#     if np.count_nonzero(mask) < 2:
#         raise ValueError("Not enough valid data points for log-log fit.")
    
#     log_x = np.log(x[mask])
#     log_y = np.log(y[mask])

#     b, log_a = np.polyfit(log_x, log_y, 1)
#     a = np.exp(log_a)

#     return a, b
