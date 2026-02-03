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

        elif F[i] != F[i - 1]:
            split_idx.append(i)
            running_max = I[i]
            running_min = I[i]

    split_idx = np.unique(split_idx)
    segments = np.split(df.reset_index(drop=True), split_idx)
    # return segments#, split_idx
    return [seg.copy(deep=True) for seg in segments] #safer



def find_ROI(iv_file, index, center_fraction, flat_threshold=0.0005):
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




def process_IV_pulse(iv_file,top,left,right,center_fraction):
    '''
    Extracts the voltage, current, field and temperature for a single IV pulse iv_file class.
    '''
    lROI,hROI = find_ROI(iv_file,top, center_fraction)

    error_std = iv_file.noise_std
    error_rms = iv_file.noise_rms

    I = np.mean(iv_file.I[lROI:hROI])
    B = np.mean(iv_file.B[lROI:hROI])
    Bdiff = np.mean(np.diff(iv_file.B[lROI:hROI]))
    tdiff = np.mean(np.diff(iv_file.t[lROI:hROI]))
    dBdt = Bdiff/tdiff
    # dBdt = np.mean(np.diff(np.array([datafile.t[lROI:hROI],datafile.B[lROI:hROI]])))
    
    T = iv_io.extract_numeric_temperature(iv_file.T)
    
    left_lROI,left_hROI = find_ROI(iv_file,left, center_fraction)
    right_lROI, right_hROI = find_ROI(iv_file,right, center_fraction)


    V, processed_V = safe_mean(iv_file.V[lROI:hROI])
    leftV, _ = safe_mean(iv_file.V[left_lROI:left_hROI])
    rightV, _ = safe_mean(iv_file.V[right_lROI:right_hROI])
    V = V - (leftV+rightV)/2 # 1st option

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


def fit_power_law_wls(
        x, 
        y, 
        voltage_criterion = None, 
        weight_power=3,
        weight_mode='index'
        ): 
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
        # w = idx ** weight_power
        w = idx ** weight_power
        w = w / np.max(w)   # normalize


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


    M = X.T @ np.diag(w) @ X
    cond = np.linalg.cond(M)
    if cond > 1e10 or abs(n) < 0.5:
        sigma_n = np.nan
        sigma_ic = np.nan
        return k, n, ic, sigma_ic, sigma_n

    # cov = results.cov_params()

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
    # sigma_n = np.sqrt(cov[1, 1])

    # print(w)

    # sigma_k, sigma_n = compute_uncertainties_nonlinear(
    #     x[mask], 
    #     y[mask], 
    #     k, 
    #     n
    #     )   
    
    # var_log_ic = ((sigma_k / k)**2 / n**2 + (np.log(ic) * sigma_n / n)**2)

    # sigma_ic = ic * np.sqrt(var_log_ic)

    cov = results.cov_params()

    sigma_n = np.sqrt(cov[1,1])

    # Jacobian-based propagation for log(Ic)
    J = np.array([-1/n, c/n**2])
    var_log_ic = J @ cov @ J

    if var_log_ic > 0:
        sigma_ic = ic * np.sqrt(var_log_ic)
    else:
        sigma_ic = np.nan

    return k, n, ic, sigma_ic, sigma_n#, results

# from scipy.optimize import curve_fit

# def compute_uncertainties_nonlinear(x, y, k, n):
#     """
#     Compute robust uncertainties for (k, n) using nonlinear least squares.
#     Uses current WLS estimates as initial guesses.

#     Parameters
#     ----------
#     x, y : arrays
#         Original data (not log transformed)
#     k, n : floats
#         Best-fit parameters from log-log WLS
#     w : array or None
#         Optional weights (same as used in WLS)

#     Returns
#     -------
#     sigma_k, sigma_n
#     """

#     try:
#         popt, pcov = curve_fit(
#             powerlaw,
#             x, y,
#             p0=(k, n),
#             maxfev=20000
#         )


#         sigma_k, sigma_n = np.sqrt(np.diag(pcov))

#     except Exception:
#         sigma_k, sigma_n = np.nan, np.nan

#     return sigma_k, sigma_n


def try_fit_power_law(x, y, voltage_criterion=None):
    """
    Try fitting power law; return (a, b) or (None, None) on failure.
    
    Parameters:
        x (np.ndarray): Current values.
        y (np.ndarray): Voltage values.
    
    Returns:
        tuple: (a, b), or (None, None) if fit fails.
    """
    # return fit_power_law_wls(x,y,voltage_criterion) 
    try:
        return fit_power_law_wls(x,y,voltage_criterion) 
    except Exception:
        return None, None, None, None, None
    

def compute_R2_weighted(
    x, 
    y, 
    a, 
    b, 
    weight_power=3, 
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

    # # Guard against pathological degeneracy
    # if ss_tot == 0:
    #     return 1.0
    # print(w,'\n')
    return 1 - ss_res / ss_tot


def lin_subtraction(x, y, cutoff=0.15, linear_sub_criterion=0.75):
    """
    Identify and subtract a linear background using log–log slope deviation.

    Parameters
    ----------
    x : array-like
        Current
    y : array-like
        Voltage
    cutoff : unused (kept for drop-in compatibility)
    linear_sub_criterion : float
        Allowed deviation of log–log slope from 1 (delta)

    Returns
    -------
    y_corr : ndarray
        Background-subtracted voltage
    """

    x = np.asarray(x)
    y = np.asarray(y)

    # --- basic sanity masking ---
    mask = (
    (x != 0) &
    (y != 0) &
    np.isfinite(x) &
    np.isfinite(y)
    )
    x0 = x[mask]
    y0 = y[mask]

    if len(x0) < 2:
        return y  # not enough data to do anything safely
    

    # --- log–log slope ---
    logx = np.log(np.abs(x0))
    logy = np.log(np.abs(y0))

    # # smooth to suppress numerical noise
    # win = min(len(logx) // 2 * 2 - 1, 11)
    # if win >= 5:
    #     logy_s = savgol_filter(logy, win, 2)
    # else:
    #     logy_s = logy

    N = len(logx)

    # conservative smoothing for short IVs
    if N >= 6:
        win = 3
    else:
        win = None

    if win is not None:
        logy_s = savgol_filter(logy, win, 2)
    else:
        logy_s = logy


    slope = np.gradient(logy_s, logx)


    # --- identify linear regime (slope ~ 1) ---
    # good = (slope > 0) & (slope < 1 + cutoff)
    good = np.abs(slope - 1) < cutoff

    # require contiguity starting from lowest |I|
    idx = np.argsort(np.abs(x0))

    # Improved 2
    max_bad = 3
    bad = 0
    last_good_idx = None

    for i, g in enumerate(good):
        if g:
            last_good_idx = i
            bad = 0
        else:
            bad += 1
            if bad > max_bad:
                break

    if last_good_idx:
        lin_idx = idx[:last_good_idx]
    else:
        return y
    

    x_lin = x0[lin_idx]
    y_lin = y0[lin_idx]

    # print('\n\n\n+++++++++++++++++++++++++++++\n\n\nCorrection:\n',y)
    # --- fit linear background ---
    try:
        p = np.polyfit(x_lin, y_lin, 1)
    except Exception:
        return y
        
    if p[0] <= 0:
        return y


    # --- subtract globally ---
    y_corr = y - np.polyval(p, x)

    y_pred = np.polyval(p, x_lin)

    ss_res = np.sum((y_lin - y_pred)**2)
    ss_tot = np.sum((y_lin - np.mean(y_lin))**2)

    # guard against degenerate case
    if ss_tot == 0:
        return y

    r2 = 1 - ss_res / ss_tot

    if r2 < linear_sub_criterion:
        return y


    print('\n\n\n+++++++++++++++++++++++++++++\n\n\nCorrection:\n',y,'\n',y_corr,'\n',p)

    return y_corr


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
    I_anchor = 1e-6 * I_min      # six order of magnitude lower
    V_anchor = 0      # baseline measurable voltage
    # V_anchor = noise_level*1e-6      # baseline measurable voltage
    # V_anchor = 0.001 * noise_level      # baseline measurable voltage
    

    # Append and re-sort
    x_aug = np.append(x, I_anchor)
    y_aug = np.append(y, V_anchor)
    order = np.argsort(x_aug)

    return x_aug[order], y_aug[order]
