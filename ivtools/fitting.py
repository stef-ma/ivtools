import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import MultiPyVu as mpv

from . import fit_utils

warnings.simplefilter(action='ignore', category=FutureWarning)

def lin_subtraction(x,y):
    # Step 1. Take some points.
    best_lin_r2 = 0
    # best_start = None
    # best_end = None
    best_p = None
    for start in range(0,len(y)//3):
    # for start in [0]:
        # if y[start] >= voltage_cutoff and start !=0:
        #     continue
        for end in range(1,len(y)):
            if end - start < 3:
                continue
            else:
                x_fit = x[start:end]
                y_fit = y[start:end]
                # Step 2. Fit linear and compute R2
                try:
                    p,ss_res, _, _, _ = np.polyfit(x_fit, y_fit, 1, full=True)
                    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                    lin_r2 = 1 - ss_res[0] / ss_tot
                except:
                    lin_r2 = 0
                if lin_r2 > best_lin_r2:
                    best_lin_r2 = lin_r2
                    # best_start = start
                    # best_end = end
                    best_p = p
    # Step 3. If the linear fit is good, subtract it from all y values
    # FINDME 1
    # if best_lin_r2 > .75:
    # if best_lin_r2 > .95:
    # if best_lin_r2 > .95:
    # if best_lin_r2 > .6:
    if best_lin_r2:
        # print(f'Best linear fit found for segment {i} in file {segment["File"].unique()[0]}: R² = {best_lin_r2}.')
        lin_fit_full_y = np.polyval(best_p, x)
        # if plotit:
        #     ax.plot(x[best_start:best_end],y[best_start:best_end],marker='D',linestyle='-.',markersize=3,label='Fit Area')
        y = y - lin_fit_full_y
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


# def masking(x, y, noise_level):
#     x = np.asarray(x)
#     y = np.asarray(y)

#     N = len(y)

#     # --- Start with full masks ---
#     keep_mask = np.ones(N, dtype=bool)

#     # --- 1. Monotonicity mask (y increasing) ---
#     monotonic_mask = np.concatenate([[True], np.diff(y) >= 0])
#     keep_mask &= monotonic_mask

#     # --- 2. Noise mask ---
#     if noise_level is not None and noise_level > 0:
#         noise_mask = np.abs(y) >= noise_level
#         keep_mask &= noise_mask

#     # --- Final masked data ---
#     x_out = x[keep_mask]
#     y_out = y[keep_mask]

#     # application_mask = the same as keep_mask (you only need one)
#     application_mask = keep_mask.copy()

#     return x_out, y_out, keep_mask, application_mask


def fit_IV_for_Ic(df, voltage_cutoff, min_fit_points=3, max_fit_points=5, noise_level=1.5e-5):
    """
    Analyze I–V data to extract segments, perform power-law fitting,
    and estimate the critical current (I_c) for each segment.

    For each segment, use points ending after the cutoff, but include earlier points
    to find the best-fit window based on R².

    Returns:
        tuple:
            - fit_successes (list of bool)
            - I_cs (list of float or None)
            - ks (list of float or None)
            - bs (list of float or None)
            - segments (list of pd.DataFrame)
            - segments_power (list of pd.DataFrame): actual fitted data range
            - H_avgs (list of float)
            - dBdt_avgs (list of float)
            - I_cHs (list of float or None)
    """
    segments = fit_utils.split_by_jump(df)
    fit_successes = []
    ks, bs, r2s, I_cs, I_cHs, simple_Ics = [], [], [], [], [], []
    H_avgs, dBdt_avgs = [], []
    segments_power = []
    best_starts = []
    best_ends = []
    processed_segments = []
    dlen = []

    for i,segment in enumerate(segments):
        H_avgs.append(np.nanmean(segment['Field [T]']))
        dBdt_avgs.append(np.nanmean(segment['dBdt [T/s]']))

        x = segment['Current [A]'].to_numpy()
        y = segment['Voltage [V]'].to_numpy()

        datapoints = len(x)


        cutoff_idx_array = np.where(y > voltage_cutoff)[0]
        if cutoff_idx_array.size == 0:
            fit_successes.append(False)
            ks.append(None)
            bs.append(None)
            r2s.append(None)
            simple_Ics.append(None)
            I_cs.append(None)
            I_cHs.append(None)
            simple_Ics.append(None)
            best_starts.append(None)
            best_ends.append(None)
            segments_power.append(pd.DataFrame(columns=['Current [A]', 'Voltage [V]']))
            dlen.append(datapoints)

            continue

        cutoff_idx = cutoff_idx_array[0]
        best_r2 = -np.inf
        best_k = best_b = best_Ic = None
        best_start = None



        # Simple Ic calculation without fitting
        if cutoff_idx != 0: 
            simple_Ic = np.mean(x[cutoff_idx-1:cutoff_idx])
        else:
            simple_Ic = None

        files_to_inspect = [
            # 'p134856_102225',
            # 'p153647_102225', #H||c 20 K
            # 'p173228_102225', #H||c 20 K second run
            # 'p191307_102225' #H||c 20 K third run
            # 'p172846_102325', # H||45 20K, has outliers
            # 'p145009_102325', # H||45 30K has outliers
            # 'p181832_102425'
            # 'p014_153958_103025',
            # 'p027_185850_103025'
            # 'p111616_102325'
            # 'p161155_102325'
            
        ]

        if df['File'].unique()[0] in files_to_inspect:
            plotit=True
        else:
            plotit=False

        if np.any(y > voltage_cutoff):
            y = lin_subtraction(x,y)

                
        x0 = x.copy()
        y0 = y.copy()


        x,y,keep_mask,application_mask = masking(x,y,noise_level)





        if np.any(y > voltage_cutoff):

            # Find best linear fit to dataset to compare power law fit R2 against
            try: 
                p_lin,ss_res_lin, _, _, _ = np.polyfit(x, y, 1, full=True)
                ss_tot_lin = np.sum((y - np.mean(y)) ** 2)
                lin_r2_full = 1 - ss_res_lin[0] / ss_tot_lin
            except:
                lin_r2_full = -np.inf

            # if lin_r2_full> 0.6:
            #     continue # if the full linear fit is too good, skip power law fitting

            print(lin_r2_full)

            # for start in range(0, cutoff_idx + 1):  # include data before cutoff
            #     for end in range(cutoff_idx,cutoff_idx+max_fit_points): # include some points after cutoff
            for start in range(0,len(y)-1):
                for end in range(1,len(y)):
                    # if end - start < 3:
                    #     continue
                    # if y[start] >= voltage_cutoff and start !=0:
                    #     continue
                    if len(x[start:end]) < min_fit_points or len(x[start:end]) > max_fit_points:
                        continue
                    # if np.all(y[start:end] < voltage_cutoff):
                    # if y[start:end].all()<voltage_cutoff: # My logic when I wrote this was bad, but it actually works as a noise guard. It rejects all fits that include values below the ec.
                        # continue    
                    else:
                        x_fit = x[start:end]
                        y_fit = y[start:end]


                        k, b = fit_utils.try_fit_power_law(x_fit, y_fit)
                        r2 = fit_utils.compute_R2(x, y, k, b) if k is not None and b is not None else -np.inf
                        if k is not None and r2 > best_r2 and b > 0 and r2>lin_r2_full:
                            # FINDME 2
                            if r2>0.95: # use 99 with noise supression
                            # if r2>0.5: # 
                                best_k = k
                                best_b = b
                                best_r2 = r2
                                best_Ic = (voltage_cutoff / k) ** (1 / b)
                                best_start = start
                                best_end = end
            # print (f'Best power law fit found for segment {i} in file {segment["File"].unique()[0]}: R² = {best_r2}.')
        fit_successful = best_k is not None and best_b is not None
        fit_successes.append(fit_successful)
        if fit_successful:
            # print(f'Found succesful fit with r2 for power law: {best_r2} against full linear: {lin_r2_full}')
            ks.append(best_k)
            bs.append(best_b)
            r2s.append(best_r2)
            I_cs.append(best_Ic)
            I_cHs.append(best_Ic * H_avgs[-1])
            best_starts.append(best_start)
            best_ends.append(best_end)
            simple_Ics.append(simple_Ic)
            # segments_power.append(segment.iloc[best_start:].copy())
            segments_power.append(segment.iloc[best_start:best_end][['Current [A]', 'Voltage [V]']].copy())
            dlen.append(datapoints)

            if plotit:

                x_fit = np.linspace(np.min(x), np.max(x), 100)
                y_fit = best_k * x_fit ** best_b
                ax.plot(x_fit, y_fit, linestyle='--', color='salmon', label=f'Power-law fit\n(R²={best_r2:.3f}, n={best_b:.3f}, k={best_k:.3e}, I$_c$={best_Ic*1e3:.2f} mA)')




        else:
            ks.append(None)
            bs.append(None)
            r2s.append(None)
            I_cs.append(None)
            I_cHs.append(None)
            best_starts.append(None)
            best_ends.append(None)
            simple_Ics.append(None)
            segments_power.append(pd.DataFrame(columns=['Current [A]', 'Voltage [V]']))
            dlen.append(datapoints)


        # Save Processed IV for analysis. Replace non monotonic points with NaN
        len_adjusted_x = []
        len_adjusted_y = []
        ii = 0
        for el in keep_mask:
            if not el:
            # if el:
                len_adjusted_y.append(np.nan)
                len_adjusted_x.append(np.nan)
                ii +=1
            else:
                len_adjusted_y.append(y0[ii])
                len_adjusted_x.append(x0[ii])
                ii +=1
        # len_adjusted_x = x0[keep_mask]
        # len_adjusted_y = y0[keep_mask]
        segment['Current [A]'] = len_adjusted_x
        segment['Voltage [V]'] = len_adjusted_y
        processed_segments.append(segment)
    
    # processed_segments
    # print('Type of segments returned:', type(segments))
    # print('Len of segments returned:', len(segments))
    # print('Type of processed_segments returned:', type(processed_segments))
    # print('Len of processed_segments returned:', len(processed_segments))
        print('__________________________________')
        print('i | mask | y0 | len_adj_y | final_y |')
        j = 0
        for i, val in enumerate(len_adjusted_y):
            a = keep_mask[i]
            b = y0[i]
            c = len_adjusted_y[i]
            if np.isnan(val):
                d = 'NaN'
            else:
                d = y0[keep_mask][j]
                j+=1
            print(i,'|',a,'|',b,'|',c,'|',d,'|')
        print('__________________________________')


    return fit_successes, I_cs, ks, bs, r2s, segments, segments_power, processed_segments, best_starts,best_ends, H_avgs, dBdt_avgs, I_cHs, simple_Ics, dlen

