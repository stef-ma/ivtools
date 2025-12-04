import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import MultiPyVu as mpv

from . import fit_utils

warnings.simplefilter(action='ignore', category=FutureWarning)

# def lin_subtraction(x,y,voltage_cutoff,linear_sub_criterion):
#     best_lin_r2 = 0
#     best_p = None
#     for start in range(0,len(y)//3):
#         if y[start] >= voltage_cutoff and start !=0:
#             continue
#         for end in range(1,len(y)):
#             if end - start < 3:
#                 continue
#             else:
#                 x_fit = x[start:end]
#                 y_fit = y[start:end]
#                 # Step 2. Fit linear and compute R2
#                 try:
#                     p,ss_res, _, _, _ = np.polyfit(x_fit, y_fit, 1, full=True)
#                     ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
#                     lin_r2 = 1 - ss_res[0] / ss_tot
#                 except:
#                     lin_r2 = 0
#                 if lin_r2 > best_lin_r2:
#                     best_lin_r2 = lin_r2
#                     best_p = p

#     if best_lin_r2 and best_lin_r2>linear_sub_criterion:
#         lin_fit_full_y = np.polyval(best_p, x)
#         y = y - lin_fit_full_y
#     return y
def lin_subtraction(x,y,voltage_cutoff,linear_sub_criterion):
    best_lin_r2 = 0
    best_p = None

    fit_check_y = y[y<voltage_cutoff*.5]
    fit_check_x = x[y<voltage_cutoff*.5]

    for start in range(0,len(y)//3):
        if y[start] >= voltage_cutoff and start !=0:
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
    # keep_mask = keep_mask & monotonic_mask
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
        # keep_mask = keep_mask & zero_mask
        application_mask = application_mask & zero_mask

        # x[~zero_mask] = 0 
        # y[~zero_mask] = 0 

        noise_mask = np.abs(y) >= noise_level
        # keep_mask = keep_mask & noise_mask
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


def fit_IV_for_Ic(
        df, 
        voltage_cutoff, 
        linear_sub_criterion,
        power_law_criterion,
        min_fit_points=3, 
        max_fit_points=5, 
        noise_level=1.5e-5
        ):
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
            # simple_Ics.append(None)
            best_starts.append(None)
            best_ends.append(None)
            segments_power.append(pd.DataFrame(columns=['Current [A]', 'Voltage [V]']))
            dlen.append(datapoints)

            continue

        cutoff_idx = cutoff_idx_array[0]
        best_r2 = -np.inf
        best_k = best_b = best_Ic = None
        best_start = None

        
        if np.any(y > voltage_cutoff):
            y = lin_subtraction(x,y,voltage_cutoff,linear_sub_criterion)
           
            x0 = x.copy()
            y0 = y.copy()

            orig_indices = np.arange(len(x))
            # print(f'\n\n\n\nIndices originally:\n{orig_indices}')

            x,y,keep_mask,application_mask = masking(x,y,noise_level)

            orig_indices = orig_indices[application_mask]
        

            # Add stabilizing anchor point
            x, y = anchor_low_voltage(x, y, noise_level)
            # print(f'Indices after masking:\n{orig_indices}')

            # anchor has no original index → use -1 or None
            orig_indices = np.append(orig_indices, -1)
            # print(f'Indices after adding -1:\n{orig_indices}')
            order = np.argsort(orig_indices)
            orig_indices = orig_indices[order]
            # print(f'Indices after sorting:\n{orig_indices}')

        # if np.any(y > voltage_cutoff):

            # Find best linear fit to dataset to compare power law fit R2 against
            try: 
                p_lin,ss_res_lin, _, _, _ = np.polyfit(x, y, 1, full=True)
                ss_tot_lin = np.sum((y - np.mean(y)) ** 2)
                lin_r2_full = 1 - ss_res_lin[0] / ss_tot_lin
            except:
                lin_r2_full = -np.inf
            # if lin_r2_full>.95:
            #     continue
            for start in range(0,len(y)-1):
                # for end in range(1,len(y)):
                for end in [len(y)-1]:
                    if len(x[start:end]) < min_fit_points or len(x[start:end]) > max_fit_points:
                        continue
                    else:
                        x_fit = x[start:end]
                        y_fit = y[start:end]

                        k, b = fit_utils.try_fit_power_law(x_fit, y_fit)
                        # r2 = fit_utils.compute_R2(x, y, k, b) if k is not None and b is not None else -np.inf
                        r2 = fit_utils.compute_R2_weighted(x, y, k, b) if k is not None and b is not None else -np.inf
                        if k is not None and r2 > best_r2 and b > 0 and r2>lin_r2_full:
                            # FINDME 2
                            if r2>power_law_criterion: # use 99 with noise supression
                            # if r2>0.5: # 
                                best_k = k
                                best_b = b
                                best_r2 = r2
                                best_Ic = (voltage_cutoff / k) ** (1 / b)
                                test_start = start
                                test_end = end
                                best_start = orig_indices[start] if orig_indices[start]!=-1 else 0
                                best_end = orig_indices[end]
            # print (f'Best power law fit found for segment {i} in file {segment["File"].unique()[0]}: R² = {best_r2}.')
        fit_successful = best_k is not None and best_b is not None
        fit_successes.append(fit_successful)
        if fit_successful:
            print(f'=====+++++=====++++======\n\n\nFound succesful fit. Slices in y:\n{y}\n are {test_start,test_end} corresponding to {y[test_start]} and {y[test_end]}.')
            ks.append(best_k)
            bs.append(best_b)
            r2s.append(best_r2)
            I_cs.append(best_Ic)
            I_cHs.append(best_Ic * H_avgs[-1])
            best_starts.append(best_start)
            best_ends.append(best_end)
            # simple_Ics.append(simple_Ic)
            # segments_power.append(segment.iloc[best_start:].copy())
            segments_power.append(segment.iloc[best_start:best_end][['Current [A]', 'Voltage [V]']].copy())
            dlen.append(datapoints)



        else:
            ks.append(None)
            bs.append(None)
            r2s.append(None)
            I_cs.append(None)
            I_cHs.append(None)
            best_starts.append(None)
            best_ends.append(None)
            # simple_Ics.append(None)
            segments_power.append(pd.DataFrame(columns=['Current [A]', 'Voltage [V]']))
            dlen.append(datapoints)


        # Save Processed IV for analysis. Replace non monotonic points with NaN
        len_adjusted_x = []
        len_adjusted_y = []
        ii = 0
        # if len(y0)!=len(y):
        #     print('alarm')
        # # print('')
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
        # len_adjusted_x, len_adjusted_y = anchor_low_voltage(len_adjusted_x, len_adjusted_y, noise_level) 
        segment['Current [A]'] = len_adjusted_x
        segment['Voltage [V]'] = len_adjusted_y
        processed_segments.append(segment)
        if fit_successful:
            print(f'Slices in processed segment:\n{segment["Voltage [V]"]}\n are {best_start,best_end} corresponding to {segment["Voltage [V]"].iloc[best_start]} and {segment["Voltage [V]"].iloc[best_end]}.')


    
    # processed_segments
    # print('Type of segments returned:', type(segments))
    # print('Len of segments returned:', len(segments))
    # print('Type of processed_segments returned:', type(processed_segments))
    # print('Len of processed_segments returned:', len(processed_segments))
        # print('__________________________________')
        # print('i | mask | y0 | len_adj_y | final_y |')
        # j = 0
        # for i, val in enumerate(len_adjusted_y):
        #     a = keep_mask[i]
        #     b = y0[i]
        #     c = len_adjusted_y[i]
        #     if np.isnan(val):
        #         d = 'NaN'
        #     else:
        #         d = y0[keep_mask][j]
        #         j+=1
        #     print(i,'|',a,'|',b,'|',c,'|',d,'|')
        # print('__________________________________')


    return fit_successes, I_cs, ks, bs, r2s, segments, segments_power, processed_segments, best_starts,best_ends, H_avgs, dBdt_avgs, I_cHs, dlen

