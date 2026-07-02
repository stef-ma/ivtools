# ivtools/fitting.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
# import MultiPyVu as mpv

from . import fit_utils
from .hooks import default_hooks

warnings.simplefilter(action='ignore', category=FutureWarning)

def fit_IV_for_Ic(
        df, 
        voltage_cutoff, 
        linear_sub_criterion,
        power_law_criterion,
        min_fit_points=3, 
        max_fit_points=30, 
        noise_level=1.5e-5,
        lin_sub_level = None,
        weight_power=1,
        weight_mode='x',
        hooks = None
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
    if lin_sub_level is None:
        lin_sub_level=0.05
    
    if hooks is None:
        hooks = default_hooks  # Use global registry by default

    # 🪝 HOOK INJECTION POINT: [pre_segmentation] ← Operate on full DataFrame of V(I,B) datapoints (input: df, voltage_cutoff; output: df)
    df = hooks.execute('pre_segmentation', df, voltage_cutoff=voltage_cutoff)

    # ============================================================
    # SEGMENTATION STRATEGY
    # ============================================================
    if 'Pnum_Segment' in df.columns:
        # ACT data: segment where Pnum_Segment resets to 0
        reset_mask = (df['Pnum_Segment'] == 0)
        reset_indices = np.where(reset_mask.values)[0]
        
        if len(reset_indices) > 0:
            # Always skip first 0 (would create empty segment)
            split_at = reset_indices[1:]  # ← Simplified!
            
            segments = np.split(df.reset_index(drop=True), split_at)
            segments = [seg.reset_index(drop=True) for seg in segments if len(seg) > 0]
            
            
            print(f"[Info] Using Pnum_Segment segmentation: {len(segments)} segments")
    else:
        # TDMS data: segment by current jumps (IV pulse detection)
        segments = fit_utils.split_by_jump(df, drop_factor=0.5)  # ← RESTORE THIS!

        print(f"[Info] TDMS segmentation: {len(segments)} segments")



    fit_successes = []
    ks, bs, r2s, I_cs, I_cHs, simple_Ics = [], [], [], [], [], []
    H_avgs, dBdt_avgs = [], []
    segments_power = []
    best_starts = []
    best_ends = []
    processed_segments = []
    dlen = []
    sigmas_ic = []
    sigmas_n = []
    segment_indices = []

    for i,segment in enumerate(segments):

        I_true_original = segment['Current [A]'].values.copy()  
        V_true_original = segment['Voltage [V]'].values.copy()

        # 🪝 HOOK INJECTION POINT: [post_segmentation] ← Operate on segment DataFrame of V(I,B) datapoints (vars: *segment*, voltage_cutoff, segment_index)
        if hooks.has_hook('post_segmentation'): # passing every segment seems inefficient if not needed.
            segment = hooks.execute('post_segmentation', segment.copy(deep=True), voltage_cutoff=voltage_cutoff, segment_index=i, lin_sub_level=lin_sub_level, linear_sub_criterion=linear_sub_criterion)

        # ============================================================
        # POLARITY & SWEEP DIRECTION NORMALIZATION (NEW)
        # ============================================================
        
        I_orig = segment['Current [A]'].values
        V_orig = segment['Voltage [V]'].values
        
        # Detect overall polarity (Q1 vs Q3) using median (robust to noise)
        median_I = np.median(I_orig)
        median_V = np.median(V_orig)
        
        in_Q3 = median_I < 0
        
        # Store metadata for provenance
        segment['Negative_Bias'] = in_Q3
        
        # Flip to Q1 if in Q3 (preserves noise structure, unlike abs())
        if in_Q3:
            segment['Current [A]'] = -I_orig
            segment['Voltage [V]'] = -V_orig
        
        # Detect sweep direction (after polarity correction)
        current_values = segment['Current [A]'].values
        if len(current_values) > 1:
            # Use median of differences to avoid single-point noise
            direction = np.sign(np.median(np.diff(current_values)))
            sweep_increasing = direction >= 0
        else:
            sweep_increasing = True  # Default for single-point segments
        
        segment['Sweep_Decreasing'] = not sweep_increasing
        
        # Sort by increasing current if sweep was decreasing
        if not sweep_increasing:
            segment = segment.sort_values('Current [A]').reset_index(drop=True)
        
            
        # H_avgs.append(np.nanmean(segment['Field [T]']))
        H_avgs.append(segment['Field [T]'].iloc[-1])
        dBdt_avgs.append(np.nanmean(segment['dBdt [T/s]']))

        x = segment['Current [A]'].to_numpy()
        y = segment['Voltage [V]'].to_numpy()

        # 🪝 HOOK INJECTION POINT: [pre_linear_subtraction] → Operate on individual IV data arrays in dict form before lin_subtraction (vars: *hook_data* (dict) [x,y,segment_index,field_avg,dBdt_avg,voltage_cutoff,segment])
        if hooks.has_hook('pre_linear_subtraction'):
            ## packing
            hook_data = {
                'x': x,
                'y': y,
                'segment_index': i,
                'field_avg': H_avgs[-1],
                'dBdt_avg': dBdt_avgs[-1],
                'voltage_cutoff': voltage_cutoff,
                'segment':segment
            }
            ## executing
            hook_data = hooks.execute('pre_linear_subtraction', hook_data)
            ## unpacking
            x = hook_data['x']
            y = hook_data['y']
            voltage_cutoff = hook_data['voltage_cutoff']

        datapoints = len(x)

        # Initialize with original data (will be overwritten if processing succeeds)
        x0 = x.copy()
        y0 = y.copy()

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
            sigmas_ic.append(None)
            sigmas_n.append(None)
            newseg = segment.copy(deep=True)
            newseg['Current [A]'] = x0
            newseg['Voltage [V]'] = y0
            newseg['Processed Current [A]'] = np.full(len(x0),np.nan)
            newseg['Processed Voltage [V]'] = np.full(len(y0),np.nan)
            processed_segments.append(newseg)
            segment_indices.append(i)
            continue

        cutoff_idx = cutoff_idx_array[0]
        best_r2 = -np.inf
        best_k = best_n = best_Ic = None
        best_start = None

        
        if np.any(y > voltage_cutoff):
            # lin_sub_level = lin_sub_level if lin_sub_level is not None else voltage_cutoff
            y = fit_utils.lin_subtraction(x,y,lin_sub_level,linear_sub_criterion)
           

            # 🪝 HOOK INJECTION POINT: [post_linear_subtraction] → Operate on individual IV data arrays in dict form after lin_subtraction (vars: *hook_data* (dict) [x,y,segment_index,field_avg,dBdt_avg,voltage_cutoff,segment])
            if hooks.has_hook('post_linear_subtraction'):
                ## packing
                hook_data = {
                    'x': x,
                    'y': y,
                    'segment_index': i,
                    'field_avg': H_avgs[-1],
                    'dBdt_avg': dBdt_avgs[-1],
                    'voltage_cutoff': voltage_cutoff,
                    'segment':segment
                }
                ## executing
                hook_data = hooks.execute('post_linear_subtraction', hook_data)
                ## unpacking
                x = hook_data['x']
                y = hook_data['y']

            x0 = x.copy()
            y0 = y.copy()

            orig_indices = np.arange(len(x))
            # print(f'\n\n\n\nIndices originally:\n{orig_indices}')

            x,y,keep_mask,application_mask = fit_utils.masking(x,y,noise_level)
            orig_indices = orig_indices[application_mask]

            # y = fit_utils.lin_subtraction(x,y,lin_sub_level,linear_sub_criterion)
            # x,y,keep_mask,application_mask = fit_utils.masking(x,y,noise_level)
            # orig_indices = orig_indices[application_mask]
        

            # Add stabilizing anchor point
            x, y = fit_utils.anchor_low_voltage(x, y, noise_level)
            # print(f'Indices after masking:\n{orig_indices}')

            # anchor has no original index → use -1 or None
            orig_indices = np.append(orig_indices, -1)
            # print(f'Indices after adding -1:\n{orig_indices}')
            order = np.argsort(orig_indices)
            orig_indices = orig_indices[order]
            # print(f'Indices after sorting:\n{orig_indices}')

            # 🪝 HOOK INJECTION POINT: [post_masking_and_anchoring] → Operate on individual IV data arrays in dict form after masking and anchor_low_voltage (vars: *hook_data* (dict) [x,y,segment_index,field_avg,dBdt_avg,voltage_cutoff,segment])
            if hooks.has_hook('post_masking_and_anchoring'):
                ## packing
                hook_data = {
                    'x': x,
                    'y': y,
                    'segment_index': i,
                    'field_avg': H_avgs[-1],
                    'dBdt_avg': dBdt_avgs[-1],
                    'voltage_cutoff': voltage_cutoff,
                    'segment':segment
                }
                ## executing
                hook_data = hooks.execute('post_masking_and_anchoring', hook_data)
                ## unpacking
                x = hook_data['x']
                y = hook_data['y']   

            # Find best linear fit to dataset to compare power law fit R2 against
            try: 
                p_lin,ss_res_lin, _, _, _ = np.polyfit(x, y, 1, full=True)
                ss_tot_lin = np.sum((y - np.mean(y)) ** 2)
                lin_r2_full = 1 - ss_res_lin[0] / ss_tot_lin
            except:
                lin_r2_full = -np.inf
            if lin_r2_full>.98:
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
                sigmas_ic.append(None)
                sigmas_n.append(None)
                newseg = segment.copy(deep=True)
                newseg['Current [A]'] = x0
                newseg['Voltage [V]'] = y0
                newseg['Processed Current [A]'] = np.full(len(x0),np.nan)
                newseg['Processed Voltage [V]'] = np.full(len(y0),np.nan)
                processed_segments.append(newseg)
                segment_indices.append(i)
                continue
            # lin_r2_full = - np.inf
            for start in range(0,len(y)-1):
                for end in range(1,len(y)):
                # for end in [len(y)-1]: #USED FOR UKAEA
                    if len(x[start:end]) < min_fit_points or len(x[start:end]) > max_fit_points:
                        continue
                    else: 

                        x_fit = x[start:end]
                        y_fit = y[start:end]

                        # 🪝 HOOK INJECTION POINT: [pre_fitting] → Operate on narrow data arrays during fitting attempts in dict form right before fitting logic (vars: *hook_data* (dict) [x_fit,y_fit,start,end,x,y,segment_index,field_avg,dBdt_avg,voltage_cutoff,segment,weight_power,weight_mode])
                        if hooks.has_hook('pre_fitting'):
                            ## packing
                            hook_data = {
                                'x_fit': x_fit,
                                'y_fit': y_fit,
                                'start': start,
                                'end': end,
                                'x': x,
                                'y': y,
                                'segment_index': i,
                                'field_avg': H_avgs[-1],
                                'dBdt_avg': dBdt_avgs[-1],
                                'voltage_cutoff': voltage_cutoff,
                                'segment':segment,
                                'weight_power':weight_power,
                                'weight_mode':weight_mode
                            }
                            ## executing
                            hook_data = hooks.execute('pre_fitting', hook_data)
                            ## unpacking
                            x_fit = hook_data['x_fit']
                            y_fit = hook_data['y_fit']  
                            weight_power = hook_data['weight_power']
                            weight_mode = hook_data['weight_mode']  
                            voltage_cutoff = hook_data['voltage_cutoff']  


                        k, n, ic, sigma_ic, sigma_n = fit_utils.try_fit_power_law(x_fit, y_fit, voltage_criterion=voltage_cutoff, weight_power=weight_power,weight_mode=weight_mode)
                        r2 = fit_utils.compute_R2_weighted(x, y, k, n, weight_power,weight_mode) if k is not None and n is not None else -np.inf

                        # 🪝 HOOK INJECTION POINT: [post_fitting] → Operate on fit results (vars: *hook_data* (dict) [k,n,sigma_ic,sigma_n,r2,x_fit,y_fit,start,end,x,y,segment_index,field_avg,dBdt_avg,voltage_cutoff,segment,weight_power,weight_mode])
                        if hooks.has_hook('post_fitting'):
                            ## packing
                            hook_data = {
                                'k':k,
                                'n':n,
                                'sigma_ic':sigma_ic,
                                'sigma_n':sigma_n,
                                'r2':r2,
                                'x_fit': x_fit,
                                'y_fit': y_fit,
                                'start': start,
                                'end': end,
                                'x': x,
                                'y': y,
                                'segment_index': i,
                                'field_avg': H_avgs[-1],
                                'dBdt_avg': dBdt_avgs[-1],
                                'voltage_cutoff': voltage_cutoff,
                                'segment':segment,
                                'weight_power':weight_power,
                                'weight_mode':weight_mode
                            }
                            ## executing
                            hook_data = hooks.execute('post_fitting', hook_data)
                            ## unpacking
                            k = hook_data['k']
                            n = hook_data['n']  
                            sigma_ic = hook_data['sigma_ic']
                            sigma_n = hook_data['sigma_n']  
                            r2 = hook_data['r2']  

                        if k is not None and r2 > best_r2 and n > 1 and abs(n - 1) > lin_sub_level:
                            if r2>power_law_criterion: 
                                best_k = k
                                best_n = n
                                best_r2 = r2
                                best_Ic = fit_utils.powerlaw_inverted(voltage_cutoff,best_k,best_n)
                                test_start = start
                                test_end = end
                                best_start = orig_indices[start] if orig_indices[start]!=-1 else 0
                                best_end = orig_indices[end]
                                best_sigma_ic = sigma_ic
                                best_sigma_n = sigma_n
            
        fit_successful = best_k is not None and best_n is not None
        fit_successes.append(fit_successful)
        if fit_successful:
            # print(f'=====+++++=====++++======\n\n\nFound succesful fit. Slices in y:\n{y}\n are {test_start,test_end} corresponding to {y[test_start]} and {y[test_end]}.')
            ks.append(best_k)
            bs.append(best_n)
            r2s.append(best_r2)
            I_cs.append(best_Ic)
            I_cHs.append(best_Ic * H_avgs[-1])
            best_starts.append(best_start)
            best_ends.append(best_end)
            # simple_Ics.append(simple_Ic)
            # segments_power.append(segment.iloc[best_start:].copy())
            segments_power.append(segment.iloc[best_start:best_end][['Current [A]', 'Voltage [V]']].copy())
            dlen.append(datapoints)
            sigmas_ic.append(best_sigma_ic)
            sigmas_n.append(best_sigma_n)
            segment_indices.append(i)



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
            sigmas_ic.append(None)
            sigmas_n.append(None)
            segment_indices.append(i)


        # Save Processed IV for analysis. Replace non monotonic points with NaN
        # # Build len_adjusted arrays with NaN for masked points 
        # len_adjusted_x = []
        # len_adjusted_y = []
        # ii = 0
        # for el in keep_mask:
        #     if not el:
        #     # if el:
        #         len_adjusted_y.append(np.nan)
        #         len_adjusted_x.append(np.nan)
        #         ii +=1
        #     else:
        #         len_adjusted_y.append(y0[ii])
        #         len_adjusted_x.append(x0[ii])
        #         ii +=1
        # newseg = segment.copy(deep=True)
        # newseg['Current [A]'] = len_adjusted_x
        # newseg['Voltage [V]'] = len_adjusted_y
        # processed_segments.append(newseg)

        # Keep all lin-subtracted values (no NaN masking)
        newseg = segment.copy(deep=True)

        # Validate array lengths match segment
        if len(x0) == len(segment) and len(y0) == len(segment):
            # Processed values (normalized, lin-subtracted, monotonic)
            newseg['Processed Current [A]'] = x0  # ← Normalized for fitting
            newseg['Processed Voltage [V]'] = y0
            
            # Raw values (true sign preserved)
            newseg['Current [A]'] = I_true_original  # ← Restore originals!
            newseg['Voltage [V]'] = V_true_original
        else:
            # If mismatch, preserve original (unprocessed) values
            print(f"[Warning] Segment {i}: len mismatch (segment={len(segment)}, "
                f"x0={len(x0)}, y0={len(y0)}). Using original values.")
            newseg['Processed Current [A]'] = np.full(len(segment), np.nan)
            newseg['Processed Voltage [V]'] = np.full(len(segment), np.nan)
            newseg['Current [A]'] = I_true_original
            newseg['Voltage [V]'] = V_true_original

        processed_segments.append(newseg)

        # if fit_successful:
        #     print(f'Slices in processed segment:\n{segment["Voltage [V]"]}\n are {best_start,best_end} corresponding to {segment["Voltage [V]"].iloc[best_start]} and {segment["Voltage [V]"].iloc[best_end]}.')


    
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
        #     n = y0[i]
        #     c = len_adjusted_y[i]
        #     if np.isnan(val):
        #         d = 'NaN'
        #     else:
        #         d = y0[keep_mask][j]
        #         j+=1
        #     print(i,'|',a,'|',n,'|',c,'|',d,'|')
        # print('__________________________________')


    # 🪝 HOOK INJECTION POINT: [results] → Operate on file-level processing results (vars: *hook_data* (dict) [fit_successes,I_cs,ks,bs,r2s,segments,segments_power,processed_segments,best_Starts,best_ends,H_avgs,dBdt+avgs,IcHs,dlens,sigmas_ic,sigmas_n])
    if hooks.has_hook('results'):
        ## packing
        hook_data = {
            'fit_successes':fit_successes, 
            'I_cs':I_cs, 
            'ks':ks, 
            'bs':bs, 
            'r2s':r2s, 
            'segments':segments, 
            'segments_power':segments_power, 
            'processed_segments':processed_segments, 
            'best_starts':best_starts, 
            'best_ends':best_ends, 
            'H_avgs':H_avgs, 
            'dBdt_avgs':dBdt_avgs, 
            'I_cHs':I_cHs, 
            'dlen':dlen, 
            'sigmas_ic':sigmas_ic, 
            'sigmas_n':sigmas_n 
        }
        ## executing
        hook_data = hooks.execute('results', hook_data)
        ## unpacking
        fit_successes=hook_data['fit_successes']
        I_cs=hook_data['I_cs']
        ks=hook_data['ks']
        bs=hook_data['bs']
        r2s=hook_data['r2s']
        segments=hook_data['segments']
        segments_power=hook_data['segments_power']
        processed_segments=hook_data['processed_segments']
        best_starts=hook_data['best_starts']
        best_ends=hook_data['best_ends']
        H_avgs=hook_data['H_avgs']
        dBdt_avgs=hook_data['dBdt_avgs']
        I_cHs=hook_data['I_cHs']
        dlen=hook_data['dlen']
        sigmas_ic=hook_data['sigmas_ic']
        sigmas_n =hook_data['sigmas_n']
 


    return fit_successes, I_cs, ks, bs, r2s, segments, segments_power, processed_segments, best_starts, best_ends, H_avgs, dBdt_avgs, I_cHs, dlen, sigmas_ic, sigmas_n, segment_indices 

