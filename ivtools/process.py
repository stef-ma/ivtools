# ivtools/process.py
import time
import re
import os
import io
import warnings
import contextlib

import numpy as np
import pandas as pd


from . import iv_io
from . import fit_utils
from . import fitting
from .hooks import default_hooks

@contextlib.contextmanager
def suppress_print(verbose=True):
    """
    Context manager to optionally suppress stdout prints.
    If verbose=False, prints are captured and discarded.
    If verbose=True, prints show as normal.
    """
    if verbose:
        # No suppression, just run normally
        yield
    else:
        # Capture and suppress prints
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            yield
        # At this point, buffer.getvalue() contains suppressed output if needed


def process_ivf(
    ivf,
    fp,
    sample,
    temperature,
    angle,
    tfield,
    voltage_cutoff,
    noise_level,
    linear_sub_criterion,
    power_law_criterion,
    minfp,
    maxfp,
    magnet='Mid Pulse',
    verbose = False,
    lin_sub_level = None,
    center_fraction=0.5,
    weight_power=1,
    weight_mode='x',
    hooks = None,
    orientation_tolerance=1.0, # for act IVs
    temperature_tolerance=0.5 # for act IVs
    ):
    
    fname = os.path.basename(fp)
    
    if verbose:
        start = time.perf_counter()

    if hooks is None:
        hooks = default_hooks

    from .ppms_io import ACT_File
    is_act_file = isinstance(ivf, ACT_File)
    
    if is_act_file and not ivf.passed:
        raise ValueError(f"ACT_File failed validation: {fp}")


    # 🪝 HOOK INJECTION POINT: [ivf_processing] ← Operate on full tdms datastream in the form of an IV_File instance (gains are accounted for and current is converted) (vars: *ivf*, fp, sample, temperature, angle, tfield, magnet)
    ivf = hooks.execute(
        'ivf_processing', 
        ivf,
        fp=fp,
        sample=sample,
        temperature=temperature,
        angle=angle,
        tfield=tfield,
        magnet=magnet,
        )
    
    # ============================================================
    # ACT_FILE: SPLIT BY CONDITIONS
    # ============================================================
    if is_act_file:
        condition_groups = ivf.split_act_by_conditions(
            orientation_tol=orientation_tolerance,
            temperature_tol=temperature_tolerance,
            verbose=verbose
        )
        
        all_results = []
        
        for cond_idx, condition in enumerate(condition_groups):
            if verbose:
                print(f"\n[Processing] Condition {cond_idx+1}/{len(condition_groups)}: "
                      f"T={condition['temperature']:.2f} K, "
                      f"θ={condition['orientation']:.1f}°")
            
            # Create filtered DataFrame for this condition
            mask = condition['mask']
            
            # Use median values for metadata (override user inputs if provided)
            effective_temperature = np.round(condition['temperature'],2)
            effective_angle = np.round(condition['orientation'],2)
            effective_tfield = np.round(condition['field'],4)
            effective_sample = ivf.sample
            iv_starts = np.where(ivf.Pnum_segment[mask]==0)[0]
            n_ivs = len(iv_starts)
            
            # Create IV_Index for each data point
            IV_Index = np.zeros(np.sum(mask), dtype=int)
            for iv_num in range(n_ivs):
                start = iv_starts[iv_num]
                end = iv_starts[iv_num + 1] if iv_num < n_ivs - 1 else len(IV_Index)
                IV_Index[start:end] = iv_num

            # Cast DataFrame
            df_cond = pd.DataFrame({
                'Current [A]': ivf.I[mask],
                'Voltage [V]': ivf.V[mask],
                'Field [T]': np.round(ivf.B[mask], 4),
                'dBdt [T/s]': 0,
                'noise_std': ivf.V_noise[mask],
                'noise_rms': ivf.V_noise[mask],
                'Temperature [K]': effective_temperature,
                'Orientation':effective_angle,
                'Target Field [T]': effective_tfield,
                'Pnum_Segment': ivf.Pnum_segment[mask],
                'File': fname,
                'Time [s]': ivf.t[mask],
                'lROI' : np.nan,
                'rROI' : np.nan,
                'Denoised Voltage Array [V]': np.nan,
                'Vavg': ivf.V[mask],
                'Magnet': 'PPMS ACT IV',
                'Sample':sample,
                'IV_Index':IV_Index
            })

            # ============================================================
            # FITTING (using Pnum_Segment for segmentation)
            # ============================================================
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                with suppress_print(verbose):
                    fit_results = fitting.fit_IV_for_Ic(
                        df_cond, 
                        voltage_cutoff, 
                        linear_sub_criterion,
                        power_law_criterion,
                        min_fit_points=minfp,
                        max_fit_points=maxfp,
                        noise_level=noise_level,
                        lin_sub_level=lin_sub_level if lin_sub_level is not None else voltage_cutoff,
                        weight_power=weight_power,
                        weight_mode=weight_mode,
                        hooks=hooks
                    )
                    
                    (fit_successes, I_cs, ks, bs, r2s, segments, 
                    segments_power, processed_segments, best_starts, best_ends, 
                    H_avgs, dBdt_avgs, I_cHs, dlen, sigmas_ic, sigmas_n,
                    segment_indices) = fit_results
                    
                    # Add processed columns
                    if len(processed_segments) > 0:
                        processed_df = pd.concat(processed_segments, ignore_index=True)
                        
                        if len(processed_df) == len(df_cond):
                            # Overwrite with RAW values from fitting (already correct!)
                            df_cond['Current [A]'] = processed_df['Current [A]'].values  # ← True originals
                            df_cond['Voltage [V]'] = processed_df['Voltage [V]'].values
                            
                            # Add processed columns (normalized)
                            df_cond['Processed Current [A]'] = processed_df['Processed Current [A]'].values
                            df_cond['Processed Voltage [V]'] = processed_df['Processed Voltage [V]'].values
                        else:
                            print(f"[Warning] Processed data length mismatch for condition {cond_idx}")
                            df_cond['Processed Current [A]'] = np.full(len(df_cond), np.nan)
                            df_cond['Processed Voltage [V]'] = np.full(len(df_cond), np.nan)
                
                if w and verbose:
                    print(f"[Warning] Fit warnings in condition {cond_idx}")

            ivs = df_cond

            # ============================================================
            # BUILD FIT RESULTS DATAFRAME
            # ============================================================
            running_iv_dicts = []
            
            for k, fit_success in enumerate(fit_successes):
                segment_id = segment_indices[k]
                
                if fit_success:
                    result_summary = {
                        'File': fname,
                        'Target Field [T]': effective_tfield,
                        'Temperature [K]': effective_temperature,
                        'IV_Index': segment_id,
                        'fit_start_index': best_starts[k],
                        'fit_end_index': best_ends[k],
                        'I_c': I_cs[k],
                        'I_cH': I_cHs[k],
                        'k': ks[k],
                        'n': bs[k],
                        'R²': r2s[k],
                        'Avg Field [T]': H_avgs[k],
                        'Avg dB/dt [T/s]': dBdt_avgs[k],
                        'Sample': effective_sample,
                        'Orientation': effective_angle,
                        'Magnet': 'PPMS ACT',
                        'Data_Length': dlen[k],
                        'I_c Error': sigmas_ic[k],
                        'n Error': sigmas_n[k]
                    }
                    running_iv_dicts.append(result_summary)

            fits = pd.DataFrame(running_iv_dicts)
            
            # Store results for this condition
            all_results.append((ivs, fits, condition))
        
        if verbose:
            end = time.perf_counter()
            print(f"\n[Complete] Processed {len(condition_groups)} conditions "
                  f"from {fname} ({end - start:.2f} s)")
        
        return all_results
    
    else:   
        highs = ivf.tops
        lows = ivf.troths
        iv = []
        # ivs = []

        fits = []

        tail = re.split('_', os.path.basename(fp))[-1].replace('.tdms', '')
        # sample = samples.get(tail, 'unknown')
        orientation = angle if angle is not None else 'UnknownAngle'
        # magnet = magnets.get(tail, 'unknown')

        if sample == 'unknown' and verbose:
            print(f"[Warning] Sample key '{tail}' not found in samples dict.")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            last_current = 0
            IV_pulse_iteration = 0
            for j, top in enumerate(highs):
                with suppress_print(verbose):
                    result, *_ = fit_utils.process_IV_pulse(ivf, top, lows[2 * j], lows[2 * j + 1],center_fraction=center_fraction,hooks=hooks)
                # IV Pulse iteration
                I = result['Current [A]']
                if I>=last_current:
                    # print(last_current,I)
                    last_current = I
                else:
                    IV_pulse_iteration+=1
                    last_current = I
                # print(IV_pulse_iteration)
                result.update({
                    'IV_Index':IV_pulse_iteration,
                    # 'Turns': turn_count,  
                    'File': fname,
                    'Target Field [T]': tfield,
                    'Sample': sample,
                    'Orientation': orientation,
                    'Magnet': magnet,
                    'Vavg [V]': ivf.Vavg[j],
                    'Time [s]': ivf.t[top]
                })
                # result['Field [T]'] = result['Field [T]']*1.05 if magnet!='PPMS' else result['Field [T]']  # Field Correction
                iv.append(result)
                # ivs.append(result)

            df = pd.DataFrame(iv)

        if w and verbose:
            print(f"[Warning] Math warnings detected in processing {os.path.basename(fp)}.")

        # Suppress and flag any warnings raised by fit_IV_for_Ic
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with suppress_print(verbose):
                fit_results = fitting.fit_IV_for_Ic(
                    df, 
                    voltage_cutoff, 
                    linear_sub_criterion,
                    power_law_criterion,
                    min_fit_points=minfp,
                    max_fit_points=maxfp,
                    noise_level = noise_level,
                    lin_sub_level = lin_sub_level if lin_sub_level is not None else voltage_cutoff,
                    weight_power=weight_power,
                    weight_mode=weight_mode,
                    hooks=hooks
                    )
                (fit_successes, I_cs, ks, bs, r2s, segments, segments_power,
                 processed_segments, best_starts,best_ends, H_avgs, dBdt_avgs, 
                 I_cHs, dlen, sigmas_ic, sigmas_n, segment_indices) = fit_results
                if len(processed_segments) > 0:
                    processed_df = pd.concat(processed_segments, ignore_index=True)
                    
                    # Validate lengths match before assignment
                    if len(processed_df) != len(df):
                        print(f"Processed data length mismatch: "
                            f"original={len(df)}, processed={len(processed_df)}")
                        # Fallback: fill with NaN or skip processed columns
                        df['Processed Current [A]'] = np.full(len(df), np.nan)
                        df['Processed Voltage [V]'] = np.full(len(df), np.nan)
                    else:
                        # Force positional assignment
                        df['Processed Current [A]'] = processed_df['Processed Current [A]'].values
                        df['Processed Voltage [V]'] = processed_df['Processed Voltage [V]'].values
            if w and verbose:
                print(f"[Warning] Fit warnings detected in processing {os.path.basename(fp)}.")
        
        ivs = df

        # print(f"File {os.path.basename(fp)}: \nfit successes {len(fit_successes)}: {fit_successes}\nbest_starts {len(best_starts)}: {best_starts}\n Ics {len(I_cs)}: {I_cs}")

        running_iv_dicts = []
        # pulse_iterator = 0
        for k, fit_success in enumerate(fit_successes):
            segment_id = segment_indices[k]
            pulse_index = segments[segment_id]['IV_Index'].unique()
            if not segments[segment_id].empty:
                if len(pulse_index) == 1:
                    pulse_index = pulse_index[0]
                else:
                    # if verbose:
                    #     print(f"[Warning] Multiple pulse indices found in segment {k}: {segments[k]}") # TODO: Figure out what is happening here.
                    pulse_index = pulse_index[0]
            if fit_success:
                result_summary = {
                    # 'Turns': turn_count,
                    'File': fname,
                    'Target Field [T]': tfield,
                    'Temperature [K]': temperature,
                    'IV_Index': pulse_index, 
                    # 'Fit OK?': fit_success,
                    'fit_start_index': best_starts[k],
                    'fit_end_index': best_ends[k],
                    # 'simple I_c': simple_Ics[k],
                    'I_c': I_cs[k],
                    'I_cH': I_cHs[k],
                    'k': ks[k],
                    'n': bs[k],
                    'R²': r2s[k],
                    'Avg Field [T]': H_avgs[k],
                    'Avg dB/dt [T/s]': dBdt_avgs[k],
                    'Sample': sample,
                    'Orientation': orientation,
                    'Magnet': magnet,
                    'Data_Length': dlen[k],
                    'I_c Error':sigmas_ic[k],
                    'n Error':sigmas_n[k]
                }
                running_iv_dicts.append(result_summary)
                fits.append(result_summary)
                # processed_iv_dicts.append(result_summary)
            # pulse_iterator+=1

        if verbose:
            end = time.perf_counter()
            print(f"Finished: {os.path.basename(fp)} ({end - start:.2f} s)")

        return pd.DataFrame(ivs), pd.DataFrame(fits), ivf

    