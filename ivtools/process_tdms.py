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

def process_tdms(
    fp,
    sample,
    temperature,
    angle,
    tfield,
    resistor,
    voltage_cutoff,
    noise_level,
    voltage_gain,
    current_gain,
    linear_sub_criterion,
    power_law_criterion,
    voltage_channel='Voltage',
    current_channel='Current',
    magnet='Mid Pulse',
    ppms_field=None,
    verbose = False
    ):
    
    fname = os.path.basename(fp)

    ivf = iv_io.IV_File(
        fp, 
        resistor, 
        temperature, 
        voltage_gain, 
        current_gain, 
        voltage_channel=voltage_channel,
        current_channel=current_channel,
        ppms_field=ppms_field
        )

    if not ivf.passed:
        return pd.DataFrame([]), pd.DataFrame([]), ivf
    
    if verbose:
        start = time.perf_counter()

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
                result, *_ = fit_utils.process_IV_pulse(ivf, top, lows[2 * j], lows[2 * j + 1])
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
            fit_successes, I_cs, ks, bs, r2s, segments, segments_power,processed_segments, best_starts,best_ends, H_avgs, dBdt_avgs, I_cHs, dlen = fitting.fit_IV_for_Ic(
                df, 
                voltage_cutoff, 
                linear_sub_criterion,
                power_law_criterion,
                min_fit_points=3,
                max_fit_points=5,
                noise_level = noise_level
                )
            if len(processed_segments)>0:
                processed_df = pd.concat(segments, ignore_index=True)
                df['Processed Current [A]'] = processed_df['Current [A]']
                df['Processed Voltage [V]'] = processed_df['Voltage [V]']
        if w and verbose:
            print(f"[Warning] Fit warnings detected in processing {os.path.basename(fp)}.")
    
    ivs = df

    # print(f"File {os.path.basename(fp)}: \nfit successes {len(fit_successes)}: {fit_successes}\nbest_starts {len(best_starts)}: {best_starts}\n Ics {len(I_cs)}: {I_cs}")

    running_iv_dicts = []
    # pulse_iterator = 0
    for k, fit_success in enumerate(fit_successes):
        pulse_index = segments[k]['IV_Index'].unique()
        if not segments[k].empty:
            if len(pulse_index) == 1:
                pulse_index = pulse_index[0]
            else:
                if verbose:
                    print(f"[Warning] Multiple pulse indices found in segment {k}: {segments[k]}") # TODO: Figure out what is happening here.
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
                'Data_Length': dlen[k]
            }
            running_iv_dicts.append(result_summary)
            fits.append(result_summary)
            # processed_iv_dicts.append(result_summary)
        # pulse_iterator+=1

    if verbose:
        end = time.perf_counter()
        print(f"Finished: {os.path.basename(fp)} ({end - start:.2f} s)")

    return pd.DataFrame(ivs), pd.DataFrame(fits), ivf

    