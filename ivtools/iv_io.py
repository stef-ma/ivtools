"""
IV_file.py
------------
Defines the IV_File class for parsing LabActor TDMS waveform data,
extracting current, voltage, field, timing, and metadata.
"""

from pathlib import Path
import re

import numpy as np
import nptdms
from scipy.interpolate import interp1d
import os
import warnings

# import nptdms
# from pathlib import Path
# import numpy as np
# import re
# from scipy.interpolate import interp1d
# import MultiPyVu as mpv

class IV_File:
    def __init__(
            self,
            filepath,
            calibrated_resistor,
            temperature,
            gain_V=1,
            gain_I=1,
            voltage_channel='V',
            current_channel='I',
            ppms_field=None
        ):
        # Read the TDMS file and initialize attributes
        self.tdms_file = nptdms.TdmsFile.read(filepath)

        self.grp_names = []
        for grp in self.tdms_file.groups():
            self.grp_names.append(grp.name)

        self.path = Path(filepath)
        self.current_chan = current_channel
        # Check for required channels
        chans = []
        for chan in self.tdms_file[self.grp_names[0]].channels():
            chans.append(chan.name)
        if ppms_field is None:
            required_channels = {voltage_channel, current_channel, 'Field'}
        else:
            required_channels = {voltage_channel, current_channel}
        missing = required_channels - set(chans)
        if missing:
            self.passed = False
            print(f'[Error] File {filepath} is missing channels: {missing}.')
        else:
            # If all required channels are present, load the data
            self.passed = True
            self.I, _, _, _=self._load_channel_data(current_channel)
            self.I = self.I/gain_I
            self.I = self.I/calibrated_resistor # Converting the measured voltage drop to current.
            self.V, self.t, self.configuration, _ =self._load_channel_data(voltage_channel)
            self.V = self.V/gain_V
            
            if ppms_field is None: # For files with field channel
                try:
                    self.B, _, _, _=self._load_channel_data('Field_fixed')
                except:
                    print(f'No Field_fixed found in {filepath}, using Field')
                    self.B, _, _, _=self._load_channel_data('Field') #TODO: Should be Field_fixed?
            elif isinstance(ppms_field, float) or isinstance(ppms_field,int): # For files without field channel but with a constant field value
                self.B = np.round(np.ones(len(self.I))*ppms_field,2)
            else:
                self.passed = False
                print(f'[Error] File {filepath} is missing channels field value.')
                                            
            # Record indices of troths and tops
            self.troths, self.tops = self._find_Pnum_indices()

            # Record Vavg and Pnum
            self.Vavg, self.t_Vavg, _, _=self._load_channel_data('Vavg')
            self.Pnum, self.t_Pnum, _, _ = self._load_channel_data('Pnum')
            self.Vavg = self.Vavg/gain_V

            # Record noise levels
            idx = int(len(self.V)*.99)
            self.baseline_region = self.V[idx:]
            self.baseline_mean = np.mean(self.baseline_region)
            self.noise_std = np.std(self.baseline_region)
            self.noise_rms = np.sqrt(np.mean(self.baseline_region**2))


            # Record temperature
            try:
                self.T = float(temperature)
            except:
                self.T = temperature
                 
    def _load_channel_data(self,selection): # With datetime start time
        '''
        Loads a channel from the p group of the TDMS file. Special treatments:
            - Pnum is a smaller array only noting the times at which IV pulses are applied. We only use it to get the 
            relevant indices
            - Field is sampled at the National Instruments DAQ which has a different sampling rate than the Red Pitaya.
            The field needs to be upsampled.
        '''
        group = self.tdms_file[self.grp_names[0]] # Our TDMS files have only one group.
        channel=group[selection]
        c = self._parse_config(channel)
        data = channel.data
        if selection not in ['Pnum','Vavg','Field','Field_fixed']:
            time_array = np.arange(len(data))*c['wf_increment']+c['Post-trigger delay']
        elif selection == 'Pnum':
            time_array = np.arange(len(data))*c['wf_increment']+c['Output delay']#+c['wf_increment']/4
        elif selection == 'Vavg':
            time_array = np.arange(len(data))*c['wf_increment']+c['Output delay']#+c['wf_increment']/4
        else: # Resample the field
            time_array = np.arange(len(data))*c['wf_increment']
            _,data_time,data_c,_=self._load_channel_data(self.current_chan) # Can also be V, or other channels from the Pitaya. EXCEPT Pnum!
            # start_time_difference = c['wf_start_time'] - data_c['wf_start_time'] # TODO: FIgure out why this is not needed anymore???
            # print(start_time_difference/1000)
            # time_array = time_array + start_time_difference.astype(int)/1e6  # Adjusting the time array to match the start time of the data.
            interp_func = interp1d(time_array, data, kind='linear', fill_value="extrapolate")
            data = interp_func(data_time)  # Resampling the data.
            time_array = data_time # Taking the correct time.

            # print(selection,c['wf_start_time'],start_time_difference.astype(int)/1e6)
        return data, time_array, c, channel

    def _parse_config(self,channel):
        '''
        Parsing algorithm for the configuration data contained in the TDMS. Mostly regex to reject the useless information.
        Extracts and converts the numerical values. The key "I-V parameters" is its own dictionary.
        '''
        configuration_dict=dict()
        if channel.name != 'Field' and channel.name!='Field_fixed':
            config = channel.properties['Configuration']
            parsed_config = config.splitlines()
            for i,line in enumerate(parsed_config[1:]):
                # print(i,line)
                splits = re.split('\.',line)
                numeric = re.split('"',line)[-2]
                try:
                    numeric=float(numeric)
                except:
                    pass
                if 'Re-name chans' in splits[1] or 'Ranges' in splits[1]:
                    pass
                else:
                    if ' = ' not in splits[1]:
                        category = splits[1]
                        if category not in configuration_dict.keys():
                            configuration_dict[category]=dict()
                            category2 = re.split(' = ',splits[2])[0]
                            if category not in configuration_dict[category].keys():
                                # print(f'Inner category saved: {category2}') 
                                configuration_dict[category][category2]=numeric
                        else:
                            category2 = re.split(' = ',splits[2])[0]
                            if category not in configuration_dict[category].keys():
                                # print(f'Inner category saved: {category2}') 
                                configuration_dict[category][category2]=numeric
                    else:
                        category = re.split(' = ',splits[1])[0]
                        if category not in configuration_dict.keys():
                            # print(f'Outer category saved: {category}')
                            configuration_dict[category]=numeric
        configuration_dict['wf_increment']=channel.properties['wf_increment']
        configuration_dict['wf_start_time']=channel.properties['wf_start_time']
        return configuration_dict

    def _find_Pnum_indices(self):
        """
        Determine the index positions of IV pulse edges using the Pnum channel.

        Returns:
            lows (np.ndarray): Interleaved array of indices marking rising and falling edges.
            highs (np.ndarray): Estimated midpoints of pulse regions.
        """
        # Load Pnum pulse times and the common data time array
        _, Pnum_time, _, _ = self._load_channel_data('Pnum')
        _, data_time, _, _ = self._load_channel_data(self.current_chan)

        n = len(Pnum_time)
        if n == 0:
            # No pulses at all
            return np.array([], dtype=int), np.array([], dtype=int)

        # Map each Pnum_time to nearest insertion point in data_time
        left_idx = np.searchsorted(data_time, Pnum_time)
        left_idx = np.clip(left_idx, 0, len(data_time) - 1)

        if n == 1:
            # Only one pulse time — assume a minimal duration
            # Here we define a short artificial right edge
            right_idx = np.array([min(left_idx[0] + 10, len(data_time) - 1)], dtype=int)
            lows = np.array([left_idx[0], right_idx[0]], dtype=int)
            highs = np.array([(left_idx[0] + right_idx[0]) // 2], dtype=int)
            return lows, highs

        # For normal multi-pulse case
        right_idx = np.roll(left_idx, -1)
        # Extrapolate final right edge safely
        extrapolated = right_idx[-2] + (right_idx[-2] - right_idx[-3]) if n > 2 else left_idx[-1] + 10
        right_idx[-1] = min(len(data_time) - 1, extrapolated)

        # Interleave left/right for 'lows'
        lows = np.empty(left_idx.size + right_idx.size, dtype=int)
        lows[0::2] = left_idx
        lows[1::2] = right_idx

        # Estimate pulse tops as midpoints of rising/falling edge pairs
        pulse_durations = np.diff(lows)[::2]  # get only left-to-right gaps
        highs = left_idx + pulse_durations // 2  # integer midpoint

        return lows, highs



def extract_numeric_temperature(temp):
    """
    Extract the numeric part from a temperature value that might be a number or a string containing a number.

    Args:
        temp (str or float or int): Temperature value, possibly a string with notes.

    Returns:
        float: The extracted numeric temperature. Returns None if no numeric value found.
    """
    if isinstance(temp, (int, float)):
        return float(temp)
    elif isinstance(temp, str):
        match = re.search(r"[-+]?\d*\.?\d+", temp)
        if match:
            return float(match.group())
    return None

# def save_ivdata_for_Origin(raw_df,fname,base_path,sample,orientation,magnet,tfield,temperature,fnames=False,verbose=False):

#     output_base = f"{fname}_{sample}_{orientation}_{magnet}_{tfield}T_{temperature}K"

#     raw_path = base_path / f"{output_base}_OriginReadable_raw.csv"



#     #Origin Reformatting
#     with open(raw_path, 'w') as f:
#         header = f'{temperature} K | {tfield} T | {orientation} deg | {magnet} | {fname}'
#         f.write(f'# {header}\n')
#         f.write(f'I,V,Uncorrected V\-(avg),Processed Voltage,\g(m)\-(0)H,dH/dt,Pulse idx,t\n' if not fnames else f'I,V,Uncorrected V\-(avg),Processed Voltage,\g(m)\-(0)H,dH/dt,Pulse idx,t,File\n')
#         f.write(f'A,V,V,V,T,T/s, ,s, \n' if not fnames else f'A,V,V,V,T,T/s, ,s, , \n')
#         f.write(f'{header},{header},{header},{header},{header},{header},{header},{header}\n' if not fnames else f'{header},{header},{header},{header},{header},{header},{header},{header},{header}\n')

    
#     raw_df['Current_A'] = raw_df['Current [A]']
#     raw_df['Voltage_V'] = raw_df['Voltage [V]']
#     raw_df['Vavg_V'] = raw_df['Vavg [V]']
#     raw_df['Processed_Voltage_V'] = raw_df['Processed Voltage [V]'] if 'Processed Voltage [V]' in raw_df.columns else np.nan 
#     raw_df['Field_T'] = raw_df['Field [T]']
#     raw_df['dBdt'] = raw_df['dBdt [T/s]']
#     raw_df['pdx'] = raw_df['IV_Index']
#     raw_df['time_s'] = raw_df['Time [s]']
#     raw_df['fnames'] = raw_df['File']


#     if not fnames:
#         raw_df = raw_df[['Current_A','Voltage_V','Vavg_V', 'Processed_Voltage_V','Field_T','dBdt','pdx','time_s']]
#     else:
#         raw_df = raw_df[['Current_A','Voltage_V','Vavg_V', 'Processed_Voltage_V','Field_T','dBdt','pdx','time_s','fnames']]
    

#     raw_df.to_csv(raw_path, mode='a', sep=',', header=False, index=False)
#     if verbose:
#         print(f'Saved fit results to: {raw_path}')

# def save_fitdata_for_Origin(fit_df,fname,base_path,sample,orientation,magnet,tfield,temperature,fnames=False,verbose=False,cross_section=None,width=None):
        
#     output_base = f"{fname}_{sample}_{orientation}_{magnet}_{tfield}T_{temperature}K"

#     fit_path = base_path/ f"{output_base}_OriginReadable_fit.csv"

#     #Origin Reformatting
#     with open(fit_path, 'w') as f:
#         header = f'{temperature} K | {tfield} T | {orientation} deg | {magnet} | {fname}'
#         f.write(f'# {header}\n')
#         f.write(f'\g(m)\-(0)H,I\-(c),I\-(c) (dH/dt>0),I\-(c) (dH/dt<=0),I\-(cpw),J\-(c),k,n,n (dH/dt>0),n (dH/dt<=0),Pulse idx,Fit start idx,Fit end idx\n' if not fnames else f'\g(m)\-(0)H,I\-(c),I\-(c) (dH/dt>0),I\-(c) (dH/dt<=0),I\-(cpw),J\-(c),k,n,n (dH/dt>0),n (dH/dt<=0),Pulse idx,Fit start idx,Fit end idx,File\n')
#         f.write(f'T,A,A,A,A/cm-w,MA/cm\+(2), , , , , \n' if not fnames else f'T,A,A,A,A/cm-w,MA/cm\+(2), , , , , , \n')
#         f.write(f'{header},{header},{header},{header},{header},{header},{header},{header},{header},{header},{header},{header},{header}\n' if not fnames else f'{header},{header},{header},{header},{header},{header},{header},{header},{header},{header},{header},{header},{header},{header}\n')
    


#     fit_df['Field_T'] = fit_df['Avg Field [T]']
#     fit_df['I_c_A'] = fit_df['I_c']
#     fit_df['I_c_A_pos_dBdt'] = fit_df[fit_df['Avg dB/dt [T/s]']>0]['I_c']
#     fit_df['I_c_A_neg_dBdt'] = fit_df[fit_df['Avg dB/dt [T/s]']<=0]['I_c']
#     fit_df['I_cpw_Acmw'] = fit_df['I_c']/width if width else np.nan
#     fit_df['J_c_MAcm2'] = fit_df['I_c']/cross_section/1e6 if cross_section else np.nan
#     fit_df['n'] = fit_df['n']
#     fit_df['n_pos_dBdt'] = fit_df[fit_df['Avg dB/dt [T/s]']>0]['n']
#     fit_df['n_neg_dBdt'] = fit_df[fit_df['Avg dB/dt [T/s]']<=0]['n']
#     fit_df['k'] = fit_df['k']
#     fit_df['pdx'] = fit_df['IV_Index']
#     fit_df['fit_start_idx'] = fit_df['fit_start_index']
#     fit_df['fit_end_idx'] = fit_df['fit_end_index']
#     fit_df['fnames'] = fit_df['File']


#     if not fnames:
#         fit_df = fit_df[['Field_T','I_c_A','I_c_A_pos_dBdt','I_c_A_neg_dBdt','I_cpw_Acmw','J_c_MAcm2','k','n','n_pos_dBdt','n_neg_dBdt','pdx','fit_start_idx','fit_end_idx']]
#     else:
#         fit_df = fit_df[['Field_T','I_c_A','I_c_A_pos_dBdt','I_c_A_neg_dBdt','I_cpw_Acmw','J_c_MAcm2','k','n','n_pos_dBdt','n_neg_dBdt','pdx','fit_start_idx','fit_end_idx','fnames']]

#     fit_df.to_csv(fit_path, header=False, mode='a', sep=',', index=False)

#     if verbose:
#         print(f'Saved fit results to: {fit_path}')



# -----------------------
# Column metadata
# -----------------------
COLUMN_META_RAW = {
    "Current_A":              ("I", "A"),
    "Voltage_V":              ("V", "V"),
    "Vavg_V":                 ("Uncorrected Vavg", "V"),
    "Processed_Voltage_V":    ("Processed Voltage", "V"),
    "Field_T":                ("Field", "T"),
    "dBdt":                   ("dH/dt", "T/s"),
    "pdx":                    ("IV idx", ""),
    "time_s":                 ("t", "s"),
    "fnames":                 ("File", ""),
}
COLUMN_META_FIT = {
    "Field_T":                ("Field", "T"),
    "I_c_A":                  ("Ic", "A"),
    "I_c_A_pos_dBdt":         ("Ic (dH/dt>0)", "A"),
    "I_c_A_neg_dBdt":         ("Ic (dH/dt<=0)", "A"),
    "I_c_A_err":              ("Ic Error", "A"),
    "I_cpw_Acmw":             ("Icpw", "A/cm-w"),
    "J_c_MAcm2":              ("Jc", "MA/cm^2"),
    "k":                      ("k", ""),
    "n":                      ("n", ""),
    "n_pos_dBdt":             ("n (dH/dt>0)", ""),
    "n_neg_dBdt":             ("n (dH/dt<=0)", ""),
    "n_err":                  ("n Error", ""),
    "pdx":                    ("IV idx", ""),
    "fit_start_idx":          ("Fit start idx", ""),
    "fit_end_idx":            ("Fit end idx", ""),
    "fnames":                 ("File", ""),
}
COLUMN_META_RAW_ORIGIN = {
    "Current_A":              ("I", "A"),
    "Voltage_V":              ("V", "V"),
    "Vavg_V":                 ("Uncorrected V\-(avg)", "V"),
    "Processed_Voltage_V":    ("Processed Voltage", "V"),
    "Field_T":                ("\g(m)-(0)H", "T"),
    "dBdt":                   ("dH/dt", "T/s"),
    "pdx":                    ("IV idx", ""),
    "time_s":                 ("t", "s"),
    "fnames":                 ("File", ""),
}
COLUMN_META_FIT_ORIGIN = {
    "Field_T":                ("\g(m)\-(0)H", "T"),
    "I_c_A":                  ("I\-(c)", "A"),
    "I_c_A_pos_dBdt":         ("I\-(c) (dH/dt>0)", "A"),
    "I_c_A_neg_dBdt":         ("I\-(c) (dH/dt<=0)", "A"),
    "I_c_A_err":              ("I\-(c) Error", "A"),
    "I_cpw_Acmw":             ("I\-(cpw)", "A/cm-w"),
    "J_c_MAcm2":              ("J\-(c)", "MA/cm^2"),
    "k":                      ("k", ""),
    "n":                      ("n", ""),
    "n_pos_dBdt":             ("n (dH/dt>0)", ""),
    "n_neg_dBdt":             ("n (dH/dt<=0)", ""),
    "n_err":                  ("n Error", ""),
    "pdx":                    ("IV idx", ""),
    "fit_start_idx":          ("Fit start idx", ""),
    "fit_end_idx":            ("Fit end idx", ""),
    "fnames":                 ("File", ""),
}


FIT_PRESETS = {
    "full": [
        "Field_T",
        "I_c_A",
        "I_c_A_pos_dBdt",
        "I_c_A_neg_dBdt",
        "I_cpw_Acmw",
        "J_c_MAcm2",
        "k",
        "n",
        "n_pos_dBdt",
        "n_neg_dBdt",
        "pdx",
        "fit_start_idx",
        "fit_end_idx",
        "fnames",
    ],
    "minimal": [
        "Field_T",
        "I_c_A",
        "pdx",
        "fnames",
    ],
    "minimal_expanded": [
        "Field_T",
        "I_c_A",
        "I_c_A_err",
        "n",
        "n_err",
        "k",
        "pdx",
        "fnames",
    ]
}

IV_PRESETS = {
    "full": [
        "Current_A",
        "Voltage_V",
        "Vavg_V",
        "Processed_Voltage_V",
        "Field_T",
        "dBdt",
        "pdx",
        "time_s",
        "fnames",
    ],
    "minimal": [
        "time_s",
        "Current_A",
        "Voltage_V",
        "Processed_Voltage_V",
        "Field_T",
        "dBdt",
        "pdx",
        "fnames",
    ],
    "minimal_expanded": [
        "time_s",
        "Current_A",
        "Voltage_V",
        "Processed_Voltage_V",
        "Field_T",
        "dBdt",
        "pdx",
        "fnames",
    ],
}


# -----------------------
# Helper: build headers from metadata dict (injected)
# -----------------------
def build_origin_headers(column_order, column_meta, header_comment):
    """
    Returns three strings: label_row, unit_row, meta_row
    column_order: list of internal column names
    column_meta: mapping internal -> (label, unit)
    header_comment: string to repeat in meta row
    """
    labels = []
    units = []
    metas = []
    for c in column_order:
        meta = column_meta.get(c)
        if meta is None:
            # fallback: use the column name, empty unit
            labels.append(c)
            units.append("")
        else:
            labels.append(meta[0])
            units.append(meta[1])
        metas.append(header_comment)
    label_row = ",".join(labels)
    unit_row = ",".join(units)
    meta_row = ",".join(metas)
    return label_row, unit_row, meta_row

# -----------------------
# Raw save function (user-selectable columns)
# -----------------------
def save_ivdata(
    raw_df,
    fname,
    base_path,
    sample,
    orientation,
    magnet,
    tfield,
    temperature,
    preset=None,
    columns=IV_PRESETS["full"],
    column_meta=None,
    origin = False,
    verbose=False,
):
    """
    Save raw IV data in an Origin-readable CSV with flexible column selection.

    - columns: list of internal column names to write (defaults provided)
    - column_meta: dict mapping internal names -> (label, unit). Defaults to COLUMN_META_RAW.
    """
    base_path = Path(base_path)
    if column_meta is None:
        column_meta = COLUMN_META_RAW if not origin else COLUMN_META_RAW_ORIGIN

    if preset is not None:
        columns = IV_PRESETS[preset]
    # default columns
    if columns is None:
        columns = [
            "Current_A","Voltage_V","Vavg_V","Processed_Voltage_V",
            "Field_T","dBdt","pdx","time_s"
        ]
    

    # Normalize / create internal columns from available raw_df fields, safe-get
    raw_df = raw_df.copy()
    raw_df["Current_A"]           = raw_df.get("Current [A]", raw_df.get("Current_A", np.nan))
    raw_df["Voltage_V"]           = raw_df.get("Voltage [V]", raw_df.get("Voltage_V", np.nan))
    raw_df["Vavg_V"]              = raw_df.get("Vavg [V]", raw_df.get("Vavg_V", np.nan))
    raw_df["Processed_Voltage_V"] = raw_df.get("Processed Voltage [V]", raw_df.get("Processed_Voltage_V", np.nan))
    raw_df["Field_T"]             = raw_df.get("Field [T]", raw_df.get("Field_T", np.nan))
    raw_df["dBdt"]                = raw_df.get("dBdt [T/s]", raw_df.get("dBdt", np.nan))
    raw_df["pdx"]                 = raw_df.get("IV_Index", raw_df.get("pdx", np.nan))
    raw_df["time_s"]              = raw_df.get("Time [s]", raw_df.get("time_s", np.nan))
    raw_df["fnames"]              = raw_df.get("File", raw_df.get("fnames", ""))

    # validate requested columns
    missing = [c for c in columns if c not in raw_df.columns]
    if missing:
        warnings.warn(f"The following requested columns are not present and will be filled with NaN or dropped: {missing}")
        # Keep them (they may be in column_meta) but ensure we can output something:
        for c in missing:
            if c not in raw_df.columns:
                raw_df[c] = np.nan

    # output_base = f"{fname}_{sample}_{orientation}_{magnet}_{tfield}T_{temperature}K"
    output_base = f"IV_{temperature}K_{tfield}T_{orientation}deg_{fname}"
    if origin:
        output_base = output_base + '_OriginReadable'
    raw_path = base_path / f"{output_base}_ivs.csv"
    header_comment = f"{temperature} K | {tfield} T | {orientation} deg | {magnet} | {fname}"

    # Build header rows
    label_row, unit_row, meta_row = build_origin_headers(columns, column_meta, header_comment)

    # Write file
    with open(raw_path, "w") as f:
        f.write(f"# {header_comment}\n")
        f.write(label_row + "\n")
        f.write(unit_row + "\n")
        if origin:
            f.write(meta_row + "\n")

    # Save data rows (ensure columns order)
    raw_df.to_csv(raw_path, mode="a", header=False, index=False, columns=columns)

    if verbose:
        print(f"Saved raw Origin-readable data to: {raw_path}")

# -----------------------
# Fit save function (user-selectable columns)
# -----------------------
def save_fitdata(
    fit_df,
    fname,
    base_path,
    sample,
    orientation,
    magnet,
    tfield,
    temperature,
    preset=None,
    columns=FIT_PRESETS["full"],
    column_meta=None,
    origin = False,
    cross_section=None,
    width=None,
    verbose=False,
):
    """
    Save fit results with flexible column selection.
    - columns: list of internal column names to write
    - column_meta: mapping for labels/units (defaults to COLUMN_META_FIT)
    """
    base_path = Path(base_path)
    if column_meta is None:
        column_meta = COLUMN_META_FIT if not origin else COLUMN_META_FIT_ORIGIN

    if preset is not None:
        columns = FIT_PRESETS[preset]

    # default columns
    if columns is None:
        columns = [
            "Field_T","I_c_A","I_c_A_pos_dBdt","I_c_A_neg_dBdt",
            "I_cpw_Acmw","J_c_MAcm2","k","n","n_pos_dBdt","n_neg_dBdt",
            "pdx","fit_start_idx","fit_end_idx"
        ]


    fit_df = fit_df.copy()
    # compute/normalize the internal names used by downstream CSV
    fit_df["Field_T"] = fit_df.get("Avg Field [T]", fit_df.get("Field_T", np.nan))
    fit_df["I_c_A"] = fit_df.get("I_c", fit_df.get("I_c_A", np.nan))
    # Pos/neg splits: create series aligned with full df by masking
    fit_df["I_c_A_pos_dBdt"] = np.where(fit_df.get("Avg dB/dt [T/s]", 0) > 0, fit_df.get("I_c", np.nan), np.nan)
    fit_df["I_c_A_neg_dBdt"] = np.where(fit_df.get("Avg dB/dt [T/s]", 0) <= 0, fit_df.get("I_c", np.nan), np.nan)
    fit_df["I_cpw_Acmw"] = (fit_df.get("I_c", np.nan) / width) if width else np.nan
    fit_df["J_c_MAcm2"] = (fit_df.get("I_c", np.nan) / cross_section / 1e6) if cross_section else np.nan
    fit_df["n"] = fit_df.get("n", np.nan)
    fit_df["n_pos_dBdt"] = np.where(fit_df.get("Avg dB/dt [T/s]", 0) > 0, fit_df.get("n", np.nan), np.nan)
    fit_df["n_neg_dBdt"] = np.where(fit_df.get("Avg dB/dt [T/s]", 0) <= 0, fit_df.get("n", np.nan), np.nan)
    fit_df["k"] = fit_df.get("k", np.nan)
    fit_df["pdx"] = fit_df.get("IV_Index", fit_df.get("pdx", np.nan))
    fit_df["fit_start_idx"] = fit_df.get("fit_start_index", fit_df.get("fit_start_idx", np.nan))
    fit_df["fit_end_idx"] = fit_df.get("fit_end_index", fit_df.get("fit_end_idx", np.nan))
    fit_df["fnames"] = fit_df.get("File", fit_df.get("fnames", ""))
    fit_df["I_c_A_err"] = fit_df.get("I_c Error", fit_df.get("I_c_err", np.nan))
    fit_df["n_err"] = fit_df.get("n Error", fit_df.get("n_err", np.nan))

    # validate columns
    missing = [c for c in columns if c not in fit_df.columns]
    if missing:
        warnings.warn(f"The following requested fit columns are not present and will be filled with NaN or dropped: {missing}")
        for c in missing:
            if c not in fit_df.columns:
                fit_df[c] = np.nan

    # output_base = f"{fname}_{sample}_{orientation}_{magnet}_{tfield}T_{temperature}K"
    output_base = f"IcH_{temperature}K_{tfield}T_{orientation}deg_{fname}"
    if origin:
        output_base = output_base + '_OriginReadable'
    fit_path = base_path / f"{output_base}_fit.csv"
    header_comment = f"{temperature} K | {tfield} T | {orientation} deg | {magnet} | {fname}"

    label_row, unit_row, meta_row = build_origin_headers(columns, column_meta, header_comment)

    with open(fit_path, "w") as f:
        f.write(f"# {header_comment}\n")
        f.write(label_row + "\n")
        f.write(unit_row + "\n")
        if origin:
            f.write(meta_row + "\n")

    fit_df.to_csv(fit_path, mode="a", header=False, index=False, columns=columns)

    if verbose:
        print(f"Saved fit Origin-readable data to: {fit_path}")
