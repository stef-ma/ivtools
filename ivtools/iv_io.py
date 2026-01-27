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

class IV_File:
    """
    Container and parser for TDMS-based IV pulse measurements.

    Responsibilities:
    - Validate channel presence
    - Load and resample time-aligned channels
    - Parse embedded TDMS configuration metadata
    - Extract IV pulse timing (Pnum-derived)
    - Compute basic noise metrics

    This class is intentionally stateful: once constructed, all
    downstream processing should rely only on stored attributes.
    """

    # ------------------------------------------------------------------
    # Construction / validation
    # ------------------------------------------------------------------
    def __init__(
        self,
        filepath,
        calibrated_resistor,
        temperature,
        gain_V=1,
        gain_I=1,
        voltage_channel='V',
        current_channel='I',
        ppms_field=None,
    ):
        # --- Load TDMS file ---
        self.tdms_file = nptdms.TdmsFile.read(filepath)
        self.path = Path(filepath)

        # Channel names
        self.vchan = voltage_channel
        self.ichan = current_channel

        # --- Discover groups and channels ---
        self.grp_names = [grp.name for grp in self.tdms_file.groups()]

        all_channels = []
        for grp in self.grp_names:
            for chan in self.tdms_file[grp].channels():
                all_channels.append(chan.name)

        # --- Required channel check ---
        if ppms_field is None:
            required_channels = {voltage_channel, current_channel, 'Field'}
        else:
            required_channels = {voltage_channel, current_channel}

        missing = required_channels - set(all_channels)
        if missing:
            self.passed = False
            print(f'[Error] File {filepath} is missing channels: {missing}.')
            return

        self.passed = True

        # ------------------------------------------------------------------
        # Establish a common reference start time across all channels
        # ------------------------------------------------------------------
        self.start_time = None
        for grp in self.grp_names:
            for chan in self.tdms_file[grp]:
                cfg = self._parse_config(self.tdms_file[grp][chan])
                chan_start = cfg['wf_start_time']
                if self.start_time is None or chan_start < self.start_time:
                    self.start_time = chan_start

        # ------------------------------------------------------------------
        # Load core channels (I, V, B)
        # ------------------------------------------------------------------
        # Current
        self.I, _, _, _ = self._load_channel_data(current_channel)
        self.I = self.I / gain_I
        self.I = self.I / calibrated_resistor  # V -> I conversion

        # Voltage (defines master time axis)
        self.V, self.t, self.configuration, _ = self._load_channel_data(voltage_channel)
        self.V = self.V / gain_V

        # Magnetic field
        if ppms_field is None:
            try:
                self.B, _, _, _ = self._load_channel_data('Field_fixed')
            except Exception:
                print(f'No Field_fixed found in {filepath}, using Field')
                self.B, _, _, _ = self._load_channel_data('Field')
        elif isinstance(ppms_field, (int, float)):
            self.B = np.round(np.ones(len(self.I)) * ppms_field, 2)
        else:
            self.passed = False
            print(f'[Error] File {filepath} is missing channels field value.')
            return

        # ------------------------------------------------------------------
        # Pulse timing (Pnum)
        # ------------------------------------------------------------------
        self.troths, self.tops = self._find_Pnum_indices()

        # Load auxiliary pulse channels
        self.Vavg, self.t_Vavg, _, _ = self._load_channel_data('Vavg')
        self.Pnum, self.t_Pnum, _, _ = self._load_channel_data('Pnum')
        self.Vavg = self.Vavg / gain_V

        # ------------------------------------------------------------------
        # Noise metrics (tail of voltage trace)
        # ------------------------------------------------------------------
        idx = int(len(self.V) * 0.99)
        self.baseline_region = self.V[idx:]
        self.baseline_mean = np.mean(self.baseline_region)
        self.noise_std = np.std(self.baseline_region)
        self.noise_rms = np.sqrt(np.mean(self.baseline_region ** 2))

        # ------------------------------------------------------------------
        # Temperature
        # ------------------------------------------------------------------
        try:
            self.T = float(temperature)
        except Exception:
            self.T = temperature

    # ------------------------------------------------------------------
    # Channel loading
    # ------------------------------------------------------------------
    def _load_channel_data(self, selection):
        """
        Load a TDMS channel and return data, time array, configuration, and channel handle.

        Special cases:
        - Field / Field_fixed are resampled onto the voltage time axis
        - Pnum and Vavg are sparse pulse-timing channels
        """
        # Field channels live in group 'p', everything else in the last group
        group_name = 'p' if selection in ['Field', 'Field_fixed'] else self.grp_names[-1]
        group = self.tdms_file[group_name]
        channel = group[selection]

        config = self._parse_config(channel)
        data = channel.data
        time_array = channel.time_track()

        # Resample field to match voltage time axis
        if selection in ['Field', 'Field_fixed']:
            data_time = self.t
            interp_func = interp1d(time_array, data, kind='linear', fill_value="extrapolate")
            data = interp_func(data_time)
            time_array = data_time

        return data, time_array, config, channel

    # ------------------------------------------------------------------
    # TDMS configuration parsing
    # ------------------------------------------------------------------
    def _parse_config(self, channel):
        """
        Parse the TDMS 'Configuration' string into a nested dictionary.

        - Numeric values are cast when possible
        - 'I-V parameters' style blocks become sub-dictionaries
        - Timing metadata is normalized across channels
        """
        configuration_dict = {}

        # Field channels inherit config from Bdot
        if channel.name in ['Field', 'Field_fixed']:
            channel.properties['Configuration'] = (
                self.tdms_file['p']['Bdot'].properties['Configuration']
            )

        config_text = channel.properties['Configuration']
        lines = config_text.splitlines()

        for line in lines[1:]:
            splits = re.split('\.', line)

            try:
                numeric = float(re.split('"', line)[-2])
            except Exception:
                numeric = re.split('"', line)[-2]

            if 'Re-name chans' in splits[1] or 'Ranges' in splits[1]:
                continue

            if ' = ' not in splits[1]:
                category = splits[1]
                configuration_dict.setdefault(category, {})
                subkey = re.split(' = ', splits[2])[0]
                configuration_dict[category][subkey] = numeric
            else:
                key = re.split(' = ', splits[1])[0]
                configuration_dict[key] = numeric

        # Normalize waveform timing
        configuration_dict['wf_start_offset'] = (
            configuration_dict['Post-trigger delay']
            if channel.name not in ['Vavg', 'Pnum']
            else configuration_dict['Output delay']
        )

        channel.properties['wf_start_offset'] = configuration_dict['wf_start_offset']
        configuration_dict['wf_increment'] = channel.properties['wf_increment']
        configuration_dict['wf_start_time'] = channel.properties['wf_start_time']

        return configuration_dict

    # ------------------------------------------------------------------
    # Pulse edge detection
    # ------------------------------------------------------------------
    def _find_Pnum_indices(self):
        """
        Determine IV pulse edges using the Pnum timing channel.

        Returns:
            troths : np.ndarray
                Interleaved left/right pulse edge indices
            tops : np.ndarray
                Midpoints of pulse regions
        """
        _, Pnum_time, _, _ = self._load_channel_data('Pnum')
        _, data_time, _, _ = self._load_channel_data(self.ichan)

        n = len(Pnum_time)
        if n == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        left_idx = np.searchsorted(data_time, Pnum_time)
        left_idx = np.clip(left_idx, 0, len(data_time) - 1)

        if n == 1:
            right_idx = np.array([
                min(left_idx[0] + 10, len(data_time) - 1)
            ])
            lows = np.array([left_idx[0], right_idx[0]])
            highs = np.array([(left_idx[0] + right_idx[0]) // 2])
            return lows, highs

        right_idx = np.roll(left_idx, -1)
        extrapolated = (
            right_idx[-2] + (right_idx[-2] - right_idx[-3])
            if n > 2
            else left_idx[-1] + 10
        )
        right_idx[-1] = min(len(data_time) - 1, extrapolated)

        lows = np.empty(left_idx.size + right_idx.size, dtype=int)
        lows[0::2] = left_idx
        lows[1::2] = right_idx

        pulse_durations = np.diff(lows)[::2]
        highs = left_idx + pulse_durations // 2

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
