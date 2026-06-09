# ivtools/ppms_io.py
"""
PPMS data handlers for IV Tools.
Provides compatibility layer between PPMS data formats and ivtools processing pipeline.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

try:
    import MultiPyVu as mpv
    MPV_AVAILABLE = True
except ImportError:
    MPV_AVAILABLE = False
    print("[Warning] MultiPyVu not installed. PPMS functionality will be limited.")


class ACT_File:
    """
    Container for PPMS QD ACT measurement data.
    
    Supports three measurement types:
    - 'IV': Continuous voltage-current sweeps with automatic segmentation
    - 'Ic': Critical current measurements with statistical error bars
    - 'resistance': Resistance measurements (R(T), R(H))
    
    Mimics IV_File structure for compatibility with ivtools processing pipeline.
    
    Key Differences from Pulsed IV_File
    -----------------------------------
    - Multiple continuous sweeps (detected via current drops or time gaps)
    - Static or slowly ramped magnetic field
    - Auto-generated segment counters (Pnum_iv, Pnum_segment)
    - Direct Ic extraction from ACT software (no fitting required)
    
    Parameters
    ----------
    filepath : str or Path
        Path to PPMS .dat file
    sample : str
        Sample identifier
    measurement : {'IV', 'Ic', 'resistance'}
        Measurement type
    channel : int, default=1
        ACT measurement channel (1-2)
    temperature_channel : str, default='Temperature (K)'
        Column name for sample temperature
    orientation : float, optional
        Field orientation in degrees. If None, extracted from file
    drop_factor : float, default=0.5
        Current drop threshold for IV segmentation (0.5 = 50% of range)
    verbose : bool, default=False
        Print detailed loading diagnostics

    Attributes (All)
    ----------------------------
    B : ndarray
        Magnetic field (T)
    T : ndarray
        Temperature (K)
    orientation : ndarray or float
        Field orientation (degrees)
    df : Pandas DataFrame
        mpv processed .dat file in the form of DataFrame
        
    Attributes (IV measurements)
    ----------------------------
    I : ndarray
        Current array (A)
    V : ndarray
        Voltage array (V)
    V_noise : ndarray
        Voltage standard deviation per point (V)
    Pnum_iv : ndarray
        Within-IV counter (resets at segment boundaries)
    Pnum_segment : ndarray
        Within-quadrant counter (resets at inflection points)
        
    Attributes (Ic measurements)
    ----------------------------
    Ic : ndarray
        Critical current (A)
    Ic_err : ndarray
        Critical current uncertainty (A)
    fits : DataFrame
        Ic results in ivtools fit format (for compatibility)
    
    Attributes (resistance measurements)
    ------------------------------------
    R : ndarray
        Resistance (Ω·cm)
    R_err : ndarray
        Resistance standard deviation (Ω·cm)
    
    Notes
    -----
    - For IV measurements, use with ivtools.process.process_ivf()
    - For Ic measurements, use ACT_File.fits directly
    - For resistance measurements, use R, T, B arrays directly
    - Voltage noise is ACT's per-point standard deviation, not file-level baseline
    """
    
    def __init__(
        self,
        filepath,
        sample,
        measurement,
        channel=1,
        temperature_channel='Temperature (K)',
        orientation=None,
        drop_factor=0.5,
        verbose=False
    ):

        
        if not MPV_AVAILABLE:
            raise ImportError("MultiPyVu is required for PPMS data. Install with: pip install MultiPyVu")
        self.fp = filepath
        self.measurement = measurement
        self.path = Path(filepath)
        self.passed = False
        
        # Parse using MultiPyVu
        try:
            datafile = mpv.DataFile().parse_MVu_data_file(str(filepath))
        except Exception as e:
            print(f"[Error] Failed to parse PPMS file {filepath}: {e}")
            return
        
        if str(channel) not in ['1','2']:
            print('[Error] Wrong channel value. Use 1 or 2.') 
            return
        
        self.sample = sample

        df = self._ensure_numeric(datafile, datafile.columns)
        self.df = df

        # ============================================================
        # FILTER BY MEASUREMENT TYPE
        # ============================================================
        if measurement == 'IV':
            # Keep only rows with valid IV data
            required_col = f'Volts ch{str(channel)}'
            if required_col not in df.columns:
                print(f"[Error] Required column '{required_col}' not found")
                return
            
            # Filter out rows where voltage is NaN
            valid_mask = df[required_col].notna()
            df = df[valid_mask].reset_index(drop=True)
            
            if len(df) == 0:
                print(f"[Error] No valid IV measurements found in {filepath}")
                return
        
        elif measurement == 'Ic':
            # Keep only rows with valid Ic data
            ic_channel = f'Crit.Cur. ch{channel} (mA)'
            if ic_channel not in df.columns:
                print(f"[Error] Required column '{ic_channel}' not found")
                return
            
            # Filter out rows where Ic is NaN
            valid_mask = df[ic_channel].notna()
            df = df[valid_mask].reset_index(drop=True)
            
            if len(df) == 0:
                print(f"[Error] No valid Ic measurements found in {filepath}")
                return
        
        elif measurement == 'resistance':
            # Keep only rows with valid resistance data
            r_channel = f'Res. ch{channel} (ohm-cm)'
            if r_channel not in df.columns:
                print(f"[Error] Required column '{r_channel}' not found")
                return
            
            # Filter out rows where resistance is NaN
            valid_mask = df[r_channel].notna()
            df = df[valid_mask].reset_index(drop=True)
            
            if len(df) == 0:
                print(f"[Error] No valid resistance measurements found in {filepath}")
                return
        
        else:
            print(f"[Error] Invalid measurement type '{measurement}'. Use 'IV', 'Ic', or 'resistance'.")
            return

        available_columns = df.columns.tolist()
        
        if verbose:
            print(f"[Info] Available columns: {available_columns}")
        
        # Extract time (with fallback to index)
        if 'Time Stamp (sec)' in df.columns:
            self.t = df['Time Stamp (sec)'].to_numpy()
            self.t -= self.t[0]
        else:
            if verbose:
                print(f"[Warning] {'Time Stamp (sec)'} not found, using index as time")
            self.t = np.arange(len(df))

        # Extract temperature (REQUIRED)
        if temperature_channel not in df.columns:
            print(f"[Error] Required column '{temperature_channel}' not found")
            return
        self.T = df[temperature_channel].to_numpy()
        
        # Extract or construct magnetic field
        if 'Magnetic Field (Oe)' in df.columns:
            # Convert Oe → T
            self.B = df['Magnetic Field (Oe)'].to_numpy() / 1e4
        else:
            print(f"[Error] No field data found")
            return
        
        if orientation is None:
            # Extract orientation (only if not given)
            orientation_channel = 'Sample Position (deg)'
            if orientation_channel not in df.columns:
                print(f"[Error] Required column '{orientation_channel}' not found")
                return
            self.orientation = df[orientation_channel].to_numpy()
        else:
            self.orientation = orientation

        # ============================================================
        # MEASUREMENT-SPECIFIC PROCESSING
        # ============================================================
        if measurement == 'IV':
        
            # Extract current (REQUIRED)
            if 'Excitation (mA)' not in df.columns:
                print(f"[Error] Required column 'Excitation (mA)' not found")
                return
            self.I = df['Excitation (mA)'].to_numpy()/1e3
            
            # Extract voltage (REQUIRED)
            voltage_channel = f'Volts ch{str(channel)}'
            if voltage_channel not in df.columns:
                print(f"[Error] Required column '{voltage_channel}' not found")
                return
            self.V = df[voltage_channel].to_numpy()

            # Extract voltage noise (REQUIRED)
            voltage_noise_channel = f'V Std.Dev. ch{str(channel)}'
            if voltage_noise_channel not in df.columns:
                print(f"[Error] Required column '{voltage_noise_channel}' not found")
                return
            self.V_noise = df[voltage_noise_channel].to_numpy()
        
            # ============================================================
            # AUTOMATIC IV SEGMENT DETECTION
            # ============================================================
            
            iv_counter, quadrant_counter = self._detect_iv_segments(
                self.I, 
                self.t,
                drop_factor=drop_factor
            )
            
            n_segments = np.sum(iv_counter == 0)
            
            if verbose:
                print(f"[Info] Detected {n_segments} IV segments")

            # Create Pnum array (within-segment counter that resets)
            self.Pnum_iv = iv_counter
            
            # Store segment IDs separately (for internal use if needed)
            self.Pnum_segment = quadrant_counter

            self._remove_voltage_offset(verbose=verbose)
            
            self.passed = True
            
            if verbose:
                print(f"[Success] Loaded PPMS IV data: {len(self.I)} points in {n_segments} IV sweeps.")
        
        elif measurement == 'Ic':
            # Extract crit current (REQUIRED)
            ic_channel = f'Crit.Cur. ch{str(channel)} (mA)'
            if ic_channel not in df.columns:
                print(f"[Error] Required column '{ic_channel}' not found")
                return
            self.Ic = df[ic_channel].to_numpy()/1e3
            
            # Extract crit current error (REQUIRED)
            ic_err_channel = f'C.Cur. Std.Dev. ch{str(channel)}'
            if ic_err_channel not in df.columns:
                print(f"[Error] Required column '{ic_err_channel}' not found")
                return
            self.Ic_err = df[ic_err_channel].to_numpy()/1e3
            
            self.fits = self._to_fit_format()
            
            self.passed = True
            
            if verbose:
                print(f"[Success] Loaded PPMS Ic data: {len(self.Ic)} critical currents recorded.")
        
        elif measurement == 'resistance':
            # Extract resistance (REQUIRED)
            r_channel = f'Res. ch{str(channel)} (ohm-cm)'
            if r_channel not in df.columns:
                print(f"[Error] Required column '{r_channel}' not found")
                return
            self.R = df[r_channel].to_numpy()
            
            # Extract resistance error (REQUIRED)
            r_err_channel = f'Res. Std.Dev. ch{str(channel)}'
            if r_err_channel not in df.columns:
                print(f"[Error] Required column '{r_err_channel}' not found")
                return
            self.R_err = df[r_err_channel].to_numpy()
            
            self.passed = True
            
            if verbose:
                print(f"[Success] Loaded PPMS resistance data: {len(self.R)} resistance values recorded.")


    def _to_fit_format(self):
        """
        Convert PPMS Ic measurements to ivtools fit results format.
        
        Creates DataFrame matching output structure of ivtools.fitting.fit_IV_for_Ic()
        for seamless integration with downstream analysis tools.
        
        Returns
        -------
        DataFrame
            Columns: File, Sample, Temperature [K], Avg Field [T], I_c, 
                    Orientation, Magnet, I_cH, I_c Error, IV_Index,
                    k, n, R², Avg dB/dt [T/s], n Error, fit_start_index,
                    fit_end_index, Data_Length
                    
            Note: k, n, R² are NaN (no power-law fit performed)
        """
        fits = pd.DataFrame({
            'File': Path(self.fp).name,
            'Sample': self.sample,
            'Temperature [K]': self.T,
            'Avg Field [T]': self.B, 
            'I_c': self.Ic,
            'Orientation': self.orientation,
            'Magnet': 'PPMS ACT',
            'I_cH': self.Ic * self.B,
            'I_c Error': self.Ic_err,
            'IV_Index': np.arange(len(self.Ic)),
            
            # Placeholder values for missing fit parameters
            'k': np.nan,
            'n': np.nan,
            'R²': np.nan,
            'Avg dB/dt [T/s]': 0,  # Static field
            'n Error': np.nan,
            'fit_start_index': np.nan,
            'fit_end_index': np.nan,
            'Data_Length': np.nan,
        })
        
        return fits
    
    def _detect_iv_segments(self, I, t=None, drop_factor=0.5, time_gap_threshold=2):
        """
        Detect IV segment boundaries based on current drops and time gaps.
        
        Uses two detection methods:
        1. Time gap > threshold (e.g., pause between measurements)
        2. Large current drop > drop_factor × current range
        
        Also tracks quadrant boundaries (current inflection points) for
        finer-grained analysis.
        
        Parameters
        ----------
        I : array-like
            Current values (A)
        t : array-like, optional
            Time stamps (s). If None, only current-drop detection used
        drop_factor : float, default=0.5
            Fraction of current range defining a reset
            (0.5 means 50% drop triggers new segment)
        time_gap_threshold : float, default=2
            Time gap (s) that forces new segment
            
        Returns
        -------
        iv_pnums : ndarray (int)
            Within-IV counter (0, 1, 2, ... resets at each new IV)
        quadrant_pnums : ndarray (int)
            Within-quadrant counter (resets at current inflection points)
            
        Examples
        --------
        For data: I = [0, 1, 2, 3, 0.1, 1.5, 2.5]  (drop at index 4)
        Returns:  iv_pnums = [0, 1, 2, 3, 0, 1, 2]
        """

        I = np.asarray(I)
        if t is not None:
            t = np.asarray(t)
        
        iv_pnums = np.zeros(len(I), dtype=int)
        quadrant_pnums = np.zeros(len(I), dtype=int)
        
        iv_counter = 0
        quadrant_counter = 0
        running_max = I[0]
        running_min = I[0]
        
        for i in range(len(I)):
            new_IV=False
            if i > 0:

                # ────────────────────────────────────────────────────────
                # 1. TIME GAP indicates new IV (if time available)
                # ────────────────────────────────────────────────────────
                if t is not None and i > 0:
                    dt = t[i] - t[i-1]
                    if dt > time_gap_threshold:
                        # Large time gap → force new IV
                        iv_counter = 0
                        quadrant_counter = 0
                        new_IV=True
                if not new_IV:
                    # ────────────────────────────────────────────────────────
                    # 3. CURRENT DROP indicates new IV
                    # ────────────────────────────────────────────────────────
                    abs_I = np.abs(I[i])
                    abs_I_prev = np.abs(I[i-1])
                    running_max = max(running_max, abs_I)
                    running_min = min(running_min, abs_I)
                    full_range = running_max - running_min
                    
                    # Detect a reset drop of more than 'drop_factor' of current span
                    if np.abs(abs_I-abs_I_prev) > drop_factor * full_range and iv_counter!=0:
                        # Start new segment
                        iv_counter = 0
                        quadrant_counter = 0 
                        
                        # Reset range tracking
                        running_max = abs_I
                        running_min = abs_I
                        new_IV=True

                    else:
                        # ────────────────────────────────────────────────────────
                        # 3. CURRENT INFLECTION indicates new quadrant
                        # ────────────────────────────────────────────────────────
                        if i>2:
                            dI_curr = np.sign(I[i]-I[i-1])
                            dI_prev = np.sign(I[i-1]-I[i-2])
                            if not dI_curr!=dI_prev and dI_curr != 0:
                                quadrant_counter = 0
                            else:
                                quadrant_counter+=1

                        iv_counter+=1
            
            # Assign values
            iv_pnums[i] = iv_counter
            quadrant_pnums[i] = quadrant_counter
            
        
        return iv_pnums, quadrant_pnums
    
    def _remove_voltage_offset(self, current_tolerance=5e-6, verbose=False):
        """
        Remove voltage offset from IV measurements by subtracting V(I=0).
        
        For each IV segment (where Pnum_iv resets to 0), finds the first point 
        where |I| < current_tolerance and subtracts that voltage from all points 
        in the segment. The offset is constant across all quadrants within an IV.
        
        Parameters
        ----------
        current_tolerance : float, default=5e-6
            Absolute current threshold (A) for considering a point as "zero current"
        verbose : bool, default=False
            Print diagnostic information per IV segment
            
        Returns
        -------
        None
            Modifies self.V in-place
            
        Raises
        ------
        AttributeError
            If called on non-IV measurement (missing V, I, or Pnum_iv)
            
        Notes
        -----
        - Offset is determined from the first |I| < tolerance point in each IV
        - If no such point exists, no correction is applied to that IV
        - Offset remains constant across quadrants (Pnum_segment changes ignored)
        - Call this AFTER __init__ completes and before analysis
        
        Examples
        --------
        >>> act = ACT_File('data.dat', 'SampleA', 'IV', channel=1)
        >>> act.remove_voltage_offset(current_tolerance=5e-7, verbose=True)
        [Info] IV 0 (idx 0): Offset = 2.31e-08 V (I = 3.2e-10 A)
        [Info] IV 1 (idx 523): Offset = -1.05e-08 V (I = 1.8e-09 A)
        """
        
        # Validation
        if not hasattr(self, 'V') or not hasattr(self, 'I') or not hasattr(self, 'Pnum_iv'):
            raise AttributeError(
                "remove_voltage_offset() only works for IV measurements "
                "(requires V, I, and Pnum_iv attributes)"
            )
        
        # Find IV segment boundaries (where Pnum_iv resets to 0)
        reset_indices = np.where(self.Pnum_iv == 0)[0]
        
        if len(reset_indices) == 0:
            if verbose:
                print("[Warning] No IV segments detected (Pnum_iv never equals 0)")
            return
        
        n_corrected = 0
        
        # Process each IV segment
        for iv_num, start_idx in enumerate(reset_indices):
            # Determine segment end (next reset or end of array)
            if iv_num < len(reset_indices) - 1:
                end_idx = reset_indices[iv_num + 1]
            else:
                end_idx = len(self.I)
            
            # Extract segment data
            I_segment = self.I[start_idx:end_idx]
            
            # Find first I≈0 point
            zero_mask = np.abs(I_segment) < current_tolerance
            
            if not np.any(zero_mask):
                if verbose:
                    print(f"[Warning] IV {iv_num} (idx {start_idx}): "
                        f"No |I| < {current_tolerance:.1e} A found, skipping")
                continue
            
            # Get offset from first zero-current point
            first_zero_rel = np.where(zero_mask)[0][0]
            offset_idx = start_idx + first_zero_rel
            offset_voltage = self.V[offset_idx]
            
            # Apply correction to entire IV segment
            self.V[start_idx:end_idx] -= offset_voltage
            n_corrected += 1
            
            if verbose:
                print(f"[Info] IV {iv_num} (idx {start_idx}): "
                    f"Offset = {offset_voltage:.3e} V "
                    f"(I = {self.I[offset_idx]:.3e} A at idx {offset_idx})")
        
        if verbose:
            print(f"[Success] Corrected {n_corrected}/{len(reset_indices)} IV segments")


    def _ensure_numeric(self, df, columns, verbose=False):
        """
        Convert string columns to numeric, handling PPMS parsing quirks.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame with potentially string-typed numeric columns
        columns : list of str
            Column names to convert
        verbose : bool
            Print conversion diagnostics
            
        Returns
        -------
        DataFrame
            DataFrame with converted columns
        """
        for col in columns:
            if col in df.columns:
                original_dtype = df[col].dtype
                
                # Convert to numeric, invalid values become NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                if verbose and original_dtype == 'object':
                    n_converted = df[col].notna().sum()
                    n_failed = df[col].isna().sum()
                    print(f"[Info] Converted '{col}': {n_converted} values, {n_failed} NaN")
        
        return df

