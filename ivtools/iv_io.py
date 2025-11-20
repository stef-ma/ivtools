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
        self.path = Path(filepath)
        self.current_chan = current_channel
        # Check for required channels
        chans = []
        for chan in self.tdms_file['p'].channels():
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
                self.B, _, _, _=self._load_channel_data('Field') #TODO: Should be Field_fixed?
                # self.B, _, _, _=self._load_channel_data('Field_fixed')
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
        group = self.tdms_file['p'] # Our TDMS files have only one group.
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
        if channel.name != 'Field':
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


    