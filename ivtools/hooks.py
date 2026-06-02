# ivtools/hooks.py
"""
Processing hook system for custom IV data transformations.

Hooks are called at specific points in the processing pipeline,
allowing experiment-specific modifications without changing core code.

Hooks:

Raw IV Data
    ↓
[ivf_processing] ← Operate on full tdms datastream in the form of an IV_File instance (gains are accounted for and current is converted) (vars: *ivf*, fp, sample, temperature, angle, tfield, magnet)
    ↓
Convert TDMS datastream into V(I,B) DataFrame with metadata
    ↓
[conversion] ← Operate on full DataFrame of V(I,B) datapoints (vars: *df*, voltage_cutoff; output: df)
    ↓
[pre_segmentation] ← Operate on full DataFrame of V(I,B) datapoints (vars: *df*, voltage_cutoff; output: df)
    ↓
Split into IV segments  
    ↓
For each segment:
    ↓
    [post_segmentation] ← Operate on segment DataFrame of V(I,B) datapoints (vars: *segment*, voltage_cutoff, segment_index, lin_sub_level, linear_sub_criterion)
    ↓
    Extract to arrays (x, y)
    ↓
    [pre_linear_subtraction] ← Operate on individual IV data arrays in dict form before lin_sub (vars: *hook_data* (dict) [x,y,segment_index,field_avg,dBdt_avg,voltage_cutoff,segment])
    ↓
    Linear background subtraction
    ↓
    [post_linear_subtraction] ← Operate on individual IV data arrays in dict form after lin_sub (vars: *hook_data* (dict) [x,y,segment_index,field_avg,dBdt_avg,voltage_cutoff,segment])
    ↓
    Masking (monotonicity, noise)
    ↓
    Anchoring
    ↓
    [post_masking_and_anchoring] ← Operate on individual IV data arrays in dict form after masking and anchor_low_voltage (vars: *hook_data* (dict) [x,y,segment_index,field_avg,dBdt_avg,voltage_cutoff,segment])
    ↓
    Fitting loop:
        [pre_fitting] ← Operate on narrow data arrays during fitting attempts in dict form right before fitting logic (vars: *hook_data* (dict) [x_fit,y_fit,start,end,x,y,segment_index,field_avg,dBdt_avg,voltage_cutoff,segment,weight_power,weight_mode])
        Power law fit
        [post_fitting] ← Operate on fit results (vars: *hook_data* (dict) [k,n,sigma_ic,sigma_n,r2,x_fit,y_fit,start,end,x,y,segment_index,field_avg,dBdt_avg,voltage_cutoff,segment,weight_power,weight_mode])
    ↓
    [results] ← Operate on fit results (vars: *hook_data* (dict) [k,n,sigma_ic,sigma_n,r2,x_fit,y_fit,start,end,x,y,segment_index,field_avg,dBdt_avg,voltage_cutoff,segment,weight_power,weight_mode])

Output: ivs, fits DataFrames

"""

from typing import Callable, Dict, List, Optional
import pandas as pd

class ProcessingHooks:
    """
    Central registry for processing hooks.
    
    Example usage:
    --------------
    >>> hooks = ProcessingHooks()
    >>> 
    >>> def my_correction(segment, **kwargs):
    ...     segment['Current [A]'] *= 0.95  # Apply correction
    ...     return segment
    >>> 
    >>> hooks.register('post_linear_subtraction', my_correction)
    """
    
    def __init__(self):
        self._hooks: Dict[str, List[Callable]] = {
            'ivf_processing': [],          # Before converting to V(I,B) data
            'conversion': [],          # Before converting to V(I,B) data
            'pre_segmentation': [],        # Before splitting by jump
            'post_segmentation': [],       # After splitting, per segment
            'pre_linear_subtraction': [],  # Before lin_sub, after unpacking
            'post_linear_subtraction': [], # After lin_sub, before masking
            'post_masking_and_anchoring': [],            # After masking and anchoring
            'pre_fitting': [],             # Just before power law fit
            'post_fitting': [],            # After successful fit
            'results': [],                 # File level results operations
        }
    
    def register(self, stage: str, func: Callable, priority: int = 50):
        """
        Register a hook function at a specific processing stage.
        
        Parameters
        ----------
        stage : str
            Processing stage name (see available stages in __init__)
        func : callable
            Function with signature: func(data, **context) -> data
            - For segment hooks: data is a DataFrame
            - For fit hooks: data is a dict of fit results
        priority : int
            Execution order (lower = earlier). Default: 50
        """
        if stage not in self._hooks:
            raise ValueError(
                f"Unknown stage '{stage}'. Available: {list(self._hooks.keys())}"
            )
        
        self._hooks[stage].append((priority, func))
        # Keep sorted by priority
        self._hooks[stage].sort(key=lambda x: x[0])
    
    def execute(self, stage: str, data, **context):
        """
        Execute all registered hooks for a stage.

        IMPORTANT: Hooks should return a MODIFIED COPY of data,
        not modify it in-place, to avoid reference contamination.
        
        Parameters
        ----------
        stage : str
            Processing stage
        data : DataFrame or dict
            Data to transform
        **context : dict
            Additional context (e.g., voltage_cutoff, segment_index, etc.)
        
        Returns
        -------
        data : DataFrame or dict
            Transformed data
        """
        if stage not in self._hooks:
            return data
        
        for priority, func in self._hooks[stage]:
            try:
                data = func(data, **context)
            except Exception as e:
                print(f"[Warning] Hook failed at stage '{stage}': {e}")
        
        return data
    
    def clear(self, stage: Optional[str] = None):
        """Clear hooks for a specific stage or all stages."""
        if stage is None:
            for s in self._hooks:
                self._hooks[s] = []
        else:
            self._hooks[stage] = []

    def has_hook(self, stage: str) -> bool:
        """
        Check if any hooks are registered for a stage.
        
        Parameters
        ----------
        stage : str
            Processing stage name
        
        Returns
        -------
        bool
            True if at least one hook is registered
        """
        return stage in self._hooks and len(self._hooks[stage]) > 0


# Global default instance (for convenience)
default_hooks = ProcessingHooks()