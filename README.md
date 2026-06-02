# IV Tools

**Version:** 0.5.0  
**Part of:** High Magnetic Field Science Toolset (LANL Copyright No. C20099)  
**Repository:** https://github.com/stef-ma/ivtools

Tools for processing nonlinear transport data for critical current measurements in pulsed magnetic fields. Designed for data acquired using the NHMFL LabActor Framework (based on National Instruments Actor Framework architecture).

---

## Features

- **TDMS File Parsing**: Automatic loading and validation of LabActor TDMS files
- **Power-Law Fitting**: Weighted least-squares fitting with uncertainty propagation
- **Flexible Processing Pipeline**: Customizable via hook system
- **Batch Processing**: Process multiple samples with experimental log integration
- **OriginLab Export**: Special CSV formatting for seamless import into Origin
- **Interactive Visualization**: Jupyter notebook with widget-based plotting

---

## Requirements

- **Python**: ≥3.9
- **Core Dependencies**:
  - `numpy` (≥1.19)
  - `pandas` (≥1.1)
  - `scipy` (≥1.5)
  - `statsmodels` (≥0.12)
  - `nptdms` (≥1.3)
- **Notebook Environment**:
  - `jupyter` (≥1.0)
  - `ipykernel` (≥6.0)
  - `ipywidgets` (≥7.6)
  - `matplotlib` (≥3.2)
  - `tqdm` (≥4.60)

---

## Installation

### From GitHub
```bash
pip install git+https://github.com/stef-ma/ivtools
```
### Editble Dev Install
```
git clone https://github.com/stef-ma/ivtools
cd ivtools
pip install -e .
```

## Example Workflow

```python
# Import IV Tools
import ivtools as ivt

# Load a single IV_File
ivf = ivt.IV_File(
   filepath='data/sample_001.tdms', 
   resistor=1,                                        # 1 Ω calibrated monitor resistor
   temperature=77,                                    # 77 K measurement temperature 
   voltage_gain=100,                                  # 100x Pre-Amp gain on sample signal channel
   current_gain=100,                                  # 100x Pre-Amp gain on curret monitor channel
   voltage_channel='V',                               # Sample signal channel name in TDMS 
   current_channel='I',                               # Current monitor channel name in TDMS
   ppms_field=tfield if magnet=='PPMS' else None      # None for pulsed field experiments (Field is read from TDMS), else fixed field value from PPMS experiments
   verbose=False
   )

# Process IV_File
ivs, fits, ivf = ivt.process_ivf(
                                 # required
                                 ivf,                             # Loaded ivf sample
                                 fp=str(ivf.path),                # Filepath to track data provenance. Already contained in ivf (redundant, can be altered. Metadata.
                                 sample='A',                      # Sample name. Metadata.
                                 temperature=ivf.T,               # 77 K measurement temperature (redundant, can be altered). Metadata.
                                 angle=0,                         # Fiel orientation (degrees) in respect to the sample plane. Metadata.
                                 tfield=10,                       # 10 T target field in pulsed field. Metadata.
                                 voltage_cutoff=1e-7,             # Arbitrary voltage criterion for Ic (V). Can be lower than noise flor, Ic and n are extrapolated.
                                 noise_level=1e-5,                # Voltage noise floor (V). Datapoints below this value are not used in fitting.
                                 linear_sub_criteron=0.95,        # R² threshold for linear background subtraction (0 most lenien, 1 most strict)
                                 power_law_criterion=0.95,        # R² threshold for accepting power law V(I) fit  (0 most lenien, 1 most strict)
                                 minfp=3,                         # Minimum fit points (default = 3)
                                 maxfp=30,                        # Maximum fit points (default = 30)
                                 # optional
                                 magnet='59T Mid Pulse',          # Magnet name used in experiment (default = 'Mid Pulse')
                                 verbose=False,                   # default = False
                                 lin_sub_level=0.5,               # |slope - 1| < lin_sub_level in log(V) vs log(I) is considered for background removal (default = 0.5)
                                 center_fraction=0.5,             # Fractional width of flat-top of square wave signal region in datastream used to convert V(t) into V(I,B). (default = 0.5)
                                 weight_power=1,                  # Exponent applied to weight mechanism to force fit to higher current datapoints. (1→natural)
                                 weight_mode='x'                  # Weight mechanism ('x'→current, 'index'→point index in IV). (default = 'x')
                                 hooks = None                     # Hooks object from hooks.py if desired. None defaults to default_hooks, which is empty unless altered manually. (default = None) 
                                 )


# Save processed data
ivt.save_ivdata(                 # Save IV data.
    ivs, 
    fname='A_001',               # File saved as f"{sample}_IV_{temperature}K_{tfield}T_{orientation}deg_{fname}"
    base_path='results/',
    sample='A',
    orientation=0,
    magnet='59T Mid Pulse',
    tfield=10,
    temperature=77,
    preset='full',               # Saving preset. Available 'full', 'minimal', 'minimal_expanded'. Custom sets can be defined as per iv_io.py. (defalt='full')
    origin=True                  # OriginLab-readable format (Auto-populated unit and comment fields)
)

ivt.save_fitdata(                # Save fit data.
    fits,
    fname='A_001',               # File saved as f"{sample}_IcH_{temperature}K_{tfield}T_{orientation}deg_{fname}"
    base_path='results/',
    sample='A',
    orientation=0,
    magnet='59T Mid Pulse',
    tfield=10,
    temperature=77,
    preset='minimal_expanded',
    origin=True
)
```

## Processing Pipeline
Hooks allow you to inject custom processing logic at ```[specific pipeline stages]``` without modifying the core code. Perfect for experiment-specific corrections, quality checks, or custom analyses.
```
Raw TDMS File
    ↓
Load IV_File instance
    ↓
[ivf_processing] ← Modify IV_File arrays (I, V, B)
    ↓
━━━ Per-Pulse Loop ━━━
    │
    │ Extract V(I,B) for one pulse
    │    ↓
    │ [conversion] ← Modify pulse result dict
    │    ↓
    │ Append to DataFrame
    │
━━━━━━━━━━━━━━━━━━━━
    ↓
Full DataFrame created
    ↓
[pre_segmentation] ← Operate on complete file DataFrame
    ↓
Split into IV segments by current jumps
    ↓
━━━ Per-Segment Loop ━━━
    │
    │ [post_segmentation] ← Modify segment DataFrame
    │    ↓
    │ Extract arrays (x, y)
    │    ↓
    │ [pre_linear_subtraction] ← Modify arrays
    │    ↓
    │ Linear background subtraction
    │    ↓
    │ [post_linear_subtraction] ← Modify arrays
    │    ↓
    │ Masking (monotonicity, noise)
    │    ↓
    │ Add anchor point
    │    ↓
    │ [post_masking_and_anchoring] ← Modify arrays
    │    ↓
    │ ━━━ Fitting Grid Search ━━━
    │     │
    │     │ For each (start, end) combination:
    │     │    ↓
    │     │ [pre_fitting] ← Modify fit window
    │     │    ↓
    │     │ Power law fit
    │     │    ↓
    │     │ [post_fitting] ← Modify fit results
    │     │
    │ ━━━━━━━━━━━━━━━━━━━━━━━━━
    │    ↓
    │ Select best fit
    │
━━━━━━━━━━━━━━━━━━━━━━
    ↓
All segments processed
    ↓
[results] ← Operate on file-level results lists
    ↓
Return: ivs, fits DataFrames
```

### Example: Apply Calibration Correction
```python
from ivtools.hooks import ProcessingHooks

# Create custom hook registry
my_hooks = ProcessingHooks()

# Define correction function
def apply_field_correction(ivf, **kwargs):
    """Correct field calibration error."""
    ivf.B = ivf.B * 1.05  # 5% correction
    return ivf

# Register hook at the appropriate stage
my_hooks.register('ivf_processing', apply_field_correction)

# Use custom hooks in processing
ivs, fits, ivf = ivt.process_ivf(
    ivf, fp, sample, temperature, angle, tfield,
    voltage_cutoff, noise_level, linear_sub_criterion,
    power_law_criterion, minfp, maxfp,
    hooks=my_hooks  # ← Pass custom hooks
)
```

### Available Hook Points

| Hook Stage | When Applied | Common Uses | Input/Output Type | Key Variables Available |
|-----------|--------------|-------------|-------------------|------------------------|
| `ivf_processing` | After TDMS load, before any processing | Instrument corrections, gain adjustments | `IV_File` object | `ivf.B`, `ivf.T`, `ivf.I`, `ivf.V` (arrays), context: `fp`, `sample`, `temperature`, `angle`, `tfield`, `magnet` |
| `conversion` | Per V(I,B) datapoint during TDMS→DataFrame conversion | Pulse-level calibrations, voltage offsets | `dict` | `result` (dict with 'Current [A]', 'Voltage [V]', etc.), `ivf`, `top`, `left`, `right`, `I`, `V`, `B`, `dBdt`, `T` |
| `pre_segmentation` | Before splitting DataFrame into IV segments | File-level filtering, field corrections | `DataFrame` | `df` (full file with columns: 'Current [A]', 'Voltage [V]', 'Field [T]', etc.), `voltage_cutoff` |
| `post_segmentation` | After splitting, per IV segment | Segment-level QA, outlier removal | `DataFrame` | `segment` (one IV with same columns as df), `voltage_cutoff`, `segment_index`, `lin_sub_level`, `linear_sub_criterion` |
| `pre_linear_subtraction` | After extracting arrays, before background removal | Contact resistance corrections | `dict` | `x` (current array), `y` (voltage array), `segment_index`, `field_avg`, `dBdt_avg`, `voltage_cutoff`, `segment` (DataFrame) |
| `post_linear_subtraction` | After background removal, before masking | Custom filtering, offset adjustments | `dict` | Same as `pre_linear_subtraction` |
| `post_masking_and_anchoring` | After noise filtering and anchor point addition | Quality checks, additional filtering | `dict` | Same as `pre_linear_subtraction` |
| `pre_fitting` | Just before each fit attempt in grid search | Fit window adjustment, custom weighting | `dict` | `x_fit`, `y_fit`, `start`, `end`, `x`, `y`, `segment_index`, `field_avg`, `dBdt_avg`, `voltage_cutoff`, `segment`, `weight_power`, `weight_mode` |
| `post_fitting` | After each fit attempt (many times per segment) | Per-attempt result filtering | `dict` | `k`, `n`, `sigma_ic`, `sigma_n`, `r2`, `x_fit`, `y_fit`, `start`, `end`, `x`, `y`, `segment_index`, `field_avg`, `dBdt_avg`, `voltage_cutoff`, `segment`, `weight_power`, `weight_mode` |
| `results` | After all segments processed (once per file) | File-level result corrections, batch filtering | `dict` | `fit_successes`, `I_cs`, `ks`, `bs`, `r2s`, `segments`, `segments_power`, `processed_segments`, `best_starts`, `best_ends`, `H_avgs`, `dBdt_avgs`, `I_cHs`, `dlen`, `sigmas_ic`, `sigmas_n` (all lists) |


## Algorithm Details
### Critical Current Determination
      Method: Weighted least-squares power-law fit in log-log space
      Model: V/Vc = (I/Ic)^n → log(V) - log(Vc) = n·log(I) - n·log(Ic)
      Weights: w = I^p or w = index^p (configurable)
      Ic Calculation: Ic = (Vc/k)^(1/n) where Vc is the voltage criterion
      Uncertainty: Propagated from covariance matrix using first-order Taylor expansion

### Background Subtraction
    Detection: Identifies linear regime via log-log slope analysis
    Criterion: Points with |slope - 1| < cutoff in log(V) vs log(I)
    Validation: Only subtracts if linear fit R² exceeds threshold

### Data Filtering

    Monotonicity: Non-monotonic voltage points replaced with NaN
    Noise Rejection: Points below noise_level removed before fitting
    Anchoring: Synthetic (I→0, V→0) point added for numerical stability

## File Structure
```
README.md
ivtools_processing.ipynb
ivtools/
├── __init__.py          # Package initialization
├── iv_io.py             # IV_File class, save functions
├── process.py           # Main processing pipeline
├── fitting.py           # fit_IV_for_Ic() implementation
├── fit_utils.py         # Utility functions (WLS, masking, etc.)
└── hooks.py             # Hook system (ProcessingHooks class)
```
## License
LANL Copyright No. C20099
See parent repository for full license details: https://github.com/ffb-LANL/High-Magnetic-Field-Science-Toolset

## Citation
If you use this software in published research, please cite
[Add your preferred citation format here]


## Common Issues

### "File missing channels" error
Ensure TDMS file contains: voltage, current, Pnum, Vavg, Field (or provide ppms_field= value).
Check channel names match your LabActor configuration

### No fits found
Check voltage_cutoff is appropriate for your data.
Verify noise_level isn't excluding valid data.
Try relaxing power_law_criterion (e.g., 0.90 instead of 0.95).

### Negative or unrealistic Ic values
Increase linear_sub_criterion to disable aggressive background subtraction.
Check voltage gain/current gain settings.
Verify calibrated resistor value.

### Slow processing
Reduce max_fit_points (default: 5).
Process fewer samples per batch.
Disable verbose=True for batch jobs.

# IV Tools Processing Notebook

This notebook (`ivtools_processing.ipynb`) provides a structured, interactive workflow for exploring, processing, and visualizing IV datasets processed with IV Tools.  
It is designed to work with IV files acquired via the LabActor framework and stored in TDMS or CSV formats.
### Core Processing
- Automated segmentation of IV curves from time-stream data
- Background-corrected voltage–current characteristics
- Power-law fits ($V = k \cdot I^n$) and critical current extraction ($I_c$ at arbitrary $V_c$)
- Batch processing across multiple files and experiments from .xlsx logs.

### Data Discovery & Management
- **Automated data crawling**: Recursive directory search for processed CSV files
- **Metadata parsing**: Extracts sample, temperature, field, orientation from filenames
- **Intelligent filtering**: Multi-parameter dataset selection and grouping
- **Dynamic data refresh**: Load new datasets without rerunning analysis cells

### Interactive Visualization & Analysis
- **Multi-axis plotting**: Simultaneous $I_c(H)$ and $n(H)$ visualization with click-to-view IV curves
- **$I_c$ vs $n$-value scatter analysis**: Correlation studies with flexible color-coding
- **Power law fitting**: Interactive $I_c = A \cdot H^{-\alpha}$ fitting with quality control
- **Real-time parameter adjustment**: Sliders, toggles, and filters for exploratory analysis

### Output Capabilities
- Processed data saved as CSV with full metadata
- Exportable matplotlib figures for publication
- Global variables for downstream scripting and automation

---

## Complete Workflow

### Single-File Processing (Sections 1-3)
1. Load a TDMS file into an `IV_File` object
2. Automatically segment the time stream into individual IV sweeps
3. Correct linear backgrounds and suppress noise
4. Fit non-linear regions using power-law models
5. Save processed IV data and fit results to disk

### Batch Processing (Section 4)
1. Define root directory and sweep parameters
2. Discover all TDMS files matching criteria
3. Process multiple files in sequence with progress tracking
4. Generate summary reports and quality metrics
5. Export organized CSV datasets with consistent structure

### Interactive Analysis (Sections 5-7)
1. **Data Crawler**: Load and filter processed CSV files with multi-parameter selection
2. **Dual-axis plotter**: Visualize $I_c(H)$ and $n(H)$ with click-to-examine IV curves
3. **$I_c-n$ scatter plot**: Explore correlations with flexible grouping and color-coding
4. **$I_c(H)$ power law fitter**: Interactive field-dependence fitting with range selection

### Advanced Scripting (Section 8)
- Custom analysis using exposed data structures
- Programmatic access to filtered datasets
- Automated reporting and figure generation

---

## Data Assumptions

This notebook assumes:
- Data are acquired using the **MAGLAB LabActor Framework**
- Raw data are stored as **TDMS** files
- Current and voltage are measured through calibrated gain chains
- Magnetic field information is provided either by:
  - PPMS metadata, or
  - an external pulsed-field trace
- Processed data follow naming convention: `*_IcH_*.csv` and `*_IV_*.csv`

No manual IV segmentation is required.

## Usage Notes

- Ensure that the IV data files are in the expected directory structure and format.
- The notebook uses widgets for interactive filtering and plotting; it works in Jupyter Notebook or JupyterLab. VSCode may duplicate outputs.
- For batch processing, provide the experimental log (Excel) with sample, temperature, field, and orientation and other metadata to automate dataset identification.
- All processed data can be saved and reused in subsequent analyses or shared with collaborators.

## Related Tools

    LabActor Framework: https://www.nationalmaglab.org/user-facilities/pulsed-field-facility/
    MultiPyVu: https://github.com/AdrienGourgout/MultiPyVu (for PPMS data integration)



## Contributing
This package is part of the High Magnetic Field Science Toolset developed at LANL. For issues or contributions, please contact the repository maintainer.

## Version
```
Maintainer: Stefan Marinkovic
Status: Beta (v0.5.0)
```


