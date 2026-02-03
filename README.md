# IV Tools

IV Tools is a part of the  High Magnetic Field Science Toolset (LANL Copyright No. C20099)

https://github.com/ffb-LANL/High-Magnetic-Field-Science-Toolset

Tools for processing Non-Linear Transport data for critical current measurements in pulsed field, acquired using the MAGLAB LabActor Framework.

LabActor framework is based on National Instruments Actor Framework architecture

# External libraries dependencies by module

requires python version >=3.9

dependencies:
    numpy
    pandas
    matplotlib
    scipy
    nptdms
    tqdm
    MultiPyVu
    statsmodels

# Installation

pip install git+https://github.com/stef-ma/ivtools

# Use

#import module

import ivtools as ivt


#load tdms file as IV_File object

ivf = ivt.IV_File(
    filepath, 
    resistor, 
    temperature, 
    voltage_gain, 
    current_gain, 
    voltage_channel=vchan,
    current_channel=ichan,
    ppms_field=tfield if magnet=='PPMS' else None
    )


#process IV_File, outputs the processed IVs, the fits, and the ivf itself

ivs, fits, ivf = ivt.process_ivf(
                                ivf,
                                filepath,
                                sample,
                                temperature,
                                angle,
                                tfield,
                                voltage_criterion,
                                noise_level,
                                linear_sub_criterion,
                                power_law_criterion,
                                minfp,
                                maxfp,
                                magnet = magnet,
                                verbose = verbose
                                )

                                
#save data

ivt.save_ivdata(
                ivs,
                fname,
                savepath,
                sample,
                angle,
                magnet,
                tfield,
                temperature,
                preset=save_preset,
                verbose=verbose
            )
            
ivt.save_fitdata(
                fits,
                fname,
                savepath,
                sample,
                angle,
                magnet,
                tfield,
                temperature,
                preset=save_preset,
                verbose=verbose
            )

# IV Tools Processing Notebook

This notebook (`ivtools_processing.ipynb`) provides a structured, interactive workflow for exploring, processing, and visualizing IV datasets processed with IV Tools.  
It is designed to work with IV files acquired via the LabActor framework and stored in TDMS or CSV formats.

## Notebook Overview

The notebook contains the following workflow:

1. **Import a Single IV File**  
   Load a single IV_File object from disk into the notebook. This allows inspecting and processing a specific measurement before batch processing.

2. **Process the Imported IV File**  
   Apply standard processing routines to the loaded IV_File:
   - Clean and standardize the IV data
   - Compute derived quantities (e.g., critical current `Ic`)
   - Fit models (e.g., power law) to the IV curve
   - Store results for visualization or export

3. **Save IVs and Fits**  
   Export processed IV curves and their corresponding fit results to disk.  
   Supports CSV or other tabular formats, including metadata for later reference or reporting.

4. **Batch Processing Across All Samples**  
   Automate processing for multiple samples in a directory:
   - Reads an experimental log (Excel `.xlsx`) to identify samples and measurement metadata
   - Scans the associated data directory for matching IV files
   - Processes each file using the standard pipeline
   - Aggregates and stores results in summary tables for easy comparison across samples

5. **Interactive Plotting of CSV Results**  
   Crawl a specified data directory for CSV results (e.g., `*_IcH_*.csv`) and provide interactive plotting:
   - Filters by temperature, orientation, or sample
   - Adjust axis limits, toggle logarithmic scales
   - Visualize multiple datasets simultaneously with distinct markers, colors, and edge styles

## Usage Notes

- Ensure that the IV data files are in the expected directory structure and format.
- The notebook uses widgets for interactive filtering and plotting; it works best in Jupyter Notebook or JupyterLab.
- For batch processing, provide the experimental log (Excel) with sample, temperature, field, and orientation metadata to automate dataset identification.
- All processed data can be saved and reused in subsequent analyses or shared with collaborators.

## Requirements

- Python >=3.9
- IV Tools installed (`pip install git+https://github.com/stef-ma/ivtools`)
- Dependencies: `numpy`, `pandas`, `matplotlib`, `scipy`, `nptdms`, `tqdm`, `MultiPyVu`, `statsmodels`

## Example Workflow

```python
# Import IV Tools
import ivtools as ivt

# Load a single IV_File
ivf = ivt.IV_File(filepath, resistor, temperature, voltage_gain, current_gain,
                   voltage_channel=vchan, current_channel=ichan,
                   ppms_field=tfield if magnet=='PPMS' else None)

# Process IV_File
ivs, fits, ivf = ivt.process_ivf(ivf, filepath, sample, temperature, angle, tfield,
                                  voltage_criterion, noise_level, linear_sub_criterion,
                                  power_law_criterion, minfp, maxfp, magnet=magnet, verbose=True)

# Save processed data
ivt.save_ivdata(ivs, fname, savepath, sample, angle, magnet, tfield, temperature, preset=save_preset)
ivt.save_fitdata(fits, fname, savepath, sample, angle, magnet, tfield, temperature, preset=save_preset)
