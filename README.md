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