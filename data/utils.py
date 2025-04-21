import os
import numpy as np
import wfdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import iirnotch, filtfilt
from scipy.interpolate import CubicSpline

SAMPLING_RATE=1000 #sampling rate of the dataset hardcoded

#https://www.geeksforgeeks.org/design-an-iir-notch-filter-to-denoise-signal-using-python/ for notch filter documentation
'''
In this study, as
shown in Fig. 2, the residual power-line interference and its
harmonicsaresuppressedbyapplyinga60-Hznotch filter with
a3-dB bandwidth of 5Hz,also known asb and-rejection filter [from the paper]
'''
def notch_filter(ecg, freq=60.0, bw=5.0):
    #get normalized frequency (nyquist)
    norm_freq = freq / ( SAMPLING_RATE / 2)
    quality_factor = freq/bw
    b,a = iirnotch(norm_freq,quality_factor)
    #apply the filter
    return filtfilt(b,a,ecg)

'''
In this technique, a cubic polynomial is fitted to a set of representative points of the original ECG signla ... the fitted baseline
is then subtracted from the ECG signal to remove this artifact. [from the paper]
'''
def remove_baseline_wander(ecg):
    time_steps = np.arange(len(ecg)) / SAMPLING_RATE
    #get the knot points for spline
    knot_points = np.arange(0, np.floor(time_steps[-1])+1)
    idxs = (knot_points*SAMPLING_RATE).astype(int)
    idx = np.clip(idxs,0,len(ecg)-1)
    #fit the cubic spline as baseline
    baseline = CubicSpline(knot_points, ecg[idx])(time_steps)
    #subtract the baseline so we have the detrended signal
    return ecg - baseline

def split_signal(data, window_size=5, overlap=0.5):
    #window size is in seconds but we adjust to sampling rate per second
    samples_per_window = int(window_size * SAMPLING_RATE)
    step_size = int(samples_per_window*(1-overlap))
    #segment the ECG
    windows = [data[i:i+samples_per_window] for i in range(0,len(data) - samples_per_window + 1, step_size)]
    return windows

def preprocess_ECG(ECG):
    preprocessed_leads = []
    num_samples = ECG.shape[1]
    for i in range(num_samples):
        lead = ECG[:,i]
        filtered = notch_filter(lead)
        detrended = remove_baseline_wander(filtered)
        preprocessed_leads.append(detrended)
    return np.stack(preprocessed_leads, axis=1)
