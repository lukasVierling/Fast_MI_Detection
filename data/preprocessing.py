import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import wfdb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import iirnotch, filtfilt
from scipy.interpolate import CubicSpline
import pywt

from utils import remove_baseline_wander, notch_filter, split_signal, preprocess_ECG

def get_record_paths(path):
    records = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(".dat"):
                records.append(os.path.join(root, f[:-4]))
    return sorted(records)

def filter_records(records):
    healthy = []
    mi_to_id = {}
    healthy_label = "Healthy control"
    disease_label = "Myocardial infarction"
    for record in records:
        header = wfdb.rdheader(record)
        comment = header.comments
        id = os.path.basename(os.path.dirname(record))
        label = next((line.split(":", 1)[1].strip() for line in comment if line.startswith("Reason for admission:")), None)
        if label== healthy_label:
            healthy.append(record)
        elif label == disease_label:
            if id in mi_to_id:
                mi_to_id[id].append(record)
            else:
                mi_to_id[id] = [record]
    #filter only the ealriest entry
    disease = [sorted(list)[0] for list in mi_to_id.values()]
    filtered_records= sorted(healthy+disease)
    print(f"After filtering, we got: {len(filtered_records)} records. Healthy: {len(healthy)}, Disease: {len(disease)}")
    return filtered_records

    
def get_dataset(records, overlap=0.5, desired_leads=['i','ii','iii','avr','avl','avf','v1','v2','v3','v4','v5','v6']):
    ECG_data, labels, ids = [],[],[]
    for record in records:
        ecg, meta_data = wfdb.rdsamp(record)
        leads = [lead.strip().lower() for lead in meta_data["sig_name"]]
        #usually all leads but for ablation
        idxs = [leads.index(l) for l in desired_leads]
        preprocessed_data = preprocess_ECG(ecg[:, idxs])
        id = os.path.basename(os.path.dirname(record))
        label = int(any("Myocardial infarction" in line for line in meta_data["comments"]))
        for window in split_signal(preprocessed_data, overlap=overlap):
            ECG_data.append(window)
            labels.append(label)
            ids.append(id)
    return np.stack(ECG_data), np.array(labels), np.array(ids)

def split_patients(records, train_ratio = 0.6, val_ratio = 0.1, seed=0):

    ids = sorted({os.path.basename(os.path.dirname(r)) for r in records})

    #get random shuffle of indices
    range = np.random.default_rng(seed)
    range.shuffle(ids)

    n_total = len(ids)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train+n_val]
    test_ids = ids[n_train+n_val : ]

    split = {"train":[], "val": [], "test": []}
    for record in records:
        id = os.path.basename(os.path.dirname(record))
        if id in train_ids:
            split["train"].append(record)
        elif id in val_ids:
            split["val"].append(record)
        else:
            split["test"].append(record)

    return split, train_ids, val_ids, test_ids

def get_dataloaders(path, preprocessed_data_path, train_ratio, val_ratio, desired_leads, seed):
    records = get_record_paths(path)
    filtered_records = filter_records(records)
    splits, train_ids, val_ids, test_ids = split_patients(filtered_records, train_ratio, val_ratio, seed)
    print(f"Patients: train: {len(train_ids)} | val: {len(val_ids)} | test: {len(test_ids)}")
    if preprocessed_data_path is None:
        print("No data path given -> create dataset")
        ECG_data, labels, ids = get_dataset(filtered_records, overlap=0.5, desired_leads=desired_leads)
        ECG_data = torch.tensor(ECG_data.transpose(0,2,1), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    else:
        print("Load data from given path")
        ECG_data, labels, ids = torch.load(preprocessed_data_path, weights_only=False)

    #get the dataloaders
    train_mask = np.isin(ids, list(train_ids))
    train_loader = DataLoader(TensorDataset(ECG_data[train_mask], labels[train_mask]))

    val_mask = np.isin(ids, list(val_ids))
    val_loader = DataLoader(TensorDataset(ECG_data[val_mask], labels[val_mask]))

    test_mask = np.isin(ids, list(test_ids))
    test_loader = DataLoader(TensorDataset(ECG_data[test_mask], labels[test_mask]))

    return train_loader, val_loader, test_loader
