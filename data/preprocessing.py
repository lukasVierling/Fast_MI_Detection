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

from .utils import remove_baseline_wander, notch_filter, split_signal, preprocess_ECG

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

def split_patients(records, train_ratio = 0.6, val_ratio = 0.1, seed=0, k_fold=1):

    ids = sorted({os.path.basename(os.path.dirname(r)) for r in records})

    #get random shuffle of indices
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    if k_fold == 1:
        n_total = len(ids)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))

        train_ids = ids[:n_train]
        val_ids = ids[n_train:n_train+n_val]
        test_ids = ids[n_train+n_val : ]

        return train_ids, val_ids, test_ids
    else:
        #return k_fold train test
        if val_ratio!=0:
            print("k_fold cv has no validation set")
        train_folds = []
        test_folds = []
        n_total = len(ids)
        n_test = n_total//k_fold # if k_fold 5 then n/5 is for test

        for k in range(k_fold):
            start = n_test * k
            end = n_test * (k+1) if k < k_fold-1 else n_total
            train_folds.append(ids[:start]+ids[end:])
            test_folds.append(ids[start:end])
        return train_folds, test_folds


def get_dataloaders(path, train_ratio, val_ratio, train_ids=None, test_ids=None, val_ids=None, batch_size=256, preprocessed_data_path=None, desired_leads=['i','ii','iii','avr','avl','avf','v1','v2','v3','v4','v5','v6'], seed=0):
    records = get_record_paths(path)
    filtered_records = filter_records(records)
    #allow to hand over predefined subsets of the data
    if train_ids is None or test_ids is None or val_ids is None:
        train_ids, val_ids, test_ids = split_patients(filtered_records, train_ratio, val_ratio, seed)
        print(f"Patients: train: {len(train_ids)} | val: {len(val_ids)} | test: {len(test_ids)}")
    
    if preprocessed_data_path is None:
        print("No data path given -> create dataset")
        ECG_data, labels, ids = get_dataset(filtered_records, overlap=0.5, desired_leads=desired_leads)
        ECG_data = torch.tensor(ECG_data.transpose(0,2,1), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        #save data
        save_path = "/content/drive/MyDrive/ptbdb/preprocessed_data.pt"
        torch.save({"ECG_data":ECG_data,"labels":labels,"ids": ids}, save_path)
        print("Data saved at: ", save_path)
    else:
        print("Load data from given path")
        preproc = torch.load(preprocessed_data_path, weights_only=False)
        ECG_data, labels, ids = preproc["ECG_data"], preproc["labels"], preproc["ids"]
    
    #get the dataloaders
    train_mask = np.isin(ids, list(train_ids))
    train_loader = DataLoader(TensorDataset(ECG_data[train_mask], labels[train_mask]),shuffle=True,batch_size=batch_size)

    val_mask = np.isin(ids, list(val_ids))
    val_loader = DataLoader(TensorDataset(ECG_data[val_mask], labels[val_mask]),batch_size=batch_size)

    test_mask = np.isin(ids, list(test_ids))
    test_loader = DataLoader(TensorDataset(ECG_data[test_mask], labels[test_mask]),batch_size=batch_size)

    return train_loader, val_loader, test_loader