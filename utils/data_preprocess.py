import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import KMeans, MiniBatchKMeans
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Data Load
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)  
        if df.empty: 
            print(f"Empty file: {file_path}")
            return [], []
        data = df.iloc[:, :-1].squeeze(axis=1) 
        labels = df.iloc[:, -1] 
       
        return data, labels
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return [], []


def load_and_split_data(file_path):

    dataset_type = file_path.split('/')[-2]

    try:
        data, labels = load_data(file_path)
        if len(data) == 0 or len(labels) == 0:  
            print(f"Empty data or labels for file: {file_path}")
            return [], [], [], []

    
        file_name_parts = file_path.split('/')[-1].split('_')
        train_end_part = [file_name_parts[i + 1] for i, part in enumerate(file_name_parts) if part == 'tr']
       

        if train_end_part and len(train_end_part[0]) > 1:
            train_end = int(train_end_part[0])
            train_data = data[:train_end]
            train_labels = labels[:train_end]
            test_data = data[train_end:]
            test_labels = labels[train_end:]
        else:
            print(f"Invalid file format or missing 'tr_' in: {file_path}")
            return [], [], [], []

        return train_data, train_labels, test_data, test_labels
    except IsADirectoryError:
        print(f"Skipped directory: {file_path}")
        return None, None, None, None
    except ValueError as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None, None, None


def preprocess_to_patches(data, patch_size, stride):
    patches = []
    for i in range(0, len(data) - patch_size + 1, stride):
        patch = data[i:i + patch_size]
        patches.append(patch)
    
    patches_array = np.array(patches)                      # (N, L) or (N, L, C)
    t = torch.tensor(patches_array, dtype=torch.float32)
    if t.ndim == 2:                   # (N, L) -> (N, 1, L)
        t = t.unsqueeze(1).contiguous()
    elif t.ndim == 3:                 # (N, L, C) -> (N, C, L)
        t = t.permute(0, 2, 1).contiguous()

    return t    


class _tsdataset(Dataset):
    def __init__(self, data, indices=None):
        # indice means relative order among patches 
        self.data = torch.from_numpy(np.array(data)).float()
        if indices is not None:
            self.indices = torch.from_numpy(np.array(indices)).long().unsqueeze(1)
        else:
            self.indices = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
            # x: (L,) or (L, C) or (C, L)
        if x.ndim == 1:          # (L,) -> (1, L)
            x = x.unsqueeze(0).contiguous()
       
        if self.indices is not None:
            return x, self.indices[idx]
        return x, torch.tensor([idx])

class PatchCreator:
    def __init__(self, L, s, random_seed=None):
        self.L = L 
        self.s = s  
        if random_seed is not None:
            torch.manual_seed(random_seed)  

    def create_patches(self, data):
        if not isinstance(data, (list, np.ndarray)):
            raise ValueError("Data must be a list or numpy array.")
        if len(data) < self.L:
            raise ValueError(f"Data length {len(data)} is less than patch size {self.L}.")
    
        num_patches = (len(data) - self.L) // self.s + 1
        patches = [data[i:i+self.L] for i in range(0, len(data) - self.L + 1, self.s)]
        indices = [i for i in range(0, len(data) - self.L + 1, self.s)]
        return patches, indices

    def create_dataloaders(self, train_data, test_data, test_labels, batch_size=512):
        
        train_patches, train_indices = self.create_patches(train_data)
        test_patches, test_indices = self.create_patches(test_data)

        
        train_patches = [p.T for p in train_patches] if train_patches[0].ndim == 2 else train_patches
        test_patches  = [p.T for p in test_patches]  if test_patches[0].ndim == 2 else test_patches

    
        train_loader = DataLoader(_tsdataset(train_patches, indices=train_indices), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(_tsdataset(test_patches, indices=test_indices), batch_size=batch_size, shuffle=False)

        true_test_labels = test_labels

        return train_loader, test_loader, true_test_labels



# cited from https://github.com/TheDatumOrg/TSB-AD/blob/main/TSB_AD/utils/slidingWindows.py
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf

# determine sliding window (period) based on ACF
def find_length_rank(data, rank=1):
    data = data.squeeze()
    if len(data.shape)>1: return 100 #0->100
    if rank==0: return 1
    data = data[:min(20000, len(data))]
    
    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]
    
    # plot_acf(data, lags=400, fft=True)
    # plt.xlabel('Lags')
    # plt.ylabel('Autocorrelation')
    # plt.title('Autocorrelation Function (ACF)')
    # plt.savefig('/data/liuqinghua/code/ts/TSAD-AutoML/AutoAD_Solution/candidate_pool/cd_diagram/ts_acf.png')

    local_max = argrelextrema(auto_corr, np.greater)[0]

    # print('auto_corr: ', auto_corr)
    # print('local_max: ', local_max)

    try:
        # max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        sorted_local_max = np.argsort([auto_corr[lcm] for lcm in local_max])[::-1]    # Ascending order
        max_local_max = sorted_local_max[0]     # Default
        if rank == 1: max_local_max = sorted_local_max[0]
        if rank == 2: 
            for i in sorted_local_max[1:]: 
                if i > sorted_local_max[0]: 
                    max_local_max = i 
                    break
        if rank == 3:
            for i in sorted_local_max[1:]: 
                if i > sorted_local_max[0]: 
                    id_tmp = i
                    break
            for i in sorted_local_max[id_tmp:]:
                if i > sorted_local_max[id_tmp]: 
                    max_local_max = i           
                    break
        # print('sorted_local_max: ', sorted_local_max)
        # print('max_local_max: ', max_local_max)
        if local_max[max_local_max]<3 or local_max[max_local_max]>300:
            return 125
        return local_max[max_local_max]+base
    except:
        return 125
    

# determine sliding window (period) based on ACF, Original version
def find_length(data):
    if len(data.shape)>1:
        return 0
    data = data[:min(20000, len(data))]
    
    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]
    
    
    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max]<3 or local_max[max_local_max]>300:
            return 125
        return local_max[max_local_max]+base
    except:
        return 125

