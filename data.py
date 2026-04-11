# Imports
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Ignore warnings
warnings.filterwarnings('ignore')

class CNNDataset(torch.utils.data.Dataset):
    """
    Dataset for loading spectrogram-based inputs and labels for CNN models.

    Loads *precomputed* spectrogram data. 
    Attention: Selects specific parts of the data depending on `data_request`, this is done
    so that the same dataset can be used for melspect and for the mfcc.

    Args:
        path - path to numpy with spectrogram data (Path)
        labels_df - IDs and labels (pd.DataFrame)
        data_request - controls which part of data is returned (int, default=0)

    Returns:
        if data_requsted:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        else:
            Tuple[torch.Tensor, torch.Tensor]
         - input tensor(s) and corresponding label tensor, depends on data_requsted
    """
    def __init__(self, path: Path, labels_df: pd.DataFrame, data_request: int = 0):
        self.data_file = np.load(path)
        self.labels_df = labels_df
        self.data_requsted = data_request

    def __len__(self):
        return self.labels_df.shape[0]

    def __getitem__(self, idx: int):
        mel_id = int(self.labels_df.iloc[idx]['id'])
        label = self.labels_df.iloc[idx]['word_labels']
        label_torch = torch.Tensor([label])
        melspect_torch = None
        
        if not self.data_requsted:
            melspect = self.data_file[mel_id][12:76, :]
            melspect_torch = self.transform_data(melspect)

        else:
            melspect = np.concatenate((self.data_file[mel_id][:12, :], self.data_file[mel_id][76:, :]), axis=0)
            melspect_torch = self.transform_data(melspect)
            if self.data_requsted != 1:
                mel_2 = self.transform_data(self.data_file[mel_id][12:76, :])
                return melspect_torch,mel_2,label_torch
            
        return melspect_torch, label_torch

    def set_to_additional_values(self,data_request):
        self.data_requsted = data_request

    def transform_data(self,data):
        data_reshaped = data.reshape((1, data.shape[0], data.shape[1]))
        return torch.Tensor(data_reshaped)
