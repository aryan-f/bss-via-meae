import os
import pickle
from glob import glob

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import get_original_cwd
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class DataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.data_path = config.data_path
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.x_plot = None

    def prepare_data(self):
        head, tail = os.path.split(self.data_path)
        feat_file = os.path.join(head, 'BCC.feat.pkl')
        with np.load(self.data_path) as data:
            self.train_data = data['X_train'], data['y_train']
            self.val_data = data['X_val'], data['y_val']
            self.test_data = data['X_test'], data['y_test']
            self.x_plot = self.val_data[0:5]
        with open(feat_file, 'rb') as outfile:
            self.feat = pickle.load(outfile, encoding='latin1')

    def setup(self, stage):
        self.train_data = MassSpecDataset(self.train_data, self.feat)
        self.val_data = MassSpecDataset(self.val_data, self.feat)
        self.test_data = MassSpecDataset(self.test_data, self.feat)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )


class MassSpecDataset(Dataset):
    def __init__(self, data, feat):
        self.data, self.labels = data
        self.feat = feat

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = np.expand_dims(self.data[idx], axis=0)
        x = x.astype(np.float32)
        return x, x[:, self.feat]
