import math
import tensorflow as tf
import numpy as np


def load_data(x, y, cfg):
    """make training data as Shen et al. 2019"""
    ngrid, nt, num_features = x.shape
    idx_grid = np.random.randint(0, ngrid, cfg["batch_size"])
    idx_time = np.random.randint(0, nt-cfg["seq_len"], 1)[0]
    x = x[idx_grid, idx_time:idx_time+cfg["seq_len"], :]
    y = y[idx_grid, idx_time+cfg["seq_len"]-1, :]
    return x, y


def reverse_normalize(input, scaler):
    """reverse normalized forecast using pre-computed statistics"""
    return input * (np.array(scaler["y_max"])-np.array(scaler["y_min"])) + \
        np.array(scaler["y_min"])



class DataGenerator(tf.keras.utils.Sequence):
    """Data generator based on Shen et al."""
    def __init__(self, x, y, cfg, shuffle=True):
        super().__init__()
        self.x, self.y = x, y  # (ngrid, nt, nfeat)-(ngrid, nt, nout)
        self.shuffle = shuffle
        self.batch_size = cfg["batch_size"]
        self.seq_len = cfg["seq_len"]
        self.ngrids = x.shape[0]
        self.nt = x.shape[1]
        self.indexes = np.arange(self.ngrids)
        
    def __len__(self):
        return math.ceil(self.ngrids / self.batch_size)

    def __getitem__(self, idx):
        # index for grids
        grid_idx = self.indexes[idx * self.batch_size:(idx+1)*self.batch_size]
        # index for timestep
        #begin_idx = np.random.randint(0, self.nt-self.seq_len, 1)[0]
        # crop
        #x_ = self.x[grid_idx, begin_idx:begin_idx+self.seq_len]
        #y_ = self.y[grid_idx, begin_idx+self.seq_len-1]
        x_ = self.x[grid_idx]
        y_ = self.y[grid_idx]
        return x_, y_

    def on_epoch_end(self):
        self.indexes = np.arange(self.ngrids)
        if self.shuffle:
            np.random.shuffle(self.indexes)
