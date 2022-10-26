import math
import tensorflow as tf
import numpy as np


# NOTE: `load_train_data` and `load_test_data` is based on
#       Fang and Shen(2020), JHM. It doesn't used all samples
#       (all timestep over all grids)to train LSTM model. 
#       Otherwise, they construct train samples by select 
#       `batch_size` grids, and select `seq_len` timesteps.
#       We found that this method suit for training data that
#       has large similarity (e.g., model's output, reanalysis)
#       However, if we trained on in-situ data such as CAMELE
#       Kratzert et al. (2019), HESS is better.    
#  
# Fang and Shen (2020), JHM 
def load_train_data(x, y, cfg):
    ngrid, nt, _ = x.shape
    idx_grid = np.random.randint(0, ngrid, cfg["batch_size"])
    idx_time = np.random.randint(0, nt-cfg["seq_len"], 1)[0]
    x = x[idx_grid, idx_time:idx_time+cfg["seq_len"]]
    y = y[idx_grid, idx_time+cfg["seq_len"]-1]
    return x, y


def load_test_data(X, y, seq_length, interval=1):
    ngrid, nt, num_features = X.shape
    _, _, num_out = y.shape
    n = (nt-seq_length+1) // interval

    x_new = np.zeros((ngrid, n, seq_length, num_features))*np.nan
    y_new = np.zeros((ngrid, n, num_out))*np.nan
    for i in range(n):
        x_new[:,i] = X[:,i*interval:i*interval+seq_length]
        y_new[:,i] = y[:,i*interval+seq_length-1]
    return x_new, y_new


# Kratzert et al.(2019), HESS
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
        x_ = self.x[grid_idx]
        y_ = self.y[grid_idx]
        return x_, y_

    def on_epoch_end(self):
        self.indexes = np.arange(self.ngrids)
        if self.shuffle:
            np.random.shuffle(self.indexes)
