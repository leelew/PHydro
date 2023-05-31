import math

import numpy as np
import tensorflow as tf


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

def make_train_data(cfg, x, y, aux, scaler, stride=1):
    ngrid, nt, _ = x.shape
    mean, std = np.array(scaler["y_mean"]), np.array(scaler["y_std"])
    # ngrid, 1, nout
    mean, std = np.transpose(mean, (1,0,2)), np.transpose(std, (1,0,2))

    # make samples for each grids        
    n = (nt-cfg["seq_len"]+1)//stride 
    x_new = np.zeros((ngrid, n, cfg["seq_len"], cfg["num_feat"]))*np.nan
    y_new = np.zeros((ngrid, n, 2, y.shape[-1]))*np.nan
    aux_new = np.zeros((ngrid, n))*np.nan
    for i in range(n):
        x_new[:,i] = x[:,i*stride:i*stride+cfg["seq_len"]]
        y_new[:,i] = y[:,i*stride+cfg["seq_len"]-2:i*stride+cfg["seq_len"]]
        aux_new[:,i] = aux[:,i*stride+cfg["seq_len"]-1] 
    mean = np.tile(mean,(1,n,1))
    std = np.tile(std,(1,n,1))

    a, b, c, d, e = [],[],[],[],[]
    for i in range(ngrid):
        a.append(x_new[i])
        b.append(y_new[i])
        c.append(aux_new[i])
        d.append(mean[i])
        e.append(std[i])
    x_new = np.concatenate(a,axis=0)
    aux_new = np.concatenate(c,axis=0)
    y_new = np.concatenate(b,axis=0)
    mean = np.concatenate(d, axis=0)
    std = np.concatenate(e, axis=0)
    return x_new, y_new, aux_new, mean, std



# Fang and Shen (2020), JHM 
def load_train_data(cfg, x, y, aux, scaler):
    ngrid, nt, _ = x.shape
    mean, std = np.array(scaler["y_mean"]), np.array(scaler["y_std"])

    # random over grids and timesteps
    idx_grid = np.random.randint(0, ngrid, cfg["batch_size"])
    #idx_time = np.random.randint(0, nt-cfg["seq_len"], 1)[0]
    # (0-158), (173,401), (416,644), (659,714)
    m1 = np.arange(0,158*24)
    m2 = np.arange(173*24,401*24)
    m3 = np.arange(416*24,644*24)
    m4 = np.arange(659*24,714*24)
    idx = np.concatenate([m1,m2,m3,m4],axis=0)
    j = np.random.randint(0,len(idx),1)[0]
    idx_time = idx[j]

    # make batch of x, y, aux(precip), mean and std
    # NOTE: we predict the last two steps
    x = x[idx_grid, idx_time:idx_time+cfg["seq_len"]]
    y = y[idx_grid, idx_time+cfg["seq_len"]-2:idx_time+cfg["seq_len"]] 
    aux = aux[idx_grid, idx_time+cfg["seq_len"]-1]
    mean = mean[0,idx_grid]
    std = std[0,idx_grid]
    return x, y, aux, mean, std


def load_test_data(cfg, x, y, aux, scaler, stride=1):
    ngrid, nt, _ = x.shape
    mean, std = np.array(scaler["y_mean"]), np.array(scaler["y_std"])
    mean, std = np.transpose(mean, (1,0,2)), np.transpose(std, (1,0,2))

    # make samples for each grids
    # NOTE: we stride the validate data to save CPU memory
    n = (nt-cfg["seq_len"]+1)//stride 
    x_new = np.zeros((ngrid, n, cfg["seq_len"], cfg["num_feat"]))*np.nan
    y_new = np.zeros((ngrid, n, 2, y.shape[-1]))*np.nan
    aux_new = np.zeros((ngrid, n))*np.nan
    for i in range(n):
        x_new[:,i] = x[:,i*stride:i*stride+cfg["seq_len"]]
        y_new[:,i] = y[:,i*stride+cfg["seq_len"]-2:i*stride+cfg["seq_len"]]
        aux_new[:,i] = aux[:,i*stride+cfg["seq_len"]-1]   
    return x_new, y_new, aux_new, np.tile(mean,(1,n,1)), np.tile(std,(1,n,1))


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
