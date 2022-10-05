import json
import math

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle


class Dataset():
    def __init__(self, cfg, mode):
        self.inputs_path = cfg["inputs_path"]
        self.use_ancillary = cfg["use_ancillary"]
        self.seq_len = cfg["seq_len"]
        self.interval = cfg["interval"]
        self.window_size = cfg["window_size"]
        self.mode = mode

    def fit(self):
        # load input data
        forcing, hydro, ancillary = self._load_input()
        self.nt = forcing.shape[0]
        if self.use_ancillary:
            ancillary = np.tile(ancillary[np.newaxis], (self.nt, 1, 1))
            forcing = np.concatenate([forcing, ancillary], axis=-1)

        # Optional: remove outlier (for nonreasonable runoff)
        hydro = self._remove_outlier(hydro)

        # get scaler
        if self.mode == 'train':
            scaler = self._get_minmax_scaler(forcing, hydro)
            self._save_scaler(self.inputs_path, scaler)
        else:
            # ensure data is fitted in train mode before
            scaler = self._load_scaler(self.inputs_path)

        # normalize input for train/valid/test
        forcing = self._minmax_normalize(forcing, scaler, is_feat=True)

        # Optional: normalize output (for train dataset)
        if self.mode == 'train':
            hydro = self._minmax_normalize(hydro, scaler, is_feat=False)
            # FIXME: process NaN in forcing, hydro;
            # NOTE: interpolate is not suit for runoff?? discuss with Wei
            hydro[np.isnan(hydro)] = 0

        # return train/valid/test data, transpose to (ngrids, nt, nfeat)
        forcing = np.transpose(forcing, (1, 0, 2))
        hydro = np.transpose(hydro, (1, 0, 2))

        if self.mode == 'train':
            # (ngrids, nt, nfeat)
            return forcing, hydro
        else:
            # (ngrids, nsamples, seq_len, nfeat)
            forcing, hydro = self._make_inference_data(
                forcing, hydro, self.seq_len, self.interval, self.window_size)
            return forcing, hydro

    def _load_input(self):
        forcing = np.load(self.inputs_path +
                          "guangdong_9km_forcing_{}.npy".format(self.mode))
        hydro = np.load(self.inputs_path +
                        "guangdong_9km_hydrology_{}.npy".format(self.mode))
        ancillary = np.load(self.inputs_path +
                            "guangdong_9km_ancillary.npy")
        return forcing, hydro, ancillary

    def _remove_outlier(self, input):  # (nt, ngrids, nfeat)
        """remove outlier larger than mean+3*std and less than mean-3*std"""
        std = np.nanstd(input, axis=(0), keepdims=True)  # (1, ngrids, nfeat)
        mean = np.nanmean(input, axis=(0), keepdims=True)  # (1, ngrids, nfeat)
        input[np.where(input > (mean+3*std))] = np.nan
        input[np.where(input < (mean-3*std))] = np.nan
        self.remove_outlier = True
        return input

    def _get_minmax_scaler(self, x, y):  # (nt, ngrids, nfeat)
        scaler = {}
        scaler["x_min"] = np.nanmin(x, axis=(0), keepdims=True).tolist()
        scaler["x_max"] = np.nanmax(x, axis=(0), keepdims=True).tolist()
        scaler["y_min"] = np.nanmin(y, axis=(0), keepdims=True).tolist()
        scaler["y_max"] = np.nanmax(y, axis=(0), keepdims=True).tolist()
        return scaler

    def _save_scaler(self, inputs_path, scaler):
        with open(inputs_path + 'scaler.json', 'w') as f:
            json.dump(scaler, f)

    def _load_scaler(self, inputs_path):
        with open(inputs_path + 'scaler.json', "r") as f:
            scaler = json.load(f)
        return scaler

    def _minmax_normalize(self, input, scaler, is_feat):
        """normalize features using pre-computed statistics."""
        if is_feat:
            input = (input - np.array(scaler["x_min"])) / (
                np.array(scaler["x_max"])-np.array(scaler["x_min"]))
        else:
            input = (input - np.array(scaler["y_min"])) / (
                np.array(scaler["y_max"])-np.array(scaler["y_min"]))
        return input

    @staticmethod
    def reverse_normalize(input, scaler):
        """reverse normalized forecast using pre-computed statistics"""
        return input * (np.array(scaler["y_max"])-np.array(scaler["y_min"])) + \
            np.array(scaler["y_min"])

    def _make_inference_data(self, X, y, seq_len, interval, window_size):
        x_, y_ = [], []
        for i in range(X.shape[0]):  # parfor each grids
            tmpx, tmpy = self._reshape_1d_data(
                X[i], y[i], seq_len, interval, window_size)  # (nt, nfeat)
            x_.append(tmpx)
            y_.append(tmpy)
        x_ = np.stack(x_, axis=0)
        y_ = np.stack(y_, axis=0)
        return x_, y_  # (ngrids, nsamples, seq_len, nfeat)

    def _reshape_1d_data(self, X, y, seq_length, interval=365, window_size=0):
        """reshape data into LSTM many-to-one input samples

        Parameters
        ----------
        x : np.ndarray
            Input features of shape [num_samples, num_features]
        y : np.ndarray
            Output feature of shape [num_samples, 1]
        seq_length : int
            Length of the requested input sequences.
        interval: int
            interval of time length to generate samples.
        window_size: int
            window size between x and y

        Returns
        -------
        x_new: np.ndarray
            shape of [num_samples*, seq_length, num_features], where 
            num_samples* is equal to num_samples - seq_length + 1, due to 
            the need of a warm start at the beginning
        y_new: np.ndarray
            The target value for each sample in x_new
        """
        num_samples, num_features = X.shape
        _, num_out = y.shape
        n = (num_samples-seq_length+1) // interval

        x_new = np.zeros((n, seq_length, num_features))*np.nan
        y_new = np.zeros((n, num_out))*np.nan

        for i in range(0, n):
            x_new[i] = X[i*interval:i*interval+seq_length, :]
            y_new[i] = y[i*interval+window_size+seq_length-1, :]
        return x_new, y_new

    @staticmethod
    def make_training_data(X, y, ngrids, seq_len):
        """make training data as Shen et al. 2019"""
        num_grids, time_len, num_features = X.shape
        idx_grid = np.random.randint(0, num_grids, ngrids)
        idx_time = np.random.randint(0, time_len-seq_len, 1)[0]
        X = X[idx_grid, idx_time:idx_time+seq_len, :]
        y = y[idx_grid, idx_time+seq_len-1, :]
        return X, y

    def __len__(self):
        return self.nt


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, cfg, shuffle=True):
        super().__init__()
        self.x, self.y = x, y  # (ngrid, nt, nfeat)-(ngrid, nt, nout)
        self.shuffle = shuffle
        self.batch_size = cfg["batch_size"]
        self.seq_len = cfg["seq_len"]
        self.ngrids = x.shape[0]
        self.nt = x.shape[1]

    def __len__(self):
        return math.ceil(self.ngrids / self.batch_size)

    def __getitem__(self, idx):
        # index for grids
        grid_idx = self.indexes[idx * self.batch_size:(idx+1)*self.batch_size]
        # index for timestep
        begin_idx = np.random.randint(0, self.nt-self.seq_len, 1)[0]
        # crop
        x_ = self.x[grid_idx, begin_idx:begin_idx+self.seq_len]
        y_ = self.y[grid_idx, begin_idx+self.seq_len-1]
        return x_, y_

    def on_epoch_end(self):
        self.indexes = np.arange(self.ngrids)
        if self.shuffle:
            np.random.shuffle(self.indexes)
