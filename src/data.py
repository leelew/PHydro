import json
import math

import numpy as np
import tensorflow as tf


class Dataset():
    def __init__(self, cfg, mode):
        self.inputs_path = cfg["inputs_path"]
        self.use_ancillary = cfg["use_ancillary"]
        self.seq_len = cfg["seq_len"]
        self.split_ratio = cfg["split_ratio"]
        self.interval = cfg["interval"]
        self.window_size = cfg["window_size"]
        self.num_out = cfg["num_out"]
        self.mode = mode

    def fit(self):
        # load input data (nt, ngrid, nfeat)
        forcing, hydro, ancillary = self._load_input()
        if self.use_ancillary:
            ancillary = np.tile(ancillary[np.newaxis], (self.nt, 1, 1))
            forcing = np.concatenate([forcing, ancillary], axis=-1)

        # remove outlier (for nonreasonable runoff)
        hydro = self._remove_outlier(hydro, threshold=100)

        # make aux data (precip(t), swvl(t-1))
        # NOTE: Before normalization to keep unit mm
        precip = np.nansum(forcing[:,:,:2], axis=-1, keepdims=True)
        swvl_prev = 0
        soil_depth = [70, 210, 720, 1864.6] # mm
        for i in range(4): swvl_prev+=hydro[:,:,i:i+1]*soil_depth[i]
        aux = np.concatenate([precip[1:], swvl_prev[:-1]], axis=-1)

        # get scaler
        if self.mode == 'train':
            scaler = self._get_minmax_scaler(forcing, hydro)
            self._save_scaler(self.inputs_path, scaler)
        else:
            # NOTE: ensure data is fitted in train mode before
            scaler = self._load_scaler(self.inputs_path)

        # normalize input for train/valid/test
        forcing = self._minmax_normalize(forcing, scaler, is_feat=True)

        # normalize output (for train dataset)
        if self.mode == 'train':
            hydro = self._minmax_normalize(hydro, scaler, is_feat=False)

        # transpose to (nt, ngrid, nfeat)
        
        # make training/test data
        if self.mode in ['train']:
            # (nt_, ngrid, nfeat)
            N = int(self.split_ratio*hydro.shape[0])
            x_train, y_train = forcing[:N], hydro[:N]
            x_valid, y_valid = forcing[N:], hydro[N:]
            # (ngrid*nyears*1/offset, seq_len, nfeat)
            x_train, y_train = self._split_into_batch(x_train, y_train, self.seq_len, 0.5)
            x_valid, y_valid = self._split_into_batch(x_valid, y_valid, self.seq_len, 0.5)
            return x_train, y_train, x_valid, y_valid
        else:
            # (ngrids, nsamples/1, seq_len, nfeat) 
            # FIXME: Maybe change interval to 365.
            x_test, y_test = self._make_inference_data(
                forcing, hydro, self.seq_len, 1, self.window_size)
            return x_test, y_test

    def _load_input(self):
        forcing = np.load(self.inputs_path +
                          "guangdong_9km_forcing_{}.npy".format(self.mode))
        hydro = np.load(self.inputs_path +
                        "guangdong_9km_hydrology_{}.npy".format(self.mode))
        ancillary = np.load(self.inputs_path +
                            "guangdong_9km_ancillary.npy")
        return forcing, hydro, ancillary

    def _remove_outlier(self, hydro, threshold=200):  # (nt, ngrids, nfeat)
        """remove outlier larger than mean+3*std and less than mean-3*std"""
        #std = np.nanstd(input, axis=(0), keepdims=True)  # (1, ngrids, nfeat)
        #mean = np.nanmean(input, axis=(0), keepdims=True)  # (1, ngrids, nfeat)
        #input[np.where(input > (mean+3*std))] = np.nan
        #input[np.where(input < (mean-3*std))] = np.nan
        #self.remove_outlier = True

        # @(Zhongwang Wei): remove unreasonable runoff > 200mm/day and 
        # interplote by adjacency two days
        rnof = hydro[:,:,-1]
        rnof[rnof>threshold] = np.nan
        nt, ngrid, nout = hydro.shape
        for i in range(ngrid):
            tmp = rnof[:,i]
            if np.isnan(tmp).any():
                idx = np.where(np.isnan(tmp))[0]
                for j in idx:
                    if j == 0: tmp[j] = tmp[j+1]
                    elif j == nt: tmp[j] = tmp[j-1]
                    else: tmp[j] = (tmp[j-1]+tmp[j+1])/2
            rnof[:,i] = tmp
        hydro[:,:,-1] = rnof
        return hydro

    def _get_minmax_scaler(self, x, y):  # (nt, ngrids, nfeat) - (1, ngrids, nfeat)
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
        for i in range(X.shape[1]):  # parfor each grids
            tmpx, tmpy = self._reshape_1d_data(
                X[:,i], y[:,i], seq_len, interval, window_size)  # (nt, nfeat)
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
        y_new = np.zeros((n, seq_length, num_out))*np.nan
        
        for i in range(n):
            x_new[i] = X[i*interval:i*interval+seq_length, :]
            y_new[i] = y[i*interval+seq_length-1, :]
        return x_new, y_new

    def _split_into_batch(self, X, y, seq_len=365, offset=1, window_size=0):
        """
        split training data into batches with size of batch_size
        :param data_array: [numpy array] array of training data with dims [nseg,
        ndates, nfeat]
        :param seq_len: [int] length of sequences (i.e., 365)
        :param offset: [float] 0-1, how to offset the batches (e.g., 0.5 means that
        the first batch will be 0-365 and the second will be 182-547)
        :return: [numpy array] batched data with dims [nbatches, nseg, seq_len
        (batch_size), nfeat]
        """
        #(nt_, ngrid, nfeat)
        x_batchs, y_batchs = [], []
        for i in range(int(1 / offset)):
            start = int(i * offset * seq_len)
            idx = np.arange(start, y.shape[0]+1, seq_len)
            split_x = np.split(X, indices_or_sections=idx, axis=0) #(seq_len,ngrid,nfeat)
            split_y = np.split(y, indices_or_sections=idx, axis=0) 
            # add all but the first and last batch since they will be smaller
            for s in split_x:
                if s.shape[0] == seq_len:
                    print(s.shape)
                    x_batchs.append(s)
            for s in split_y:
                if s.shape[0] == seq_len:
                    print(s.shape)
                    y_batchs.append(s)
        x_batchs = np.concatenate(x_batchs, axis=1)
        y_batchs = np.concatenate(y_batchs, axis=1) #(seq_len,ngrid*nyears*1/offset,nfeat)
        print(x_batchs.shape, y_batchs.shape)
        return np.transpose(x_batchs,(1,0,2)), np.transpose(y_batchs,(1,0,2))

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


