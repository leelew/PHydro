"""
Make inputs for all types of models.

<<<<<<<< HEAD
Author: Lu Li 
11/1/2022 - First edition
"""

import json
import numpy as np


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
        # load input data [nt,ngrid,...]
        forcing, hydro, ancillary = self._load_input()

        # Correct water balance [nt-1,ngrid,...]
        # NOTE: CoLM soil moisture is available as mean of
        #       day, however, the delta(swc) in water balance
        #       should be swc(24)-swc(0). Thus we need to
        #       correct the water balance by add the residual
        #       of water balance into precipitation.
        forcing, hydro, aux = self._correct_mass_conserve(forcing, hydro)

        # remove unreasonable runoff
        # NOTE: 30mm/day may not be the best setting, but we didn't 
        #       found a better way to remove unreasonable runoff.
        hydro = self._remove_outlier(hydro, threshold=30)

        # get scaler
        if self.mode == 'train':
            scaler = self._get_z_scaler(forcing, hydro)
            self._save_scaler(self.inputs_path, scaler)
        elif self.mode == 'test':
            # NOTE: ensure data is fitted in train mode before
            scaler = self._load_scaler(self.inputs_path)

        # normalize input 
        # NOTE: Z-score normalize is much better than minmax 
        #       normalize based on prelinminary test.
        forcing = self._z_normalize(forcing, scaler, is_feat=True)

        # normalize output (only for train dataset)
        if self.mode == 'train':
            hydro = self._z_normalize(hydro, scaler, is_feat=False)

        # (Optional) add spatial normalized ancillary data
        if self.use_ancillary:
            ancillary = self._spatial_normalize(ancillary)
            ancillary = np.tile(ancillary[np.newaxis],(forcing.shape[0],1,1))
            forcing = np.concatenate([forcing, ancillary], axis=-1)

        # trans nt and ngrid [ngrid, nt, nfeat]
        forcing = np.transpose(forcing, (1, 0, 2))
        hydro = np.transpose(hydro, (1, 0, 2))
        aux = np.transpose(aux, (1, 0))

        # make training/test data
        if self.mode == 'train':
            return forcing, hydro, aux
        elif self.mode == 'test':
            forcing, hydro, aux = self._make_inference_data(forcing, hydro, aux, self.seq_len)
            return forcing, hydro, aux

    def _load_input(self):
        forcing = np.load(self.inputs_path+"forcing_gd_9km_{}.npy".format(self.mode))
        hydro = np.load(self.inputs_path+"hydro_gd_9km_{}.npy".format(self.mode))
        ancillary = np.load(self.inputs_path+"ancil_gd_9km.npy")
        return forcing, hydro, ancillary

    def _correct_mass_conserve(self, forcing, hydro):
        soil_depth = [70, 210, 720, 1864.6]  # mm
        hydro_prev, hydro, forcing = hydro[:-1], hydro[1:], forcing[1:]
        swvl, swvl_prev = 0, 0
        for i in range(4):
            swvl += hydro[:, :, i]*soil_depth[i]
            swvl_prev += hydro_prev[:,:,i]*soil_depth[i]
        mc_in = np.nansum(forcing[:,:,:2], axis=-1) + swvl_prev
        mc_out = swvl + np.nansum(hydro[:,:,4:], axis=-1)
        diff = mc_in - mc_out

        # if diff>0, then add to runoff;
        # if diff<0, then add to precipitation;
        for i in range(diff.shape[0]):
            for j in range(diff.shape[1]):
                tmp = diff[i,j]
                if tmp < 0:
                    forcing[i,j,0] = forcing[i,j,0]-diff[i,j]
                else:
                    hydro[i,j,-1] = hydro[i,j,-1]+diff[i,j]
        # get mass in
        aux = np.nansum(forcing[:,:,:2], axis=-1) + swvl_prev
        return forcing, hydro, aux 

    def _remove_outlier(self, hydro, threshold=30):
        """
        std = np.nanstd(input, axis=(0), keepdims=True)  # (1, ngrids, nfeat)
        mean = np.nanmean(input, axis=(0), keepdims=True)  # (1, ngrids, nfeat)
        input[np.where(input > (mean+3*std))] = np.nan
        input[np.where(input < (mean-3*std))] = np.nan
        self.remove_outlier = True
        """
        """
        # @(Zhongwang Wei): remove unreasonable runoff > 200mm/day and
        # interplote by adjacency two days
        rnof = hydro[:, :, -1]
        rnof[rnof > threshold] = np.nan
        nt, ngrid, nout = hydro.shape
        for i in range(ngrid):
            tmp = rnof[:, i]
            if np.isnan(tmp).any():
                idx = np.where(np.isnan(tmp))[0]
                for j in idx:
                    if j == 0:
                        tmp[j] = tmp[j+1]
                    elif j == nt:
                        tmp[j] = tmp[j-1]
                    else:
                        tmp[j] = np.nanmean(tmp)  # (tmp[j-1]+tmp[j+1])/2
            rnof[:, i] = tmp
        hydro[:, :, -1] = rnof
        """
        rnof = hydro[:, :, -1]
        rnof[rnof > threshold] = np.nan
        hydro[:, :, -1] = rnof
        return hydro

    def _get_minmax_scaler(self, x, y):
        scaler = {}
        scaler["x_min"] = np.nanmin(x, axis=(0), keepdims=True).tolist()
        scaler["x_max"] = np.nanmax(x, axis=(0), keepdims=True).tolist()
        scaler["y_min"] = np.nanmin(y, axis=(0), keepdims=True).tolist()
        scaler["y_max"] = np.nanmax(y, axis=(0), keepdims=True).tolist()
        return scaler

    def _get_z_scaler(self, x, y):
        scaler = {}
        scaler["x_mean"] = np.nanmean(x, axis=(0), keepdims=True).tolist()
        scaler["x_std"] = np.nanstd(x, axis=(0), keepdims=True).tolist()
        scaler["y_mean"] = np.nanmean(y, axis=(0), keepdims=True).tolist()
        scaler["y_std"] = np.nanstd(y, axis=(0), keepdims=True).tolist()
        return scaler

    def _save_scaler(self, inputs_path, scaler):
        with open(inputs_path + 'scaler.json', 'w') as f:
            json.dump(scaler, f)

    def _load_scaler(self, inputs_path):
        with open(inputs_path + 'scaler.json', "r") as f:
            scaler = json.load(f)
        return scaler

    def _z_normalize(self, input, scaler, is_feat):
        if is_feat:
            input = (input - np.array(scaler["x_mean"])) / (
                np.array(scaler["x_std"]))
        else:
            input = (input - np.array(scaler["y_mean"])) / (
                np.array(scaler["y_std"]))
        return input

    def _minmax_normalize(self, input, scaler, is_feat):
        """normalize features using pre-computed statistics."""
        if is_feat:
            input = (input - np.array(scaler["x_min"])) / (
                np.array(scaler["x_max"])-np.array(scaler["x_min"]))
        else:
            input = (input - np.array(scaler["y_min"])) / (
                np.array(scaler["y_max"])-np.array(scaler["y_min"]))
        return input

    def _spatial_normalize(self, static):
        # (ngrid, nfeat) for static data
        mean = np.nanmean(static, axis=(0), keepdims=True)
        std = np.nanstd(static, axis=(0), keepdims=True)
        return (static-mean)/std

    def _make_inference_data(self, 
                             x, 
                             y, 
                             aux, 
                             seq_len=365, 
                             interval=1, 
                             window_size=0):
        x_, y_, aux_ = [], [], []
        for i in range(x.shape[0]): 
            tmpx, tmpy, tmp_aux = self._reshape_1d_data(
                x[i], y[i], aux[i], seq_len, interval, window_size) 
            x_.append(tmpx)
            y_.append(tmpy)
            aux_.append(tmp_aux)
        # (ngrids, nsamples, seq_len, nfeat)
        return np.stack(x_, axis=0), np.stack(y_, axis=0), np.stack(aux_, axis=0) 

    def _reshape_1d_data(self, 
                         x, 
                         y, 
                         aux, 
                         seq_len=365, 
                         interval=1,
                         window_size=0):
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
        num_samples, num_features = x.shape
        _, num_out = y.shape
        n = (num_samples-seq_len+1) // interval
        x_new = np.zeros((n, seq_len, num_features))*np.nan
        y_new = np.zeros((n, num_out))*np.nan
        aux_new = np.zeros((n, ))*np.nan

        for i in range(n):
            x_new[i] = x[i*interval:i*interval+seq_len]
            y_new[i] = y[i*interval+seq_len-1]
            aux_new[i] = aux[i*interval+seq_len-1]
        return x_new, y_new, aux_new

    def __split_into_batch(self, X, y, seq_len=365, offset=1, window_size=0):
        """split training data into batches with size of batch_size

        Params
        ------
            offset: [float] 0-1, how to offset the batches (e.g., 0.5 means that
                    the first batch will be 0-365 and the second will be 182-547)
        """
        #(nt_, ngrid, nfeat)
        x_batchs, y_batchs = [], []
        for i in range(int(1 / offset)):
            start = int(i * offset * seq_len)
            idx = np.arange(start, y.shape[0]+1, seq_len)
            split_x = np.split(X, indices_or_sections=idx,
                               axis=0)  # (seq_len,ngrid,nfeat)
            split_y = np.split(y, indices_or_sections=idx, axis=0)
            # add all but the first and last batch since they will be smaller
            for s in split_x:
                if s.shape[0] == seq_len:
                    x_batchs.append(s)
            for s in split_y:
                if s.shape[0] == seq_len:
                    y_batchs.append(s)
        x_batchs = np.concatenate(x_batchs, axis=1)
        # (seq_len,ngrid*nyears*1/offset,nfeat)
        y_batchs = np.concatenate(y_batchs, axis=1)
        return np.transpose(x_batchs, (1, 0, 2)), np.transpose(y_batchs, (1, 0, 2))
