"""
Make inputs for all types of models.

<<<<<<<< HEAD
Author: Lu Li 
11/1/2022 - V1.0
12/9/2022 - V2.0
"""

import json
import numpy as np
from utils import make_CoLM_soil_depth


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
        # calculate soil depth in CoLM
        depth, zi = make_CoLM_soil_depth() # m/cm
        self.soil = [70, 210, 720, 10*(zi[9]-100)] # mm

    def fit(self):
        # load input data shape as [nt,ngrid,.]
        forcing, hydro, ancillary = self._load_input()

        # remove unreasonable runoff
        hydro = self._remove_rnof_outlier(hydro, threshold=20)

        # Correct water balance [nt-1,ngrid,...]
        # NOTE: CoLM water balance is defined as
        #       P = ET + R + delta(SWC+ldew+wa+scv)
        #       In our study, we focus on simulating variables ET, 
        #       R, SWC, thus we treat `wa`... as static variable.
        #       Thus, to enforce water balance, we need to 
        #       remove the residual of water balance to P.
        forcing, hydro, aux = self._correct_mass_conserve(forcing, hydro)

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

        # trans shape as [ngrid,nt,.]
        forcing = np.transpose(forcing, (1, 0, 2))
        hydro = np.transpose(hydro, (1, 0, 2))
        aux = np.transpose(aux, (1, 0))

        # make training/test data
        return forcing, hydro, aux

    def _load_input(self):
        forcing = np.load(self.inputs_path+"gd_9km_forcing_{}.npy".format(self.mode))
        hydro = np.load(self.inputs_path+"gd_9km_hydrology_{}.npy".format(self.mode))
        ancillary = np.load(self.inputs_path+"gd_9km_ancillary.npy")        
        return forcing, hydro, ancillary

    def _remove_rnof_outlier(self, hydro, threshold=20):
        # NOTE: In CoLM, there are some runoff larger than 1e3 mm/h, this is
        #       caused by the parameterization of soil moisture, which cannot
        #       be solved now. Thus I remove runoff larger than 20 mm/h to 
        #       ensure the robustness of results.
        rnof = hydro[:, :, -1]
        rnof[rnof > threshold] = np.nan
        # @(Zhongwang Wei): interplote by adjacency two days
        """
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
        """
        hydro[:, :, -1] = rnof
        return hydro

    def _correct_mass_conserve(self, forcing, hydro):
        hydro_prev, hydro, forcing = hydro[:-1], hydro[1:], forcing[1:]
        # cal mass in/out
        swvl, swvl_prev = 0, 0
        for i in range(4):
            swvl += hydro[:, :, i]*self.soil[i]
            swvl_prev += hydro_prev[:,:,i]*self.soil[i]
        mc_in = np.sum(forcing[:,:,:2], axis=-1) + swvl_prev
        mc_out = swvl + np.sum(hydro[:,:,4:], axis=-1)
        # cal diff in mass balance caused by wa, ldew, scv, xerror
        diff = mc_in - mc_out
        # get mass in after remove diff in balance
        aux = np.sum(forcing[:,:,:2], axis=-1) - diff
        return forcing, hydro, aux 

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
        """normalize features using pre-computed statistics."""
        if is_feat:
            input = (input - np.array(scaler["x_mean"])) / (
                np.array(scaler["x_std"]))
        else:
            input = (input - np.array(scaler["y_mean"])) / (
                np.array(scaler["y_std"]))
        return input

    def _spatial_normalize(self, static):
        # (ngrid, nfeat) for static data
        mean = np.nanmean(static, axis=(0), keepdims=True)
        std = np.nanstd(static, axis=(0), keepdims=True)
        return (static-mean)/std