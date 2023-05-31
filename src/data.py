import json
import numpy as np


class Dataset():
    def __init__(self, cfg, mode):
        self.inputs_path = cfg["inputs_path"]
        self.use_ancillary = cfg["use_ancillary"]
        self.seq_len = cfg["seq_len"]
        self.mode = mode
        self.soil = [70, 210, 720] # mm
        self.ngrid = cfg["ngrid"]

    def fit(self):
        # load input data shape as [4,nt,ngrid,.]
        forcing, hydro, ancillary = self._load_input()
        _, n_t, n_grid, n_in = forcing.shape
        _, n_t, _, n_out = hydro.shape
        _, _, n_aux = ancillary.shape
        print(forcing.shape, hydro.shape, ancillary.shape)
        print(np.isnan(hydro).any())

        # reshape to [nt,200,nfeat] 
        # NOTE: aware using reshape!!!
        a, b, c = [], [], []
        if self.ngrid == 600:
            R = 12
        if self.ngrid == 400:
            R = 8
        if self.ngrid == 200:
            R = 4
        for i in range(R):
            a.append(forcing[i])
            b.append(hydro[i])
            c.append(ancillary[i])
        forcing = np.concatenate(a, axis=1)
        hydro = np.concatenate(b, axis=1)
        ancillary = np.concatenate(c, axis=0)
        print(forcing.shape, hydro.shape, ancillary.shape)

        # Correct water balance [nt-1,200,...]
        forcing, hydro, aux, p = self._correct_mass_conserve(forcing, hydro)
        print(np.isnan(hydro).any())

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
        return forcing, hydro, aux, p

    def _load_input(self):
        forcing = np.load(self.inputs_path+"ERA5Land_forcing_200_{}.npy".format(self.mode))
        hydro = np.load(self.inputs_path+"ERA5Land_hydrology_200_{}.npy".format(self.mode))
        ancillary = np.load(self.inputs_path+"ancillary_200.npy")    
        if self.ngrid == 400:
            forcing1 = np.load(self.inputs_path+"ERA5Land_forcing_200_{}_1.npy".format(self.mode))
            hydro1 = np.load(self.inputs_path+"ERA5Land_hydrology_200_{}_1.npy".format(self.mode))
            ancillary1 = np.load(self.inputs_path+"ancillary_200_1.npy") 
            forcing = np.concatenate([forcing, forcing1], axis=0)
            hydro = np.concatenate([hydro, hydro1], axis=0)
            ancillary = np.concatenate([ancillary, ancillary1], axis=0)
        if self.ngrid == 600:
            forcing1 = np.load(self.inputs_path+"ERA5Land_forcing_200_{}_1.npy".format(self.mode))
            hydro1 = np.load(self.inputs_path+"ERA5Land_hydrology_200_{}_1.npy".format(self.mode))
            ancillary1 = np.load(self.inputs_path+"ancillary_200_1.npy") 
            forcing = np.concatenate([forcing, forcing1], axis=0)
            hydro = np.concatenate([hydro, hydro1], axis=0)
            ancillary = np.concatenate([ancillary, ancillary1], axis=0)
            forcing1 = np.load(self.inputs_path+"ERA5Land_forcing_200_{}_2.npy".format(self.mode))
            hydro1 = np.load(self.inputs_path+"ERA5Land_hydrology_200_{}_2.npy".format(self.mode))
            ancillary1 = np.load(self.inputs_path+"ancillary_200_2.npy") 
            forcing = np.concatenate([forcing, forcing1], axis=0)
            hydro = np.concatenate([hydro, hydro1], axis=0)
            ancillary = np.concatenate([ancillary, ancillary1], axis=0)
        return forcing, hydro, ancillary

    def _remove_rnof_outlier(self, hydro, threshold=20):
        # NOTE: In CoLM, there are some runoff larger than 1e3 mm/h, this is
        #       caused by the parameterization of soil moisture, which cannot
        #       be solved now. Thus I remove runoff larger than 20 mm/h to 
        #       ensure the robustness of results.
        rnof = hydro[:, :, -1]
        rnof[rnof > threshold] = np.nan
        hydro[:, :, -1] = rnof
        return hydro

    def _correct_mass_conserve(self, forcing, hydro):
        hydro_prev, hydro, forcing = hydro[:-1], hydro[1:], forcing[1:]
        # cal mass in/out
        swvl, swvl_prev = 0, 0
        for i in range(3):
            swvl += hydro[:, :, i]*self.soil[i]
            swvl_prev += hydro_prev[:,:,i]*self.soil[i]
        mc_in = forcing[:,:,0] + swvl_prev
        mc_out = swvl + np.sum(hydro[:,:,3:], axis=-1)
        # cal diff in mass balance caused by wa, ldew, scv, xerror
        diff = mc_in - mc_out
        # get mass in after remove diff in balance
        aux = forcing[:,:,0] - diff
        return forcing, hydro, aux, forcing[:,:,0]

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