from ast import Raise
import numpy as np
import json
import tensorflow as tf
import warnings

class Dataset():
    """Dataset structure for Multi-task Hydro forecasting

    args:
    -----
    cfg:
    is_train:
    interval:
    is_tensorflow:
    drop_remainder:
    shuffle:

    example:
    --------
    -> f = Dataset(args)
    -> X, y = f.fit()
    -> X, y = f.fit_transform()

    Raise:
    ------

    """
    def __init__(self, 
                 cfg, 
                 is_train,      
                 interval,
                 is_tensorflow=True,           
                 drop_remainder=False,
                 shuffle=True):
        self.epochs = cfg["epochs"]
        self.batch_size = cfg["batch_size"]
        self.data_root = cfg["data_root"]
        self.cfg_root = cfg["cfg_root"]
        self.scaler_type = cfg["scaler_type"]
        self.scaler_method = cfg["scaler_method"]
        self.seq_length = cfg["seq_length"]
        self.window_size = cfg["window_size"]
        self.sample_type = cfg["sample_type"]
        self.split_ratio = cfg["split_ratio"]

        self.is_train = is_train
        self.interval = interval
        self.drop_remainder = drop_remainder
        self.shuffle = shuffle
        self.is_tensorflow = is_tensorflow

        self.remove_outlier = False

    def fit(self):
        """generate pre-processed train/valid/test data"""
        # load raw data
        x, y = self._load_raw_data(self.data_root, self.is_train) # (grids, time, features)

        # get attribute
        self.num_grids, self.num_samples, self.num_features = x.shape
        N = int(self.split_ratio*self.num_samples) # index to split train/valid data

        # remove outlier
        y = self._remove_outlier(x, y, self.scaler_type)
        
        # get or load cfg.scaler 
        if self.is_train == 'train':
            scaler = self._get_scaler(x, y, self.scaler_type, self.scaler_method, self.is_train)
        else: 
            scaler = self._load_scaler(self.cfg_root) # (grids, 1, features)
        
        # normalize X
        x = self._normalize(x, "input", scaler, self.scaler_method)

        # if train datasets, normalize y
        if self.is_train in ['train', 'valid']:
            #y[np.isnan(y)] = 0 #FIXME: Maybe is wrong
            y = self._normalize(y, "output", scaler, self.scaler_method)
            y[np.isnan(y)] = 0 #must behind normalize, other present <0
            #print(np.nanmin(y[:]), np.nanmax(y[:]))

        if self.is_train == 'train': return x[:,:N,:], y[:,:N,:]
        if self.is_train == 'valid': return x[:,N:,:], y[:,N:,:]
        if self.is_train == 'test': return x, y

    def fit_transform(self, X, y):
        if self.is_train == 'train':
            x_batch, y_batch = self._make_training_data(X, y, self.batch_size, self.seq_length)
            return x_batch, y_batch
        else:
            X, y = self._make_test_data(X, y, self.seq_length, self.interval, self.window_size)
            return X, y


    """
    def fit_generator(self, X, y):
        # make input data (optional)
        if self.is_train=='train':
            if self.sample_type == 'shen':
                x_batch, y_batch = self._make_training_data(X, y, self.batch_size, self.seq_length)
                return x_batch, y_batch
            #FIXME:Correct for generate batch
            elif self.sample_type == 'kratzert':
                x_all, y_all = [],[]
                for i in range(X.shape[0]):
                    tmpx, tmpy = self._reshape_1d_data(X[i], 
                                                    y[i], 
                                                    self.seq_length, 
                                                    self.interval,
                                                    self.window_size)
                    x_all.append(tmpx)
                    y_all.append(tmpy)

                # if train, change list to tensorflow structure
                if self.is_train=='train':
                    x_all = np.concatenate(x_all, axis=0)
                    y_all = np.concatenate(y_all, axis=0)
                    self.num_samples = y_all.shape[0]
                    if self.is_tensorflow:
                        return self._make_tensorflow_data(x_all, y_all)
        elif self.is_train in ['test','valid']:
            X, y = self._make_test_data(X, y, self.seq_length, self.interval, self.window_size)
            return X, y
    """



    def _load_raw_data(self, data_root: str, is_train: str):
        """load raw data shape as (num_grids, num_timesteps, num_features)"""
        if is_train in ['train','valid']:
            x = np.load(data_root / 'x_train.npy')
            y = np.load(data_root / 'y_train.npy')
        elif is_train == 'test':
            x = np.load(data_root / 'x_test.npy')
            y = np.load(data_root / 'y_test.npy')
        else:
            IOError("mode must setting in ['train', 'valid', 'test]")
        #TODO: assert x(y)_train(test) dims == 3 
        return x, y
    
    def _get_scaler(self, X, y, scaler_type, scaler_method, is_train): 
        """get scaler from train dataset"""
        if not self.remove_outlier:
            warnings.warn("You hadn't remove outlier before get scaler, if you decide to do this, aware of this message.")

        if scaler_method == 'minmax':
            scaler = self.__get_minmax_scaler(X, y, scaler_type)
        elif scaler_method == 'standard':
            scaler = self.__get_standard_scaler(X, y, scaler_type)
        else:
            raise IOError('')

        self._save_scaler(scaler, self.cfg_root)
        return scaler

    def _remove_outlier(self, X, y, scaler_type: str):
        """remove outlier larger than mean+3*std and less than mean-3*std"""
        scaler = self.__get_standard_scaler(X, y, type=scaler_type) 
        threshold = 3*np.array(scaler["output_std"])+np.array(scaler["output_mean"]) #(256, 6)
        y[np.where(y>threshold)] = np.nan

        self.remove_outlier = True
        return y

    def __get_standard_scaler(self, X, y, type: str) -> dict:
        scaler = {}
        if type == 'global':
            scaler["input_mean"] = np.nanmean(X, axis=(0,1), keepdims=True).tolist()
            scaler["input_std"] = np.nanstd(X, axis=(0,1), keepdims=True).tolist()
            scaler["output_mean"] = np.nanmean(y, axis=(0,1), keepdims=True).tolist()
            scaler["output_std"] = np.nanstd(y, axis=(0,1), keepdims=True).tolist()
        elif type == 'region':
            scaler["input_mean"] = np.nanmean(X, axis=(1), keepdims=True).tolist() # (256, 6)
            scaler["input_std"] = np.nanstd(X, axis=(1), keepdims=True).tolist()
            scaler["output_mean"] = np.nanmean(y, axis=(1), keepdims=True).tolist()
            scaler["output_std"] = np.nanstd(y, axis=(1), keepdims=True).tolist()
        else:
            raise IOError(f"Unknown variable type {type}")        
        return scaler

    def __get_minmax_scaler(self, X, y, type: str) -> dict:
        scaler = {}
        if type == 'global': #TODO:change the key
            scaler["input_mean"] = np.nanmin(X, axis=(0,1), keepdims=True).tolist() 
            scaler["input_std"] = np.nanmax(X, axis=(0,1), keepdims=True).tolist()
            scaler["output_mean"] = np.nanmin(y, axis=(0,1), keepdims=True).tolist()
            scaler["output_std"] = np.nanmax(y, axis=(0,1), keepdims=True).tolist()
        elif type == 'region':
            scaler["input_mean"] = np.nanmin(X, axis=(1), keepdims=True).tolist()
            scaler["input_std"] = np.nanmax(X, axis=(1), keepdims=True).tolist()
            scaler["output_mean"] = np.nanmin(y, axis=(1), keepdims=True).tolist()
            scaler["output_std"] = np.nanmax(y, axis=(1), keepdims=True).tolist()
        else:
            raise IOError(f"Unknown variable type {type}")        
        return scaler

    def _save_scaler(self, scaler, cfg_root: str):
        with open(cfg_root / 'cfg.json', 'w') as f:
            json.dump(scaler, f)

    def _load_scaler(self, cfg_root: str):
        with open(cfg_root / 'cfg.json', "r") as f:
            self.scaler = json.load(f)
        return self.scaler

    def _normalize(self, feature: np.ndarray, variable: str, scaler: dict, scaler_type: str) -> np.ndarray:
        """normalize features using pre-computed statistics.

        Parameters
        ----------
        feature : np.ndarray, data to normalize
        variable : str, one of ['inputs', 'output']
        scaler: pre-computed statistics
        """
        if scaler_type == 'standard':
            if variable == 'input':
                feature = (feature - np.array(scaler["input_mean"])) / np.array(scaler["input_std"])
            elif variable == 'output':
                feature = (feature - np.array(scaler["output_mean"])) / np.array(scaler["output_std"])
            else:
                raise RuntimeError(f"Unknown variable type {variable}")
        elif scaler_type == 'minmax':
            if variable == 'input':
                print(feature.shape, np.array(scaler["input_mean"]).shape, np.array(scaler["input_std"]).shape)
                feature = (feature - np.array(scaler["input_mean"])) / (np.array(scaler["input_std"])-np.array(scaler["input_mean"]))
            elif variable == 'output':
                feature = (feature - np.array(scaler["output_mean"])) / (np.array(scaler["output_std"])-np.array(scaler["output_mean"]))
            else:
                raise RuntimeError(f"Unknown variable type {variable}")            
        return feature 
    
    def _reshape_1d_data(self, 
                         X: np.ndarray, 
                         y: np.ndarray, 
                         seq_length: int, 
                         interval: int=365, 
                         window_size: int=0):
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
            reshaped input features of shape [num_samples*, seq_length, num_features], where 
            num_samples* is equal to num_samples - seq_length + 1, due to the need of a warm start at the beginning
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

    def _make_tensorflow_data(self, X, y):
        """make data structure for tensorflow"""
        train_dataset = tf.data.Dataset.from_tensor_slices((X, y))
        train_dataset = train_dataset.batch(batch_size=self.batch_size,
                                            drop_remainder=self.drop_remainder)
        train_dataset = train_dataset.shuffle(buffer_size=self.shuffle_buffer_size)
        return train_dataset

    def _make_training_data(self, X, y, ngrids, seq_len):
        """make training data as Shen et al. 2019"""
        num_grids, time_len, num_features = X.shape
        idx_grid = np.random.randint(0,num_grids,ngrids)
        idx_time = np.random.randint(0, time_len-seq_len, 1)[0]
        X = X[idx_grid, idx_time:idx_time+seq_len, :]
        y = y[idx_grid, idx_time+seq_len-1, :]
        return X, y
    
    def _make_test_data(self, X, y, seq_len, interval, window_size):
        x_, y_ = [],[]
        for i in range(X.shape[0]):
            tmpx, tmpy = self._reshape_1d_data(X[i],
                                               y[i],
                                               seq_len,
                                               interval,
                                               window_size)
            x_.append(tmpx)
            y_.append(tmpy)
        x_ = np.stack(x_, axis=0)
        y_ = np.stack(y_, axis=0)
        return x_, y_ #(grids, sample, intime, features)

    def __len__(self):
        return self.num_samples

    def _load_attributes(self):
        pass

    def reverse_normalize(self, 
                          feature, 
                          variable: str, 
                          scaler_method: str, 
                          is_multivars: int) -> np.ndarray:
        """reverse normalized features using pre-computed statistics"""
        print(is_multivars)
        a, b = np.array(self.scaler["input_mean"]), np.array(self.scaler["input_std"])
        c, d = np.array(self.scaler["output_mean"]), np.array(self.scaler["output_std"])
        print(a.shape, b.shape, c.shape, d.shape) 
        if is_multivars != -1:
            a, b = a[:,:,is_multivars:is_multivars+1], b[:,:,is_multivars:is_multivars+1]
            c, d = c[:,:,is_multivars:is_multivars+1], d[:,:,is_multivars:is_multivars+1]

        if variable == 'input':
            if scaler_method == 'standard':
                feature = feature * b + a
            else:
                feature = feature * (b-a) + a
        elif variable == 'output':
            if scaler_method == 'standard':
                feature = feature * d + c
            else:
                feature = feature * (d-c) + c
        else:
            raise RuntimeError(f"Unknown variable type {variable}")
        return feature

    
    

        
        



