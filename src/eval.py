import numpy as np

from model import (HardMTLModel_v1, HardMTLModel_v2, HardMTLModel_v3, MTLModel_v1, MTLModel_v2,
                   STModel)
from utils import r2_score


# NOTE: To save CPU memory, we don't generate and save test 
#       data (already preprocessed for LSTM type models), we
#       use `batcher` to generate it in inference mode. Another 
#       notation is that we only check the predictability on t. 
#       The prediction on t-1 is only used for ensure balance.
def batcher(x_test, y_test, seq_len, aux_test=None):
    n_t, n_feat = x_test.shape
    _, n_out = y_test.shape
    n = (n_t-seq_len+1)
    x_new = np.zeros((n, seq_len, n_feat))*np.nan
    y_new = np.zeros((n, n_out))*np.nan
    aux_new = np.zeros((n, ))*np.nan
    for i in range(n):
        x_new[i] = x_test[i:i+seq_len]
        y_new[i] = y_test[i+seq_len-1]
        if aux_test is not None:
            aux_new[i] = aux_test[i+seq_len-1]
    if aux_test is None:
        return x_new, y_new
    else:
        return x_new, y_new, aux_new    


# NOTE: We provide prediction (ngrid,nsample,2,nfeat), true
#       (ngrid,nsample,nfeat), aux (ngrid,nsample)
def eval_single(x, y, scaler, cfg):
    # get mean, std (1, ngrid, nfeat)
    mean, std = np.array(scaler["y_mean"]), np.array(scaler["y_std"])  

    y_pred_ens = []
    y_pred_ens_bk = []
    # for each feat
    for j in range(cfg["num_out"]):
        save_folder = cfg["outputs_path"]+"saved_model/" +\
            cfg["model_name"]+'/'+str(j)+'/'

        y_pred_seed = []
        # for each random seed
        for m in range(cfg["num_repeat"]):
            # load model
            model = STModel(cfg)
            model.load_weights(save_folder+str(m)+'/')

            tmp = []
            y_true = []
            # for each grid
            for i in range(x.shape[0]):
                x_new, y_new = batcher(x[i], y[i], cfg["seq_len"])
                pred = model(x_new)
                pred = pred*std[:, i, j]+mean[:, i, j] #(nsample,2,1)
                tmp.append(pred)
                y_true.append(y_new)
            y_pred_seed.append(np.stack(tmp, axis=0))  # (ngrid, nsample, 2, 1)

        # cal ensemble mean of diff exps (ngrid, nsample, 1)
        y_pred_seed = np.concatenate(y_pred_seed, axis=-1)
        y_pred_seed_bk = y_pred_seed
        y_pred_seed = np.nanmean(y_pred_seed, axis=-1, keepdims=True) #(ngrid, nsample, 2,1)
        y_true = np.stack(y_true, axis=0) #(ngrid, nsample, 1)

        # log
        r2_ens = []
        for i in range(y_pred_seed.shape[0]):
            r2_ens.append(r2_score(y_true[i, :, j], y_pred_seed[i, :, 1, 0]))
        print('\033[1;31m%s\033[0m' %
              "Var {} Median NSE {:.3f}".format(j, np.nanmedian(r2_ens)))

        y_pred_ens.append(y_pred_seed)
        y_pred_ens_bk.append(y_pred_seed_bk)
    return np.concatenate(y_pred_ens, axis=-1), np.stack(y_pred_ens_bk, axis=-2)


def eval_multi(x, y, aux, scaler, cfg):
    # get mean, std (1, ngrid, 6)
    mean, std = np.array(scaler["y_mean"]), np.array(scaler["y_std"])  

    y_pred_ens = []
    save_folder = cfg["outputs_path"]+"saved_model/" +\
        cfg["model_name"]+'/'

    # for each random seed
    for m in range(cfg["num_repeat"]):
        # load model
        if cfg["model_name"] in ['multi_tasks_v1', 'multi_tasks_v3', 'soft_multi_tasks']:
            model = MTLModel_v1(cfg)
        elif cfg["model_name"] in ["multi_tasks_v2"]:
            model = MTLModel_v2(cfg)
        elif cfg["model_name"] in ['hard_multi_tasks_v1']:
            model = HardMTLModel_v1(cfg)
        elif cfg["model_name"] in ["hard_multi_tasks_v2"]:
            model = HardMTLModel_v2(cfg)
        elif cfg["model_name"] in ["hard_multi_tasks_v3"]:
            model = HardMTLModel_v3(cfg)
        model.load_weights(save_folder+str(m)+'/')

        tmp = []
        y_true = []
        aux_true = []
        # for each grid
        for i in range(x.shape[0]):
            print(m, i)
            # generate batch 
            x_new, y_new, aux_new = batcher(x[i], y[i], cfg["seq_len"], aux[i])

            if cfg["model_name"] in ['multi_tasks_v1', \
                'multi_tasks_v2', 'multi_tasks_v3','soft_multi_tasks']:
                pred = model(x_new, training=False)
            elif cfg["model_name"] in ['hard_multi_tasks_v3']:
                _, pred = model(x_new, aux_new, mean[0,i], std[0,i], training=False)
            else:
                pred = model(x_new, aux_new, mean[0,i], std[0,i], training=False)
            pred = pred*std[:, i]+mean[:, i]
            tmp.append(pred)
            y_true.append(y_new)
            aux_true.append(aux_new)
        y_pred_ens.append(np.stack(tmp, axis=0))  # (ngrid, nsample, 6)
    # cal ensemble mean of diff exps (ngrid, nsample, 6)
    y_pred_ens = np.stack(y_pred_ens, axis=-1)
    y_pred_ens_bk = y_pred_ens
    y_pred_ens = np.nanmean(y_pred_ens, axis=-1)
    y_true = np.stack(y_true, axis=0)
    aux_true = np.stack(aux_true, axis=0)

    # log
    for j in range(cfg["num_out"]):
        r2_ens = []
        for i in range(y_pred_ens.shape[0]):
            r2_ens.append(r2_score(y_true[i, :, j], y_pred_ens[i, :, 1, j]))
        print('\033[1;31m%s\033[0m' %
              "Var {} Median NSE {:.3f}".format(j, np.nanmedian(r2_ens)))

    return y_pred_ens, y_true, aux_true, y_pred_ens_bk



