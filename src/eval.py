import numpy as np
from utils import r2_score
from model import VanillaLSTM, MTLLSTM, MTLHardLSTM_v1, MTLHardLSTM_v2


def eval_single(x, y, scaler, cfg):
    # get mean, std (1, ngrid, 6)
    mean, std = np.array(scaler["y_mean"]), np.array(scaler["y_std"])  

    y_pred_ens = []
    # for each feat
    for j in range(cfg["num_out"]):
        save_folder = cfg["outputs_path"]+"saved_model/" +\
            cfg["model_name"]+'/'+str(j)+'/'

        y_pred_seed = []
        # for each random seed
        for m in range(cfg["num_repeat"]):
            # load model
            model = VanillaLSTM(cfg)
            model.load_weights(save_folder+str(m)+'/')

            tmp = []
            # for each grid
            for i in range(x.shape[0]):
                pred = model(x[i])
                pred = pred*std[:, i, j]+mean[:, i, j]
                tmp.append(pred)
            y_pred_seed.append(np.stack(tmp, axis=0))  # (ngrid, nsample, 1)
        # cal ensemble mean of diff exps (ngrid, nsample, 1)
        y_pred_seed = np.concatenate(y_pred_seed, axis=-1)
        y_pred_seed = np.nanmean(y_pred_seed, axis=-1, keepdims=True)

        # log
        r2_ens = []
        for i in range(y_pred_seed.shape[0]):
            r2_ens.append(r2_score(y[i, :, j], y_pred_seed[i, :, 0]))
        print('\033[1;31m%s\033[0m' %
              "Var {} Median NSE {:.3f}".format(j, np.nanmedian(r2_ens)))

        y_pred_ens.append(y_pred_seed)
    return np.concatenate(y_pred_ens, axis=-1)


def eval_multi(x, y, aux, scaler, cfg):
    #FIXME: still have some bugs
    # get mean, std
    mean, std = np.array(scaler["y_mean"]), np.array(scaler["y_std"])  # (1, ngrid, 6)

    y_pred_ens = []
    save_folder = cfg["outputs_path"]+"saved_model/" +\
        cfg["model_name"]+'/'

    # for each random seed
    for m in range(cfg["num_repeat"]):
        # load model
        if cfg["model_name"] in ['multi_tasks', 'soft_multi_tasks']:
            model = MTLLSTM(cfg)
        elif cfg["model_name"] in ['hard_multi_tasks_v2', 'hard_multi_tasks_v3']:
            model = MTLHardLSTM_v2(cfg)
        elif cfg["model_name"] == 'hard_multi_tasks_v1':
            model = MTLHardLSTM_v1(cfg)
        model.load_weights(save_folder+str(m)+'/')

        tmp = []
        # for each grid
        for i in range(x.shape[0]):
            if cfg["model_name"] in ['single_task', 'multi_tasks', 'soft_multi_tasks']:
                pred = model(x[i], training=False)
            else:
                pred = model(x[i], aux[i], mean[0,i], std[0,i], training=False)
            pred = pred*std[:, i]+mean[:, i]
            tmp.append(pred)
        y_pred_ens.append(np.stack(tmp, axis=0))  # (ngrid, nsample, 6)
    # cal ensemble mean of diff exps (ngrid, nsample, 6)
    y_pred_ens = np.stack(y_pred_ens, axis=-1)
    y_pred_ens = np.nanmean(y_pred_ens, axis=-1)

    # log
    r2_ens = []
    for j in range(cfg["num_out"]):
        for i in range(y_pred_ens.shape[0]):
            r2_ens.append(r2_score(y[i, :, j], y_pred_ens[i, :, j]))
        print('\033[1;31m%s\033[0m' %
              "Var {} Median NSE {:.3f}".format(j, np.nanmedian(r2_ens)))

    return y_pred_ens

