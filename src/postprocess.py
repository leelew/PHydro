import os

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

from config import parse_args
from utils import cal_phy_cons, site2grid, unbiased_rmse


def postprocess(cfg):
    # load forecast (ngrid, nt, 2, nout)
    path = cfg["outputs_path"]+'forecast/'+cfg["model_name"]+'/'
    y_pred = np.load(path+cfg["model_name"]+'_gd_9km.npy')

    # load observation (ngrid, nt, nout)
    y_test = np.load(cfg["outputs_path"]+'forecast/obs_gd_9km.npy')
    aux_test = np.load(cfg["outputs_path"]+'forecast/aux_gd_9km.npy')

    y_test = y_test[:,-731:,:]
    y_pred = y_pred[:,-731:,:]
    aux_test = aux_test[:,-731:]
    print(y_pred.shape, y_test.shape, aux_test.shape)

    # get shape
    ngrids, nt, _, nfeat = y_pred.shape

    # cal perf
    r2 = np.full((ngrids, nfeat), np.nan)
    urmse = np.full((ngrids, nfeat), np.nan)
    r = np.full((ngrids, nfeat), np.nan)
    for i in range(ngrids):
        for t in range(nfeat):
            print(i)
            if not (np.isnan(y_test[i, :, t]).any()):
                a, b = y_test[i,:,t], y_pred[i,:,-1,t]
                a = np.delete(a, np.isnan(b))
                b = np.delete(b, np.isnan(b))
                urmse[i, t] = unbiased_rmse(a, b)
                r2[i, t] = r2_score(a, b)
                r[i, t] = np.corrcoef(a, b)[0,1]
            else:
                a, b = y_test[i,:,t], y_pred[i,:,-1,t]
                b = np.delete(b, np.isnan(a))
                a = np.delete(a, np.isnan(a))
                urmse[i, t] = unbiased_rmse(a, b)
                r2[i, t] = r2_score(a, b)
                r[i, t] = np.corrcoef(a, b)[0,1]

    # cal overall perf
    print(y_pred.shape, y_test.shape, aux_test.shape)

    # cal physical consistency
    phy_cons, w_b = cal_phy_cons(y_pred, aux_test)
    
    np.save('perf_'+cfg["model_name"]+'.npy',np.stack([r, r2, urmse], axis=0))
    np.save('phy_cons_'+cfg["model_name"]+'.npy',phy_cons)
    np.save('w_b_'+cfg["model_name"]+'.npy',w_b)
    os.system('mv {} {}'.format('*.npy', path))


if __name__ == '__main__':
    cfg = parse_args()
    postprocess(cfg)




               


