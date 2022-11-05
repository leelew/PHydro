import os
import numpy as np
from sklearn.metrics import r2_score

from utils import site2grid, cal_phy_cons, unbiased_rmse
from config import parse_args


def postprocess(cfg):
    # load forecast and observation (ngrid, nt, nout)
    path = cfg["outputs_path"]+'forecast/'+cfg["model_name"]+'/'
    y_pred = np.load(path+cfg["model_name"]+'_gd_9km.npy')
    path = cfg["inputs_path"]
    y_test = np.load(path+'y_test.npy')
    aux_test = np.load(path+'z_test.npy')
    print(y_pred.shape, y_test.shape, aux_test.shape)

    # get shape
    ngrids, nt, nfeat = y_pred.shape

    # load site and grid lon/lat
    attr = np.load(path+"coord_gd_9km.npy")
    lon_min, lat_min = np.nanmin(attr, axis=0)
    lon_max, lat_max = np.nanmax(attr, axis=0)
    ease_9km_grids = np.load(path+"coord_global_9km.npy") #(1800, 3600, 2)
    lon, lat = ease_9km_grids[0,:,0], ease_9km_grids[:,0,1]
    idx_lat = np.where((lat>=lat_min) & (lat<=lat_max))
    idx_lon = np.where((lon>=lon_min) & (lon<=lon_max))
    lon_gd = lon[idx_lon]
    lat_gd = lat[idx_lat]
    grid_lon, grid_lat = np.meshgrid(lon_gd, lat_gd)

    # cal perf
    r2 = np.full((ngrids, nfeat), np.nan)
    urmse = np.full((ngrids, nfeat), np.nan)
    for i in range(ngrids):
        for t in range(nfeat):
            if not (np.isnan(y_test[i, :, t]).any()):
                urmse[i, t] = unbiased_rmse(y_test[i, :, t], y_pred[i, :, t])
                r2[i, t] = r2_score(y_test[i, :, t], y_pred[i, :, t])
            else:
                a, b = y_test[i,:,t], y_pred[i,:,t]
                b = np.delete(b, np.isnan(a))
                a = np.delete(a, np.isnan(a))
                urmse[i, t] = unbiased_rmse(a, b)
                r2[i, t] = r2_score(a, b)

    # cal physical consistency
    phy_cons = cal_phy_cons(aux_test, y_pred, y_test)
    
    # turn r2, urmse, physical consist, y_pred, y_test to grids
    phy_cons_grid = site2grid(phy_cons, attr[:,1], attr[:,0], lat_gd, lon_gd)
    r2_grid = site2grid(r2, attr[:,1], attr[:,0], lat_gd, lon_gd)
    urmse_grid = site2grid(urmse, attr[:,1], attr[:,0], lat_gd, lon_gd)
    y_pred_ = site2grid(y_pred, attr[:,1], attr[:,0], lat_gd, lon_gd)
    y_test_ = site2grid(y_test, attr[:,1], attr[:,0], lat_gd, lon_gd)

    # save
    np.save('r2_'+cfg["model_name"]+'.npy', r2_grid)
    np.save('urmse_'+cfg["model_name"]+'.npy', urmse_grid)
    np.save('y_pred_'+cfg["model_name"]+'.npy', y_pred_)
    np.save('y_test_'+cfg["model_name"]+'.npy', y_test_)
    np.save('phy_cons_'+cfg["model_name"]+'.npy', phy_cons_grid)
    path = cfg["outputs_path"]+'forecast/'+cfg["model_name"]+'/'
    print(path)
    os.system('mv {} {}'.format('*.npy', path))


if __name__ == '__main__':
    cfg = parse_args()
    postprocess(cfg)




               


