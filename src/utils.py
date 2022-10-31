import os
import numpy as np


def unbiased_rmse(y_true, y_pred):
    predmean = np.nanmean(y_pred)
    targetmean = np.nanmean(y_true)
    predanom = y_pred-predmean
    targetanom = y_true - targetmean
    return np.sqrt(np.nanmean((predanom-targetanom)**2))


def init_fold(work_path):
    print("[PHydro] Construct work path in {}".format(work_path))
    if not os.path.exists(work_path+"/input"):
        os.mkdir(work_path+"/input")
    if not os.path.exists(work_path+"/output"):
        os.mkdir(work_path+"/output")
    if not os.path.exists(work_path+"/logs"):
            os.mkdir(work_path+"/logs")
    if not os.path.exists(work_path+"/output/saved_model"):
        os.mkdir(work_path+"/output/saved_model")
    if not os.path.exists(work_path+"/output/forecast"):
        os.mkdir(work_path+"/output/forecast")


# postprocess (return sites to grids)
def site2grid(input, site_lat, site_lon, grid_lat, grid_lon):
    if input.ndim == 2:
        ngrids, nfeat = input.shape
        input_grid = np.full((len(grid_lat), len(grid_lon), nfeat), np.nan)
    elif input.ndim == 3:
        ngrids, nt, nfeat = input.shape
        input_grid = np.full((len(grid_lat), len(grid_lon), nt, nfeat), np.nan)

    for i in range(len(site_lat)):
        lat, lon = site_lat[i], site_lon[i]
        idx_lat = np.where(grid_lat==lat)[0]
        idx_lon = np.where(grid_lon==lon)[0]
        input_grid[idx_lat, idx_lon] = input[i]
    return input_grid


# cal physical consistency
def cal_phy_cons(aux, y_pred):
    # trans predict soil moisture from mm3/mm3 to mm
    soil_depth = [70, 210, 720, 1864.6]
    for j in range(4):
        y_pred[:,:,j] = y_pred[:,:,j]*soil_depth[j]

    phy_cons = np.full((y_pred.shape[0],1),np.nan)
    # for each grid
    for i in range(y_pred.shape[0]):
        precip, swvl_prev = aux[i,:,0], aux[i,:,1]
        swvl = np.nansum(y_pred[i,:,:4], axis=-1)
        phy_cons[i,0] = np.nanmean(precip-(swvl-swvl_prev+y_pred[i,:,4]+y_pred[i,:,5]))
    return phy_cons/(24*3600)
