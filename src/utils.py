import os
import numpy as np


def make_CoLM_soil_depth():
    # central depth unit (m)
    fs = 0.025
    z = [fs*(np.exp(0.5*(i-0.5))-1) for i in range(1, 11)]
    # thickness of soil (m)
    depth = []
    for i in range(10):
        if i == 0: 
            depth.append(0.5*(z[0]+z[1]))
        elif i == 9:
            depth.append(z[-1]-z[-2])
        else:
            depth.append(0.5*(z[i+1]-z[i-1]))
    # depth at layer interface (m)
    zi = []
    for i in range(10):
        if i == 9:
            zi.append(z[-1]+0.5*depth[-1])
        else:
            zi.append(0.5*(z[i]+z[i+1])) 
    zi = np.array(zi)*100 # m->cm
    return depth, zi


def make_swc_CoLM2EC(wice, wliq):
    """turn CoLM to EC soil setting."""
    depth, zi = make_CoLM_soil_depth()
    # cal h2osoi
    h2osoi = (wliq+wice)/(np.array(depth)*1000)
    # 1.75 4.50 9,05 16.55 28.91 49.29 82.89 138.28 229.61 343.30
    swvl1 = (h2osoi[:,:,:,0]*zi[0]+h2osoi[:,:,:,1]*(zi[1]-zi[0])+h2osoi[:,:,:,2]*(7-zi[1]))/7
    swvl2 = (h2osoi[:,:,:,2]*(zi[2]-7)+h2osoi[:,:,:,3]*(zi[3]-zi[2])+h2osoi[:,:,:,4]*(28-zi[3]))/21
    swvl3 = (h2osoi[:,:,:,4]*(zi[4]-28)+h2osoi[:,:,:,5]*(zi[5]-zi[4])+h2osoi[:,:,:,6]*(zi[6]-zi[5])+h2osoi[:,:,:,7]*(100-zi[6]))/72
    swvl4 = (h2osoi[:,:,:,7]*(zi[7]-100)+h2osoi[:,:,:,8]*(zi[8]-zi[7])+h2osoi[:,:,:,9]*(zi[9]-zi[8]))/(zi[9]-100)
    return swvl1, swvl2, swvl3, swvl4, zi


def unbiased_rmse(y_true, y_pred):
    predmean = np.nanmean(y_pred)
    targetmean = np.nanmean(y_true)
    predanom = y_pred-predmean
    targetanom = y_true - targetmean
    return np.sqrt(np.nanmean((predanom-targetanom)**2))


def r2_score(y_true, y_pred):
    mask = y_true == y_true
    a, b = y_true[mask], y_pred[mask]
    unexplained_error = np.nansum(np.square(a-b))
    total_error = np.nansum(np.square(a - np.nanmean(a)))
    return 1. - unexplained_error/total_error


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


def site2grid(input, site_lat, site_lon, grid_lat, grid_lon):
    # postprocess (return sites to grids)
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


def cal_phy_cons(aux, y_pred, y_true):
    print(aux.shape, y_pred.shape, y_true.shape)
    # cal physical consistency
    soil_depth = [70, 210, 720, 1864.6]
    for j in range(4): 
        y_pred[:,:,j] = y_pred[:,:,j]*soil_depth[j]


    phy_cons = np.full((y_pred.shape[0],1),np.nan)
    for i in range(y_pred.shape[0]):
        t = y_true[i,:,-1]
        print(t.shape)
        swvl = np.nansum(y_pred[i,:,:4], axis=-1)
        m = np.abs(aux[i]-(swvl+y_pred[i,:,4]+y_pred[i,:,5]))
        print(m.shape)
        m = np.delete(m, np.isnan(t))

        phy_cons[i,0] = np.mean(m)
        print(phy_cons[i])
    return phy_cons
