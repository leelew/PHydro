import os
import numpy as np

# NNSE for 1D
def nnse(y_true, y_pred): 
    #rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    #return rmse / (np.nanmax(y_true)-np.nanmin(y_true))
    #return rmse / (np.nanmean(y_true))
    return 1/(2-r2_score(y_true, y_pred))

    r = np.corrcoef(y_true, y_pred)[0,1]
    beta = np.mean(y_pred)/np.mean(y_true)
    gamma = (np.std(y_pred)/np.mean(y_pred)) / \
        (np.std(y_true)/np.mean(y_true))
    return 1/2-(1-np.sqrt((r-1)**2+(beta-1)**2+(gamma-1)**2))



def cal_kge(y_true, y_pred):
    r = pearsonr(y_true, y_pred)[0]
    beta = np.mean(y_pred)/np.mean(y_true)
    gamma = (np.std(y_pred)/np.mean(y_pred)) / \
        (np.std(y_true)/np.mean(y_true))
    return 1-np.sqrt((r-1)**2+(beta-1)**2+(gamma-1)**2)

def cal_mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100



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

def cal_phy_cons(y_pred, aux): #(n,t,f),(n,t,2,f)
    soil_depth = [70, 210, 720] # mm 

    # cal water budget
    swvl_prev = np.multiply(y_pred[:,:,0,:3], soil_depth) # (n,t,4)
    swvl_now = np.multiply(y_pred[:,:,1,:3], soil_depth) # (n,t,4)
    delta_swvl = np.sum(swvl_now-swvl_prev, axis=-1) #(n,t)
    w_b = aux-delta_swvl-y_pred[:,:,-1,-2]-y_pred[:,:,-1,-1] #(n,t)
    return np.nanmean(np.abs(w_b), axis=-1), w_b

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


