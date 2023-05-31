import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmcrameri import cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from mpl_toolkits.basemap import Basemap
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from config import parse_args
from utils import unbiased_rmse, cal_phy_cons, nnse


# parameters
cfg = parse_args()
path = cfg["outputs_path"]+'forecast/'

# load single-task vs multi-task
#pred = np.load(path+"hard_multi_tasks_v3/hard_multi_tasks_v3_ens_gd_9km.npy")[:,-731:,1]
#pred = np.load(path+"hard_multi_tasks_v3_alpha_0.1_data_1%/hard_multi_tasks_v3_ens_gd_9km.npy")[:,-731:,1]
#pred = np.load(path+"soft_multi_tasks/soft_multi_tasks_ens_gd_9km.npy")[:,-731:,1]
#pred = np.load(path+"soft_multi_tasks_alpha_0.1_data_1%/soft_multi_tasks_ens_gd_9km.npy")[:,-731:,1]
#pred = np.load(path+"multi_tasks_v1/multi_tasks_v1_ens_gd_9km.npy")[:,-731:,1]
#pred = np.load(path+"multi_tasks_v1_factor_5/multi_tasks_v1_ens_gd_9km.npy")[:,-731:,1]

#pred = np.load(path+"multi_tasks_v1_factor_5_data_10%/multi_tasks_v1_ens_gd_9km.npy")[:,-731:,1]
#pred = np.load(path+"multi_tasks_v1_factor_2.5/multi_tasks_v1_ens_gd_9km.npy")[:,-731:,1]
#pred = np.load(path+"multi_tasks_v1_factor_2.5_bk/multi_tasks_v1_ens_gd_9km.npy")[:,-731:,1]
#pred = np.load(path+"multi_tasks_v1_factor_5_bkbk/multi_tasks_v1_ens_gd_9km.npy")[:,-731:,1]

#pred = np.load(path+"single_task/single_task_ens_gd_9km.npy")[:,-731:,1]
#pred = np.load(path+"expA/single_task/single_task_ens_gd_9km.npy")[:,-731:,1]

#pred = np.load(path+"single_task_bk/single_task_ens_gd_9km.npy")[:,-731:,1]
#
#pred = np.load("/data/lilu/PHydro_era/output_old/forecast/hard_multi_tasks_v3_0.1_train_0.1_type1/hard_multi_tasks_v3_ens_gd_9km.npy")[:,-731:,1]
#pred = np.load(path+"hard_multi_tasks_v3_alpha_0.5_data_1%_bk/hard_multi_tasks_v3_ens_gd_9km.npy")[:,-731:,1]
#pred = np.load(path+"hard_multi_tasks_v3_alpha_0.5_data_1%/hard_multi_tasks_v3_ens_gd_9km.npy")[:,-731:,1]

#pred = np.load(path+"hard_multi_tasks_v3_alpha_0.1_data_1%/hard_multi_tasks_v3_ens_gd_9km.npy")[:,-731:,1]
pred = np.load(path+"expB/hard_multi_tasks_v3_alpha_0.1_data_1%/hard_multi_tasks_v3_ens_gd_9km.npy")[:,-731:,1]



# load test (ngrid,nt,nfeat)
obs = np.load(path+'obs_gd_9km.npy')[:,-731:]
aux = np.load(path+'aux_gd_9km.npy')[:,-731:]

# cal perf 
nrmse_mean_group1 = np.full((200,5,5), np.nan)
for i in range(200):
    for m in range(5):
        for k in range(5):
            nrmse_mean_group1[i,m,k] = nnse(obs[i,:,m],pred[i,:,m,k])

print(np.nanmean(nrmse_mean_group1, axis=(0,1)))

