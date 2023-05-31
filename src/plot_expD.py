import numpy as np
import matplotlib.pyplot as plt

from config import parse_args
from utils import cal_phy_cons, nnse


# parameters
cfg = parse_args()
path = cfg["outputs_path"]+'forecast/'

# load simulate
pred_soft_alpha_0p1 = np.load(path+"expB/soft_multi_tasks_alpha_0.1_data_100%/soft_multi_tasks_ens_gd_9km.npy")
pred_soft_alpha_0p5 = np.load(path+"expD/soft_multi_tasks_alpha_0.5_data_100%/soft_multi_tasks_ens_gd_9km.npy")
pred_hybrid_alpha_0p1 = np.load(path+"expB/hard_multi_tasks_v3_alpha_0.1_data_100%/hard_multi_tasks_v3_ens_gd_9km.npy") 
pred_hybrid_alpha_0p5 = np.load(path+"expD/hard_multi_tasks_v3_alpha_0.5_data_100%/hard_multi_tasks_v3_ens_gd_9km.npy")
pred_group = np.stack([pred_soft_alpha_0p1,pred_soft_alpha_0p5,pred_hybrid_alpha_0p1,pred_hybrid_alpha_0p5], axis=-1)[:,-731:,1] 

# load test (ngrid,nt,nfeat)
obs = np.load(path+'obs_gd_9km.npy')[:,-731:]
aux = np.load(path+'aux_gd_9km.npy')[:,-731:]

# get shape
ngrid, nt, nfeat, nens, nmodel = pred_group.shape

# cal general performance
nnse_group = np.full((ngrid,nfeat,nens,nmodel), np.nan)
r_group = np.full((ngrid,nfeat,nens,nmodel), np.nan)
for i in range(ngrid):
    for m in range(nfeat):
        for j in range(nens):
            for k in range(nmodel):
                nnse_group[i,m,j,k] = nnse(obs[i,:,m],pred_group[i,:,m,j,k])
                r_group[i,m,j,k] = np.corrcoef(obs[i,:,m],pred_group[i,:,m,j,k])[0,1]

# load scv 100% data alpha 0.1
pred_soft_cv_0_alpha_0p1 = np.load(path+"expC/soft_multi_tasks_alpha_0.1_data_100%_scv_0/soft_multi_tasks_ens_gd_9km.npy")
pred_soft_cv_0_alpha_0p5 = np.load(path+"expD/soft_multi_tasks_alpha_0.5_data_100%_scv_0/soft_multi_tasks_ens_gd_9km.npy")
pred_hybrid_cv_0_alpha_0p1 = np.load(path+"expC/hard_multi_tasks_v3_alpha_0.1_data_100%_scv_0/hard_multi_tasks_v3_ens_gd_9km.npy")
pred_hybrid_cv_0_alpha_0p5 = np.load(path+"expD/hard_multi_tasks_v3_alpha_0.5_data_100%_scv_0/hard_multi_tasks_v3_ens_gd_9km.npy")

pred_soft_cv_1_alpha_0p1 = np.load(path+"expC/soft_multi_tasks_alpha_0.1_data_100%_scv_1/soft_multi_tasks_ens_gd_9km.npy")
pred_soft_cv_1_alpha_0p5 = np.load(path+"expD/soft_multi_tasks_alpha_0.5_data_100%_scv_1/soft_multi_tasks_ens_gd_9km.npy")
pred_hybrid_cv_1_alpha_0p1 = np.load(path+"expC/hard_multi_tasks_v3_alpha_0.1_data_100%_scv_1/hard_multi_tasks_v3_ens_gd_9km.npy")
pred_hybrid_cv_1_alpha_0p5 = np.load(path+"expD/hard_multi_tasks_v3_alpha_0.5_data_100%_scv_1/hard_multi_tasks_v3_ens_gd_9km.npy")

pred_soft_cv_2_alpha_0p1 = np.load(path+"expC/soft_multi_tasks_alpha_0.1_data_100%_scv_2/soft_multi_tasks_ens_gd_9km.npy")
pred_soft_cv_2_alpha_0p5 = np.load(path+"expD/soft_multi_tasks_alpha_0.5_data_100%_scv_2/soft_multi_tasks_ens_gd_9km.npy")
pred_hybrid_cv_2_alpha_0p1 = np.load(path+"expC/hard_multi_tasks_v3_alpha_0.1_data_100%_scv_2/hard_multi_tasks_v3_ens_gd_9km.npy")
pred_hybrid_cv_2_alpha_0p5 = np.load(path+"expD/hard_multi_tasks_v3_alpha_0.5_data_100%_scv_2/hard_multi_tasks_v3_ens_gd_9km.npy")

pred_soft_cv_3_alpha_0p1 = np.load(path+"expC/soft_multi_tasks_alpha_0.1_data_100%_scv_3/soft_multi_tasks_ens_gd_9km.npy")
pred_soft_cv_3_alpha_0p5 = np.load(path+"expD/soft_multi_tasks_alpha_0.5_data_100%_scv_3/soft_multi_tasks_ens_gd_9km.npy")
pred_hybrid_cv_3_alpha_0p1 = np.load(path+"expC/hard_multi_tasks_v3_alpha_0.1_data_100%_scv_3/hard_multi_tasks_v3_ens_gd_9km.npy")
pred_hybrid_cv_3_alpha_0p5 = np.load(path+"expD/hard_multi_tasks_v3_alpha_0.5_data_100%_scv_3/hard_multi_tasks_v3_ens_gd_9km.npy")

pred_group_scv_0 = np.stack([pred_soft_cv_0_alpha_0p1,pred_soft_cv_0_alpha_0p5,pred_hybrid_cv_0_alpha_0p1,pred_hybrid_cv_0_alpha_0p5], axis=-1)[:,-731:,1] 
pred_group_scv_1 = np.stack([pred_soft_cv_1_alpha_0p1,pred_soft_cv_1_alpha_0p5,pred_hybrid_cv_1_alpha_0p1,pred_hybrid_cv_1_alpha_0p5], axis=-1)[:,-731:,1] 
pred_group_scv_2 = np.stack([pred_soft_cv_2_alpha_0p1,pred_soft_cv_2_alpha_0p5,pred_hybrid_cv_2_alpha_0p1,pred_hybrid_cv_2_alpha_0p5], axis=-1)[:,-731:,1] 
pred_group_scv_3 = np.stack([pred_soft_cv_3_alpha_0p1,pred_soft_cv_3_alpha_0p5,pred_hybrid_cv_3_alpha_0p1,pred_hybrid_cv_3_alpha_0p5], axis=-1)[:,-731:,1] 

# load tcv 100% data alpha 0.1
pred_soft_tcv_0_alpha_0p1 = np.load(path+"expC/soft_multi_tasks_alpha_0.1_data_100%_tcv_summer/soft_multi_tasks_ens_gd_9km.npy")
pred_soft_tcv_0_alpha_0p5 = np.load(path+"expD/soft_multi_tasks_alpha_0.5_data_100%_tcv_summer/soft_multi_tasks_ens_gd_9km.npy")
pred_hybrid_tcv_0_alpha_0p1 = np.load(path+"expC/hard_multi_tasks_v3_alpha_0.1_data_100%_tcv_summer/hard_multi_tasks_v3_ens_gd_9km.npy")
pred_hybrid_tcv_0_alpha_0p5 = np.load(path+"expD/hard_multi_tasks_v3_alpha_0.5_data_100%_tcv_summer/hard_multi_tasks_v3_ens_gd_9km.npy")
pred_group_tcv_0 = np.stack([pred_soft_tcv_0_alpha_0p1,pred_soft_tcv_0_alpha_0p5,pred_hybrid_tcv_0_alpha_0p1,pred_hybrid_tcv_0_alpha_0p5], axis=-1)[:,-731:,1] 

# get shape
ngrid, nt, nfeat, nens, nmodel = pred_group_scv_0.shape

# cal general performance
nnse_group_scv_0 = np.full((50,nfeat,nens,nmodel), np.nan)
r_group_scv_0 = np.full((50,nfeat,nens,nmodel), np.nan)
for i in range(50):
    for m in range(nfeat):
        for j in range(nens):
            for k in range(nmodel):
                nnse_group_scv_0[i,m,j,k] = nnse(obs[i,:,m],pred_group_scv_0[i,:,m,j,k])
                r_group_scv_0[i,m,j,k] = np.corrcoef(obs[i,:,m],pred_group_scv_0[i,:,m,j,k])[0,1]

nnse_group_scv_1 = np.full((50,nfeat,nens,nmodel), np.nan)
r_group_scv_1 = np.full((50,nfeat,nens,nmodel), np.nan)
for i in range(50,100):
    for m in range(nfeat):
        for j in range(nens):
            for k in range(nmodel):
                nnse_group_scv_1[i-50,m,j,k] = nnse(obs[i,:,m],pred_group_scv_1[i,:,m,j,k])
                r_group_scv_1[i-50,m,j,k] = np.corrcoef(obs[i,:,m],pred_group_scv_1[i,:,m,j,k])[0,1]
             
nnse_group_scv_2 = np.full((50,nfeat,nens,nmodel), np.nan)
r_group_scv_2 = np.full((50,nfeat,nens,nmodel), np.nan)
for i in range(100,150):
    for m in range(nfeat):
        for j in range(nens):
            for k in range(nmodel):
                nnse_group_scv_2[i-100,m,j,k] = nnse(obs[i,:,m],pred_group_scv_2[i,:,m,j,k])
                r_group_scv_2[i-100,m,j,k] = np.corrcoef(obs[i,:,m],pred_group_scv_2[i,:,m,j,k])[0,1]
         
nnse_group_scv_3 = np.full((50,nfeat,nens,nmodel), np.nan)
r_group_scv_3 = np.full((50,nfeat,nens,nmodel), np.nan)
for i in range(150,200):
    for m in range(nfeat):
        for j in range(nens):
            for k in range(nmodel):
                nnse_group_scv_3[i-150,m,j,k] = nnse(obs[i,:,m],pred_group_scv_3[i,:,m,j,k])
                r_group_scv_3[i-150,m,j,k] = np.corrcoef(obs[i,:,m],pred_group_scv_3[i,:,m,j,k])[0,1]
     
nnse_group_tcv_0 = np.full((ngrid,nfeat,nens,nmodel), np.nan)
r_group_tcv_0 = np.full((ngrid,nfeat,nens,nmodel), np.nan)
idx = []
idx.append(np.arange(172,264))
idx.append(np.arange(172+365,264+365))
idx = np.concatenate(idx,axis=0)
for i in range(ngrid):
    for m in range(nfeat):
        for j in range(nens):
            for k in range(nmodel):
                nnse_group_tcv_0[i,m,j,k] = nnse(obs[i,:,m],pred_group_tcv_0[i,:,m,j,k])
                r_group_tcv_0[i,m,j,k] = np.corrcoef(obs[i,:,m],pred_group_tcv_0[i,:,m,j,k])[0,1]
    


# --------------------------------------------------------------------------------------
# Fig 13.
# --------------------------------------------------------------------------------------
fig = plt.figure(figsize=(9.9,13.2))
kk = 0.03
ms = 9

mean_rmse = np.nanmean(nnse_group_scv_0, axis=(0,1)) #(5,12)
mean_rmse_mean = np.nanmean(mean_rmse, axis=(0)) #(12)
mean_rmse_min = np.nanmin(mean_rmse,axis=0) #(12)
mean_rmse_max = np.nanmax(mean_rmse,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)

ax = plt.subplot(431)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='red',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='blue',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=7,  marker='h', mfc='w', color='blue',capsize=5)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.3)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])

mean_rmse = np.nanmean(nnse_group_scv_1, axis=(0,1)) #(5,12)
mean_rmse_mean = np.nanmean(mean_rmse, axis=(0)) #(12)
mean_rmse_min = np.nanmin(mean_rmse,axis=0) #(12)
mean_rmse_max = np.nanmax(mean_rmse,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)

ax = plt.subplot(432)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='red',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='blue',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=7,  marker='h', mfc='w', color='blue',capsize=5)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.3)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])

mean_rmse = np.nanmean(nnse_group_scv_2, axis=(0,1)) #(5,12)
mean_rmse_mean = np.nanmean(mean_rmse, axis=(0)) #(12)
mean_rmse_min = np.nanmin(mean_rmse,axis=0) #(12)
mean_rmse_max = np.nanmax(mean_rmse,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)

ax = plt.subplot(434)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='red',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='blue',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=7,  marker='h', mfc='w', color='blue',capsize=5)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.3)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])

mean_rmse = np.nanmean(nnse_group_scv_3, axis=(0,1)) #(5,12)
mean_rmse_mean = np.nanmean(mean_rmse, axis=(0)) #(12)
mean_rmse_min = np.nanmin(mean_rmse,axis=0) #(12)
mean_rmse_max = np.nanmax(mean_rmse,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)

ax = plt.subplot(435)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='red',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='blue',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=7,  marker='h', mfc='w', color='blue',capsize=5)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.3)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])

mean_rmse = np.nanmean(nnse_group_tcv_0, axis=(0,1)) #(5,12)
mean_rmse_mean = np.nanmean(mean_rmse, axis=(0)) #(12)
mean_rmse_min = np.nanmin(mean_rmse,axis=0) #(12)
mean_rmse_max = np.nanmax(mean_rmse,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)

ax = plt.subplot(436)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='red',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='blue',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=7,  marker='h', mfc='w', color='blue',capsize=5)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.3)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])

m1 = nnse_group_scv_0 - nnse_group[:50,:,:,-4:]
m2 = nnse_group_scv_1 - nnse_group[50:100,:,:,-4:]
m3 = nnse_group_scv_2 - nnse_group[100:150,:,:,-4:]
m4 = nnse_group_scv_3 - nnse_group[150:200,:,:,-4:]

mean_rmse = np.nanmean(m1, axis=(0,1)) #(5,12)
mean_rmse_mean = np.nanmean(mean_rmse, axis=(0)) #(12)
mean_rmse_min = np.nanmin(mean_rmse,axis=0) #(12)
mean_rmse_max = np.nanmax(mean_rmse,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)

ax = plt.subplot(4,3,7)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='red',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='blue',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=7,  marker='h', mfc='w', color='blue',capsize=5)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.3)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])

mean_rmse = np.nanmean(m2, axis=(0,1)) #(5,12)
mean_rmse_mean = np.nanmean(mean_rmse, axis=(0)) #(12)
mean_rmse_min = np.nanmin(mean_rmse,axis=0) #(12)
mean_rmse_max = np.nanmax(mean_rmse,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)

ax = plt.subplot(4,3,8)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='red',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='blue',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=7,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.3)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])

mean_rmse = np.nanmean(m3, axis=(0,1)) #(5,12)
mean_rmse_mean = np.nanmean(mean_rmse, axis=(0)) #(12)
mean_rmse_min = np.nanmin(mean_rmse,axis=0) #(12)
mean_rmse_max = np.nanmax(mean_rmse,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)

ax = plt.subplot(4,3,10)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='red',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='blue',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=7,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.3)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])

mean_rmse = np.nanmean(m4, axis=(0,1)) #(5,12)
mean_rmse_mean = np.nanmean(mean_rmse, axis=(0)) #(12)
mean_rmse_min = np.nanmin(mean_rmse,axis=0) #(12)
mean_rmse_max = np.nanmax(mean_rmse,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)

ax = plt.subplot(4,3,11)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='red',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='blue',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=7,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.3)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])

m1 = nnse_group_tcv_0 - nnse_group[:,:,:,-4:]
mean_rmse = np.nanmean(m1, axis=(0,1)) #(5,12)
mean_rmse_mean = np.nanmean(mean_rmse, axis=(0)) #(12)
mean_rmse_min = np.nanmin(mean_rmse,axis=0) #(12)
mean_rmse_max = np.nanmax(mean_rmse,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)

ax = plt.subplot(4,3,12)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='red',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='blue',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=7,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.3)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.savefig('figure11.pdf')
print('Figure 11 completed!')



# load soft with different alpha
pred_soft_0p1 = np.load(path+"expB/soft_multi_tasks_alpha_0.1_data_100%/soft_multi_tasks_ens_gd_9km.npy")
pred_soft_0p2 = np.load(path+"expD/soft_multi_tasks_alpha_0.2/soft_multi_tasks_ens_gd_9km.npy")
pred_soft_0p5 = np.load(path+"expD/soft_multi_tasks_alpha_0.5_data_100%/soft_multi_tasks_ens_gd_9km.npy")
pred_soft_0p8 = np.load(path+"expD/soft_multi_tasks_alpha_0.8/soft_multi_tasks_ens_gd_9km.npy")
pred_group_soft = np.stack([pred_soft_0p1,pred_soft_0p2,pred_soft_0p5,pred_soft_0p8], axis=-1)[:,-731:,1] 
pred_group_soft_bk = np.stack([pred_soft_0p1,pred_soft_0p2,pred_soft_0p5,pred_soft_0p8], axis=-1)[:,-731:] 

# get shape
ngrid, nt, nfeat, nens, nmodel = pred_group.shape

# cal general performance
nnse_group_soft = np.full((ngrid,nfeat,nens,nmodel), np.nan)
r_group_soft = np.full((ngrid,nfeat,nens,nmodel), np.nan)
for i in range(ngrid):
    for m in range(nfeat):
        for j in range(nens):
            for k in range(nmodel):
                nnse_group_soft[i,m,j,k] = nnse(obs[i,:,m],pred_group_soft[i,:,m,j,k])
                r_group_soft[i,m,j,k] = np.corrcoef(obs[i,:,m],pred_group_soft[i,:,m,j,k])[0,1]

# cal physical
phy_group_soft = np.full((ngrid,nens,nmodel), np.nan)
for j in range(nens):
    for k in range(nmodel):
        phy_group_soft[:,j,k],_ = cal_phy_cons(pred_group_soft_bk[:,:,:,:,j,k],aux)

plt.figure()
tmp1 = np.nanmean(nnse_group_soft, axis=(0,1,2))
tmp2 = np.nanmean(r_group_soft, axis=(0,1,2))
tmp3 = np.nanmean(phy_group_soft, axis=(0,1))
ax = plt.subplot(221)
plt.plot(np.arange(0,tmp1.shape[0]*0.5,0.5),tmp1,c='gray',marker='o')
ax1 = ax.twinx()
plt.plot(np.arange(0,tmp1.shape[0]*0.5,0.5),tmp3,c='black',marker='o')
ax1.set_xticks([0,0.5,1,1.5])
ax1.set_xticklabels(['0.1','0.2','0.5','0.8'])
ax1.set_xlabel('Lambda')
plt.savefig('figureS6.pdf')
print('Figure S6 completed!')


# load soft with different alpha
pred_soft_0p1 = np.load(path+"expB/soft_multi_tasks_alpha_0.1_data_1%/soft_multi_tasks_ens_gd_9km.npy")
pred_soft_0p5 = np.load(path+"expD/soft_multi_tasks_alpha_0.5_data_1%/soft_multi_tasks_ens_gd_9km.npy")
pred_hybrid_0p1 = np.load(path+"expB/hard_multi_tasks_v3_alpha_0.1_data_1%/hard_multi_tasks_v3_ens_gd_9km.npy")
pred_hybrid_0p5 = np.load(path+"expD/hard_multi_tasks_v3_alpha_0.5_data_1%/hard_multi_tasks_v3_ens_gd_9km.npy")
pred_group_0p01 = np.stack([pred_soft_0p1,pred_soft_0p5,pred_hybrid_0p1,pred_hybrid_0p5], axis=-1)[:,-731:,1] 

# cal general performance
nnse_group_0p01 = np.full((ngrid,nfeat,nens,nmodel), np.nan)
r_group_0p01 = np.full((ngrid,nfeat,nens,nmodel), np.nan)
for i in range(ngrid):
    for m in range(nfeat):
        for j in range(nens):
            for k in range(nmodel):
                nnse_group_0p01[i,m,j,k] = nnse(obs[i,:,m],pred_group_0p01[i,:,m,j,k])
                r_group_0p01[i,m,j,k] = np.corrcoef(obs[i,:,m],pred_group_0p01[i,:,m,j,k])[0,1]


plt.figure(figsize=(6,3))
a5 = np.nanmean(nnse_group, axis=(0,1))
mean_rmse_mean = np.nanmean(a5, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a5,axis=0) #(12)
mean_rmse_max = np.nanmax(a5,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(1,2,1)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='red',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2],ms=ms, marker='o', mfc='w', color='red',capsize=5)
ax.errorbar(0.10,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='blue',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms, marker='D', mfc='w', color='blue',capsize=5)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title("100% training data", fontsize=13)

a6 = np.nanmean(nnse_group_0p01, axis=(0,1))
mean_rmse_mean = np.nanmean(a6, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a6,axis=0) #(12)
mean_rmse_max = np.nanmax(a6,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(1,2,2)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='red',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2],ms=ms, marker='o', mfc='w', color='red',capsize=5)
ax.errorbar(0.10,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='blue',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms, marker='D', mfc='w', color='blue',capsize=5)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title("1% training data", fontsize=13)

plt.savefig('figureS7.pdf')
print('Figure S7 completed!')



# load hard compare
pred_hard_v2_0 = np.load(path+"hard_multi_tasks_v2_0/hard_multi_tasks_v2_ens_gd_9km.npy")
pred_hard_v2_1 = np.load(path+"hard_multi_tasks_v2_1/hard_multi_tasks_v2_ens_gd_9km.npy")
pred_hard_v2_2 = np.load(path+"hard_multi_tasks_v2_2/hard_multi_tasks_v2_ens_gd_9km.npy")
pred_hard_v1_type1 = np.load(path+"hard_multi_tasks_v1_train_1_type1/hard_multi_tasks_v1_ens_gd_9km.npy")
pred_hard_v1_type3 = np.load(path+"hard_multi_tasks_v1_train_1_type3/hard_multi_tasks_v1_ens_gd_9km.npy")
pred_group7 = np.stack([pred_hard_v2_0, pred_hard_v2_1,pred_hard_v2_2,pred_hard_v1_type3,pred_hard_v1_type1], axis=-1)[:,-731:,1] 

# group 7: un,soft,hard,hybrid (ngrid,nfeat,nens,nmodel)
nrmse_group7 = np.full((200,5,5,pred_group7.shape[-1]), np.nan)
r_group7 = np.full((200,5,5,pred_group7.shape[-1]), np.nan)
for i in range(200):
    for m in range(5):
        for j in range(5):
            for k in range(pred_group7.shape[-1]):
                nrmse_group7[i,m,j,k] = nnse(obs[i,:,m],pred_group7[i,:,m,j,k])
                r_group7[i,m,j,k] = np.corrcoef(obs[i,:,m],pred_group7[i,:,m,j,k])[0,1]


# --------------------------------------------------------------------------------------
# Fig S8. 
# --------------------------------------------------------------------------------------
mean_rmse = np.nanmean(nrmse_group7, axis=(0,1)) #(5,12)
mean_rmse_mean = np.nanmean(mean_rmse, axis=(0)) #(12)
mean_rmse_min = np.nanmin(mean_rmse,axis=0) #(12)
mean_rmse_max = np.nanmax(mean_rmse,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)

fig = plt.figure(figsize=(7.5,8))
ax = plt.subplot(212)
ax.errorbar(np.arange(0,2.5,0.5),mean_rmse_mean,yerr=d,fmt='o',c='gray',capsize=4)
ax.set_xticks(np.arange(0,2.5,0.5))
ax.set_ylabel('Test NNSE')
plt.axhline(mean_rmse_mean[-2], linestyle='--', color='gray')
plt.savefig('figureS8.pdf')
print('Figure S8 completed!')



# --------------------------------------------------------------------------------------
# Fig S9. 
# --------------------------------------------------------------------------------------
# obs
obs_200 = np.load(path+'obs_gd_9km.npy')[:,-731:]
obs_400 = np.load(path+'obs_gd_9km_400.npy')[:,-731:]
obs_600 = np.load(path+'obs_gd_9km_600.npy')[:,-731:]

# pred
pred_200 = np.load(path+"multi_tasks_v1_5/multi_tasks_v1_ens_gd_9km.npy")[:,-731:,1,:,-1] 
pred_400 = np.load(path+"multi_tasks_v1_200%/multi_tasks_v1_ens_gd_9km.npy")[:,-731:,1,:,1] 
pred_600 = np.load(path+"multi_tasks_v1_300%/multi_tasks_v1_ens_gd_9km.npy")[:,-731:,1,:,-2] 

# 400, n, 5, 5
nrmse_200 = np.full((200,5), np.nan)
for i in range(200):
    for m in range(5):
        nrmse_200[i,m] = nnse(obs_200[i,:,m],pred_200[i,:,m])
nrmse_400 = np.full((200,5), np.nan)
for i in range(200):
    for m in range(5):
        nrmse_400[i,m] = nnse(obs_400[i,:,m],pred_400[i,:,m])
nrmse_600 = np.full((200,5), np.nan)
for i in range(200):
    for m in range(5):
        nrmse_600[i,m] = nnse(obs_600[i,:,m],pred_600[i,:,m])

plt.figure()
for i in range(5):
    plt.subplot(3,2,i+1)
    plt.boxplot(nrmse_200[:,i], positions=[0])
    plt.boxplot(nrmse_400[:,i], positions=[1])
    plt.boxplot(nrmse_600[:,i], positions=[2])
plt.savefig('figureS9.pdf')
print('Figure S9 completed!')








# load simulate
pred_soft_alpha_0p1 = np.load(path+"expB/soft_multi_tasks_alpha_0.1_data_50%/soft_multi_tasks_ens_gd_9km.npy")
pred_soft_alpha_0p5 = np.load(path+"expD/soft_multi_tasks_alpha_0.5_data_50%/soft_multi_tasks_ens_gd_9km.npy")
pred_hybrid_alpha_0p1 = np.load(path+"expB/hard_multi_tasks_v3_alpha_0.1_data_100%/hard_multi_tasks_v3_ens_gd_9km.npy") 
pred_hybrid_alpha_0p5 = np.load(path+"expD/hard_multi_tasks_v3_alpha_0.5_data_100%/hard_multi_tasks_v3_ens_gd_9km.npy")
pred_group = np.stack([pred_soft_alpha_0p1,pred_soft_alpha_0p5,pred_hybrid_alpha_0p1,pred_hybrid_alpha_0p5], axis=-1)[:,-731:,1] 

# load test (ngrid,nt,nfeat)
obs = np.load(path+'obs_gd_9km.npy')[:,-731:]
aux = np.load(path+'aux_gd_9km.npy')[:,-731:]

# get shape
ngrid, nt, nfeat, nens, nmodel = pred_group.shape

# cal general performance
nnse_group = np.full((ngrid,nfeat,nens,nmodel), np.nan)
r_group = np.full((ngrid,nfeat,nens,nmodel), np.nan)
for i in range(ngrid):
    for m in range(nfeat):
        for j in range(nens):
            for k in range(nmodel):
                nnse_group[i,m,j,k] = nnse(obs[i,:,m],pred_group[i,:,m,j,k])
                r_group[i,m,j,k] = np.corrcoef(obs[i,:,m],pred_group[i,:,m,j,k])[0,1]

# load scv 100% data alpha 0.1
pred_soft_cv_0_alpha_0p1 = np.load(path+"expC/soft_multi_tasks_alpha_0.1_data_100%_scv_0/soft_multi_tasks_ens_gd_9km.npy")
pred_soft_cv_0_alpha_0p5 = np.load(path+"expD/soft_multi_tasks_alpha_0.5_data_100%_scv_0/soft_multi_tasks_ens_gd_9km.npy")
pred_hybrid_cv_0_alpha_0p1 = np.load(path+"expC/hard_multi_tasks_v3_alpha_0.1_data_100%_scv_0/hard_multi_tasks_v3_ens_gd_9km.npy")
pred_hybrid_cv_0_alpha_0p5 = np.load(path+"expD/hard_multi_tasks_v3_alpha_0.5_data_100%_scv_0/hard_multi_tasks_v3_ens_gd_9km.npy")

pred_soft_cv_1_alpha_0p1 = np.load(path+"expC/soft_multi_tasks_alpha_0.1_data_100%_scv_1/soft_multi_tasks_ens_gd_9km.npy")
pred_soft_cv_1_alpha_0p5 = np.load(path+"expD/soft_multi_tasks_alpha_0.5_data_100%_scv_1/soft_multi_tasks_ens_gd_9km.npy")
pred_hybrid_cv_1_alpha_0p1 = np.load(path+"expC/hard_multi_tasks_v3_alpha_0.1_data_100%_scv_1/hard_multi_tasks_v3_ens_gd_9km.npy")
pred_hybrid_cv_1_alpha_0p5 = np.load(path+"expD/hard_multi_tasks_v3_alpha_0.5_data_100%_scv_1/hard_multi_tasks_v3_ens_gd_9km.npy")

pred_soft_cv_2_alpha_0p1 = np.load(path+"expC/soft_multi_tasks_alpha_0.1_data_100%_scv_2/soft_multi_tasks_ens_gd_9km.npy")
pred_soft_cv_2_alpha_0p5 = np.load(path+"expD/soft_multi_tasks_alpha_0.5_data_100%_scv_2/soft_multi_tasks_ens_gd_9km.npy")
pred_hybrid_cv_2_alpha_0p1 = np.load(path+"expC/hard_multi_tasks_v3_alpha_0.1_data_100%_scv_2/hard_multi_tasks_v3_ens_gd_9km.npy")
pred_hybrid_cv_2_alpha_0p5 = np.load(path+"expD/hard_multi_tasks_v3_alpha_0.5_data_100%_scv_2/hard_multi_tasks_v3_ens_gd_9km.npy")

pred_soft_cv_3_alpha_0p1 = np.load(path+"expC/soft_multi_tasks_alpha_0.1_data_100%_scv_3/soft_multi_tasks_ens_gd_9km.npy")
pred_soft_cv_3_alpha_0p5 = np.load(path+"expD/soft_multi_tasks_alpha_0.5_data_100%_scv_3/soft_multi_tasks_ens_gd_9km.npy")
pred_hybrid_cv_3_alpha_0p1 = np.load(path+"expC/hard_multi_tasks_v3_alpha_0.1_data_100%_scv_3/hard_multi_tasks_v3_ens_gd_9km.npy")
pred_hybrid_cv_3_alpha_0p5 = np.load(path+"expD/hard_multi_tasks_v3_alpha_0.5_data_100%_scv_3/hard_multi_tasks_v3_ens_gd_9km.npy")

pred_group_scv_0 = np.stack([pred_soft_cv_0_alpha_0p1,pred_soft_cv_0_alpha_0p5,pred_hybrid_cv_0_alpha_0p1,pred_hybrid_cv_0_alpha_0p5], axis=-1)[:,-731:,1] 
pred_group_scv_1 = np.stack([pred_soft_cv_1_alpha_0p1,pred_soft_cv_1_alpha_0p5,pred_hybrid_cv_1_alpha_0p1,pred_hybrid_cv_1_alpha_0p5], axis=-1)[:,-731:,1] 
pred_group_scv_2 = np.stack([pred_soft_cv_2_alpha_0p1,pred_soft_cv_2_alpha_0p5,pred_hybrid_cv_2_alpha_0p1,pred_hybrid_cv_2_alpha_0p5], axis=-1)[:,-731:,1] 
pred_group_scv_3 = np.stack([pred_soft_cv_3_alpha_0p1,pred_soft_cv_3_alpha_0p5,pred_hybrid_cv_3_alpha_0p1,pred_hybrid_cv_3_alpha_0p5], axis=-1)[:,-731:,1] 

# load tcv 100% data alpha 0.1
pred_soft_tcv_0_alpha_0p1 = np.load(path+"expC/soft_multi_tasks_alpha_0.1_data_100%_tcv_summer/soft_multi_tasks_ens_gd_9km.npy")
pred_soft_tcv_0_alpha_0p5 = np.load(path+"expD/soft_multi_tasks_alpha_0.5_data_100%_tcv_summer/soft_multi_tasks_ens_gd_9km.npy")
pred_hybrid_tcv_0_alpha_0p1 = np.load(path+"expC/hard_multi_tasks_v3_alpha_0.1_data_100%_tcv_summer/hard_multi_tasks_v3_ens_gd_9km.npy")
pred_hybrid_tcv_0_alpha_0p5 = np.load(path+"expD/hard_multi_tasks_v3_alpha_0.5_data_100%_tcv_summer/hard_multi_tasks_v3_ens_gd_9km.npy")
pred_group_tcv_0 = np.stack([pred_soft_tcv_0_alpha_0p1,pred_soft_tcv_0_alpha_0p5,pred_hybrid_tcv_0_alpha_0p1,pred_hybrid_tcv_0_alpha_0p5], axis=-1)[:,-731:,1] 

# get shape
ngrid, nt, nfeat, nens, nmodel = pred_group_scv_0.shape

# cal general performance
nnse_group_scv_0 = np.full((50,nfeat,nens,nmodel), np.nan)
r_group_scv_0 = np.full((50,nfeat,nens,nmodel), np.nan)
for i in range(50):
    for m in range(nfeat):
        for j in range(nens):
            for k in range(nmodel):
                nnse_group_scv_0[i,m,j,k] = nnse(obs[i,:,m],pred_group_scv_0[i,:,m,j,k])
                r_group_scv_0[i,m,j,k] = np.corrcoef(obs[i,:,m],pred_group_scv_0[i,:,m,j,k])[0,1]

nnse_group_scv_1 = np.full((50,nfeat,nens,nmodel), np.nan)
r_group_scv_1 = np.full((50,nfeat,nens,nmodel), np.nan)
for i in range(50,100):
    for m in range(nfeat):
        for j in range(nens):
            for k in range(nmodel):
                nnse_group_scv_1[i-50,m,j,k] = nnse(obs[i,:,m],pred_group_scv_1[i,:,m,j,k])
                r_group_scv_1[i-50,m,j,k] = np.corrcoef(obs[i,:,m],pred_group_scv_1[i,:,m,j,k])[0,1]
             
nnse_group_scv_2 = np.full((50,nfeat,nens,nmodel), np.nan)
r_group_scv_2 = np.full((50,nfeat,nens,nmodel), np.nan)
for i in range(100,150):
    for m in range(nfeat):
        for j in range(nens):
            for k in range(nmodel):
                nnse_group_scv_2[i-100,m,j,k] = nnse(obs[i,:,m],pred_group_scv_2[i,:,m,j,k])
                r_group_scv_2[i-100,m,j,k] = np.corrcoef(obs[i,:,m],pred_group_scv_2[i,:,m,j,k])[0,1]
         
nnse_group_scv_3 = np.full((50,nfeat,nens,nmodel), np.nan)
r_group_scv_3 = np.full((50,nfeat,nens,nmodel), np.nan)
for i in range(150,200):
    for m in range(nfeat):
        for j in range(nens):
            for k in range(nmodel):
                nnse_group_scv_3[i-150,m,j,k] = nnse(obs[i,:,m],pred_group_scv_3[i,:,m,j,k])
                r_group_scv_3[i-150,m,j,k] = np.corrcoef(obs[i,:,m],pred_group_scv_3[i,:,m,j,k])[0,1]
     
nnse_group_tcv_0 = np.full((ngrid,nfeat,nens,nmodel), np.nan)
r_group_tcv_0 = np.full((ngrid,nfeat,nens,nmodel), np.nan)
idx = []
idx.append(np.arange(172,264))
idx.append(np.arange(172+365,264+365))
idx = np.concatenate(idx,axis=0)
for i in range(ngrid):
    for m in range(nfeat):
        for j in range(nens):
            for k in range(nmodel):
                nnse_group_tcv_0[i,m,j,k] = nnse(obs[i,:,m],pred_group_tcv_0[i,:,m,j,k])
                r_group_tcv_0[i,m,j,k] = np.corrcoef(obs[i,:,m],pred_group_tcv_0[i,:,m,j,k])[0,1]
    
