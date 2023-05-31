import numpy as np
import matplotlib.pyplot as plt

from config import parse_args
from utils import cal_phy_cons, nnse


# parameters
cfg = parse_args()
path = cfg["outputs_path"]+'forecast/'

# load simulate
pred_mt_train_1    = np.load(path+"expB/multi_tasks_v1_factor_5_data_100%/multi_tasks_v1_ens_gd_9km.npy") 
pred_mt_train_0p5  = np.load(path+"expB/multi_tasks_v1_factor_5_data_50%/multi_tasks_v1_ens_gd_9km.npy")
pred_mt_train_0p2  = np.load(path+"expB/multi_tasks_v1_factor_5_data_20%/multi_tasks_v1_ens_gd_9km.npy")
pred_mt_train_0p1  = np.load(path+"expB/multi_tasks_v1_factor_5_data_10%/multi_tasks_v1_ens_gd_9km.npy")
pred_mt_train_0p01 = np.load(path+"expB/multi_tasks_v1_factor_5_data_1%/multi_tasks_v1_ens_gd_9km.npy")

pred_soft_train_1    = np.load(path+"expB/soft_multi_tasks_alpha_0.1_data_100%/soft_multi_tasks_ens_gd_9km.npy")
pred_soft_train_0p5  = np.load(path+"expB/soft_multi_tasks_alpha_0.1_data_50%/soft_multi_tasks_ens_gd_9km.npy")
pred_soft_train_0p2  = np.load(path+"expB/soft_multi_tasks_alpha_0.1_data_20%/soft_multi_tasks_ens_gd_9km.npy")
pred_soft_train_0p1  = np.load(path+"expB/soft_multi_tasks_alpha_0.1_data_10%/soft_multi_tasks_ens_gd_9km.npy")
pred_soft_train_0p01 = np.load(path+"expB/soft_multi_tasks_alpha_0.1_data_1%/soft_multi_tasks_ens_gd_9km.npy")

pred_hard_train_1    = np.load(path+"expB/hard_multi_tasks_v1_data_100%/hard_multi_tasks_v1_ens_gd_9km.npy")
pred_hard_train_0p5  = np.load(path+"expB/hard_multi_tasks_v1_data_50%/hard_multi_tasks_v1_ens_gd_9km.npy")
pred_hard_train_0p2  = np.load(path+"expB/hard_multi_tasks_v1_data_20%/hard_multi_tasks_v1_ens_gd_9km.npy")
pred_hard_train_0p1  = np.load(path+"expB/hard_multi_tasks_v1_data_10%/hard_multi_tasks_v1_ens_gd_9km.npy")
pred_hard_train_0p01 = np.load(path+"expB/hard_multi_tasks_v1_data_1%/hard_multi_tasks_v1_ens_gd_9km.npy")

pred_hybrid_train_1    = np.load(path+"expB/hard_multi_tasks_v3_alpha_0.1_data_100%/hard_multi_tasks_v3_ens_gd_9km.npy")
pred_hybrid_train_0p5  = np.load(path+"expB/hard_multi_tasks_v3_alpha_0.1_data_50%/hard_multi_tasks_v3_ens_gd_9km.npy")
pred_hybrid_train_0p2  = np.load(path+"expB/hard_multi_tasks_v3_alpha_0.1_data_20%/hard_multi_tasks_v3_ens_gd_9km.npy")
pred_hybrid_train_0p1  = np.load(path+"expB/hard_multi_tasks_v3_alpha_0.1_data_10%/hard_multi_tasks_v3_ens_gd_9km.npy")
pred_hybrid_train_0p01 = np.load(path+"expB/hard_multi_tasks_v3_alpha_0.1_data_1%/hard_multi_tasks_v3_ens_gd_9km.npy")

# concat & mean (ngrid,nt,nfeat,nens,nmodel)
pred_group = np.stack([
    pred_mt_train_0p01, pred_soft_train_0p01, pred_hard_train_0p01,pred_hybrid_train_0p01,
    pred_mt_train_0p1, pred_soft_train_0p1, pred_hard_train_0p1,pred_hybrid_train_0p1,
    pred_mt_train_0p2,pred_soft_train_0p2,pred_hard_train_0p2,pred_hybrid_train_0p2,
    pred_mt_train_0p5,pred_soft_train_0p5,pred_hard_train_0p5,pred_hybrid_train_0p5,
    pred_mt_train_1,pred_soft_train_1,pred_hard_train_1,pred_hybrid_train_1], axis=-1)[:,-731:,1] 

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
pred_mt_cv_0 = np.load(path+"expC/multi_tasks_v1_factor_5_data_100%_scv_0/multi_tasks_v1_ens_gd_9km.npy")
pred_soft_cv_0 = np.load(path+"expC/soft_multi_tasks_alpha_0.1_data_100%_scv_0/soft_multi_tasks_ens_gd_9km.npy")
pred_hard_cv_0 = np.load(path+"expC/hard_multi_tasks_v1_data_100%_scv_0/hard_multi_tasks_v1_ens_gd_9km.npy")
pred_hybrid_cv_0 = np.load(path+"expC/hard_multi_tasks_v3_alpha_0.1_data_100%_scv_0/hard_multi_tasks_v3_ens_gd_9km.npy")

pred_mt_cv_1 = np.load(path+"expC/multi_tasks_v1_factor_5_data_100%_scv_1/multi_tasks_v1_ens_gd_9km.npy")
pred_soft_cv_1 = np.load(path+"expC/soft_multi_tasks_alpha_0.1_data_100%_scv_1/soft_multi_tasks_ens_gd_9km.npy")
pred_hard_cv_1 = np.load(path+"expC/hard_multi_tasks_v1_data_100%_scv_1/hard_multi_tasks_v1_ens_gd_9km.npy")
pred_hybrid_cv_1 = np.load(path+"expC/hard_multi_tasks_v3_alpha_0.1_data_100%_scv_1/hard_multi_tasks_v3_ens_gd_9km.npy")

pred_mt_cv_2 = np.load(path+"expC/multi_tasks_v1_factor_5_data_100%_scv_2/multi_tasks_v1_ens_gd_9km.npy")
pred_soft_cv_2 = np.load(path+"expC/soft_multi_tasks_alpha_0.1_data_100%_scv_2/soft_multi_tasks_ens_gd_9km.npy")
pred_hard_cv_2 = np.load(path+"expC/hard_multi_tasks_v1_data_100%_scv_2/hard_multi_tasks_v1_ens_gd_9km.npy")
pred_hybrid_cv_2 = np.load(path+"expC/hard_multi_tasks_v3_alpha_0.1_data_100%_scv_2/hard_multi_tasks_v3_ens_gd_9km.npy")

pred_mt_cv_3 = np.load(path+"expC/multi_tasks_v1_factor_5_data_100%_scv_3/multi_tasks_v1_ens_gd_9km.npy")
pred_soft_cv_3 = np.load(path+"expC/soft_multi_tasks_alpha_0.1_data_100%_scv_3/soft_multi_tasks_ens_gd_9km.npy")
pred_hard_cv_3 = np.load(path+"expC/hard_multi_tasks_v1_data_100%_scv_3/hard_multi_tasks_v1_ens_gd_9km.npy")
pred_hybrid_cv_3 = np.load(path+"expC/hard_multi_tasks_v3_alpha_0.1_data_100%_scv_3/hard_multi_tasks_v3_ens_gd_9km.npy")

pred_group_scv_0 = np.stack([pred_mt_cv_0, pred_soft_cv_0,pred_hard_cv_0,pred_hybrid_cv_0], axis=-1)[:,-731:,1] 
pred_group_scv_1 = np.stack([pred_mt_cv_1, pred_soft_cv_1,pred_hard_cv_1,pred_hybrid_cv_1], axis=-1)[:,-731:,1]
pred_group_scv_2 = np.stack([pred_mt_cv_2, pred_soft_cv_2,pred_hard_cv_2,pred_hybrid_cv_2], axis=-1)[:,-731:,1]
pred_group_scv_3 = np.stack([pred_mt_cv_3, pred_soft_cv_3,pred_hard_cv_3,pred_hybrid_cv_3], axis=-1)[:,-731:,1]

# load tcv 100% data alpha 0.1
pred_mt_tcv_0 = np.load(path+"expC/multi_tasks_v1_factor_5_data_100%_tcv_summer/multi_tasks_v1_ens_gd_9km.npy")
pred_soft_tcv_0 = np.load(path+"expC/soft_multi_tasks_alpha_0.1_data_100%_tcv_summer/soft_multi_tasks_ens_gd_9km.npy")
pred_hard_tcv_0 = np.load(path+"expC/hard_multi_tasks_v1_data_100%_tcv_summer/hard_multi_tasks_v1_ens_gd_9km.npy")
pred_hybrid_tcv_0 = np.load(path+"expC/hard_multi_tasks_v3_alpha_0.1_data_100%_tcv_summer/hard_multi_tasks_v3_ens_gd_9km.npy")
pred_group_tcv_0 = np.stack([pred_mt_tcv_0, pred_soft_tcv_0,pred_hard_tcv_0,pred_hybrid_tcv_0], axis=-1)[:,-731:,1] 

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
     
#nnse_group_tcv = np.full((ngrid,nfeat,nens,nmodel), np.nan)
#r_group_tcv = np.full((ngrid,nfeat,nens,nmodel), np.nan)
#idx = []
#idx.append(np.arange(172,264))
#idx.append(np.arange(172+365,264+365))
#idx = np.concatenate(idx,axis=0)
#idx.append(np.arange(0,60))
#idx.append(np.arange(334,365))
#idx.append(np.arange(0+365,60+365))
#idx.append(np.arange(334+365,365+365))
#for i in range(ngrid):
#    for m in range(nfeat):
#        for j in range(nens):
#            for k in range(nmodel):
#                nnse_group_tcv[i,m,j,k] = nnse(obs[i,:,m],pred_group[i,:,m,j,k])
#                r_group_tcv[i,m,j,k] = np.corrcoef(obs[i,:,m],pred_group[i,:,m,j,k])[0,1]
            

# --------------------------------------------------------------------------------------
# Fig 10.
# --------------------------------------------------------------------------------------
fig = plt.figure(figsize=(9,12))
kk = 0.03

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
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
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
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
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
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
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
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
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
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
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

ax = plt.subplot(437)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
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

ax = plt.subplot(438)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
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
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
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
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
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
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=7,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.3)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.savefig('figure10.pdf')
print('Figure 10 completed!')


# load scv 50% data alpha 0.1
pred_mt_cv_0 = np.load(path+"expC/multi_tasks_v1_factor_5_data_50%_scv_0/multi_tasks_v1_ens_gd_9km.npy")
pred_soft_cv_0 = np.load(path+"expC/soft_multi_tasks_alpha_0.1_data_50%_scv_0/soft_multi_tasks_ens_gd_9km.npy")
pred_hard_cv_0 = np.load(path+"expC/hard_multi_tasks_v1_data_50%_scv_0/hard_multi_tasks_v1_ens_gd_9km.npy")
pred_hybrid_cv_0 = np.load(path+"expC/hard_multi_tasks_v3_alpha_0.1_data_50%_scv_0/hard_multi_tasks_v3_ens_gd_9km.npy")

pred_mt_cv_1 = np.load(path+"expC/multi_tasks_v1_factor_5_data_50%_scv_1/multi_tasks_v1_ens_gd_9km.npy")
pred_soft_cv_1 = np.load(path+"expC/soft_multi_tasks_alpha_0.1_data_50%_scv_1/soft_multi_tasks_ens_gd_9km.npy")
pred_hard_cv_1 = np.load(path+"expC/hard_multi_tasks_v1_data_50%_scv_1/hard_multi_tasks_v1_ens_gd_9km.npy")
pred_hybrid_cv_1 = np.load(path+"expC/hard_multi_tasks_v3_alpha_0.1_data_50%_scv_1/hard_multi_tasks_v3_ens_gd_9km.npy")

pred_mt_cv_2 = np.load(path+"expC/multi_tasks_v1_factor_5_data_50%_scv_2/multi_tasks_v1_ens_gd_9km.npy")
pred_soft_cv_2 = np.load(path+"expC/soft_multi_tasks_alpha_0.1_data_50%_scv_2/soft_multi_tasks_ens_gd_9km.npy")
pred_hard_cv_2 = np.load(path+"expC/hard_multi_tasks_v1_data_50%_scv_2/hard_multi_tasks_v1_ens_gd_9km.npy")
pred_hybrid_cv_2 = np.load(path+"expC/hard_multi_tasks_v3_alpha_0.1_data_50%_scv_2/hard_multi_tasks_v3_ens_gd_9km.npy")

pred_mt_cv_3 = np.load(path+"expC/multi_tasks_v1_factor_5_data_50%_scv_3/multi_tasks_v1_ens_gd_9km.npy")
pred_soft_cv_3 = np.load(path+"expC/soft_multi_tasks_alpha_0.1_data_50%_scv_3/soft_multi_tasks_ens_gd_9km.npy")
pred_hard_cv_3 = np.load(path+"expC/hard_multi_tasks_v1_data_50%_scv_3/hard_multi_tasks_v1_ens_gd_9km.npy")
pred_hybrid_cv_3 = np.load(path+"expC/hard_multi_tasks_v3_alpha_0.1_data_50%_scv_3/hard_multi_tasks_v3_ens_gd_9km.npy")

pred_group_scv_0 = np.stack([pred_mt_cv_0, pred_soft_cv_0,pred_hard_cv_0,pred_hybrid_cv_0], axis=-1)[:,-731:,1] 
pred_group_scv_1 = np.stack([pred_mt_cv_1, pred_soft_cv_1,pred_hard_cv_1,pred_hybrid_cv_1], axis=-1)[:,-731:,1]
pred_group_scv_2 = np.stack([pred_mt_cv_2, pred_soft_cv_2,pred_hard_cv_2,pred_hybrid_cv_2], axis=-1)[:,-731:,1]
pred_group_scv_3 = np.stack([pred_mt_cv_3, pred_soft_cv_3,pred_hard_cv_3,pred_hybrid_cv_3], axis=-1)[:,-731:,1]

# load tcv 100% data alpha 0.1
pred_mt_tcv_0 = np.load(path+"expC/multi_tasks_v1_factor_5_data_50%_tcv_summer/multi_tasks_v1_ens_gd_9km.npy")
pred_soft_tcv_0 = np.load(path+"expC/soft_multi_tasks_alpha_0.1_data_50%_tcv_summer/soft_multi_tasks_ens_gd_9km.npy")
pred_hard_tcv_0 = np.load(path+"expC/hard_multi_tasks_v1_data_50%_tcv_summer/hard_multi_tasks_v1_ens_gd_9km.npy")
pred_hybrid_tcv_0 = np.load(path+"expC/hard_multi_tasks_v3_alpha_0.1_data_50%_tcv_summer/hard_multi_tasks_v3_ens_gd_9km.npy")
pred_group_tcv_0 = np.stack([pred_mt_tcv_0, pred_soft_tcv_0,pred_hard_tcv_0,pred_hybrid_tcv_0], axis=-1)[:,-731:,1] 

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
# Fig S5.
# --------------------------------------------------------------------------------------
fig = plt.figure(figsize=(9,12))
kk = 0.03

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
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
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
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
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
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
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
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
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
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
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

ax = plt.subplot(437)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
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

ax = plt.subplot(438)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
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
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
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
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
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
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=7,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.3)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.savefig('figureS5.pdf')
print('Figure S5 completed!')
