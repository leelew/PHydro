import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

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
pred_group_bk = np.stack([
    pred_mt_train_0p01, pred_soft_train_0p01, pred_hard_train_0p01,pred_hybrid_train_0p01,
    pred_mt_train_0p1, pred_soft_train_0p1, pred_hard_train_0p1,pred_hybrid_train_0p1,
    pred_mt_train_0p2,pred_soft_train_0p2,pred_hard_train_0p2,pred_hybrid_train_0p2,
    pred_mt_train_0p5,pred_soft_train_0p5,pred_hard_train_0p5,pred_hybrid_train_0p5,
    pred_mt_train_1,pred_soft_train_1,pred_hard_train_1,pred_hybrid_train_1], axis=-1)[:,-731:]
pred_group_mean = np.nanmean(pred_group[:,:,:,:,-4:], axis=-2)

# load obs (ngrid,nt,nfeat)
obs = np.load(path+'obs_gd_9km.npy')[:,-731:]
aux = np.load(path+'aux_gd_9km.npy')[:,-731:]
p = np.load(path+'p_test.npy')[-731:]
p = np.transpose(p,(1,0))
p = p.reshape(-1,)

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

# Calc performance of each feat under extreme condition grid-by-grid
nnse_10th_group = np.full((ngrid,nfeat,nens,nmodel), np.nan)
r_10th_group = np.full((ngrid,nfeat,nens,nmodel), np.nan)
nnse_90th_group = np.full((ngrid,nfeat,nens,nmodel), np.nan)
r_90th_group = np.full((ngrid,nfeat,nens,nmodel), np.nan)

for i in range(ngrid):
    for m in range(nfeat):
        for j in range(nens):
            for k in range(nmodel):
                tmp = obs[i,:,m]
                t1,t2 = np.nanquantile(tmp,0.1),np.nanquantile(tmp,0.9)
                idx1,idx2 = np.where(tmp<=t1)[0],np.where(tmp>=t2)[0]
                nnse_10th_group[i,m,j,k] = nnse(obs[i,idx1,m],pred_group[i,idx1,m,j,k])
                r_10th_group[i,m,j,k] = np.corrcoef(obs[i,idx1,m],pred_group[i,idx1,m,j,k])[0,1]
                nnse_90th_group[i,m,j,k] = nnse(obs[i,idx2,m],pred_group[i,idx2,m,j,k])
                r_90th_group[i,m,j,k] = np.corrcoef(obs[i,idx2,m],pred_group[i,idx2,m,j,k])[0,1]


"""# Calc performance of each feat under extreme condition in all grids
nnse_10th_group = np.full((nfeat,nens,nmodel), np.nan)
r_10th_group = np.full((nfeat,nens,nmodel), np.nan)
nnse_90th_group = np.full((nfeat,nens,nmodel), np.nan)
r_90th_group = np.full((nfeat,nens,nmodel), np.nan)

for m in range(nfeat):
    for j in range(nens):
        for k in range(nmodel):
            y_obs, y_sim = obs[:,:,m].reshape(-1,),pred_group[:,:,m,j,k].reshape(-1,)
            t1,t2 = np.nanquantile(y_obs,0.03),np.nanquantile(y_obs,0.97)
            idx1,idx2 = np.where(y_obs<=t1)[0],np.where(y_obs>=t2)[0]
            nnse_10th_group[m,j,k] = nnse(y_obs[idx1],y_sim[idx1])
            r_10th_group[m,j,k] = np.corrcoef(y_obs[idx1],y_sim[idx1])[0,1]
            nnse_90th_group[m,j,k] = nnse(y_obs[idx2],y_sim[idx2])
            r_90th_group[m,j,k] = np.corrcoef(y_obs[idx2],y_sim[idx2])[0,1]

#(nfeat,4)
c, d = np.nanmean(nnse_10th_group,axis=(1)), np.nanmean(nnse_90th_group,axis=(1))
e, f = c[:,-4:],d[:,-4:]
a1 = (e[:,1:]-e[:,0:1])*100/np.abs(e[:,0:1])
a2 = (f[:,1:]-f[:,0:1])*100/np.abs(f[:,0:1])
print(np.around((e[:,1:]-e[:,0:1])*100/np.abs(e[:,0:1]),2))
print(np.around((f[:,1:]-f[:,0:1])*100/np.abs(f[:,0:1]),2))
e, f = c[:,:4],d[:,:4]
a3 = (e[:,1:]-e[:,0:1])*100/np.abs(e[:,0:1])
a4 = (f[:,1:]-f[:,0:1])*100/np.abs(f[:,0:1])  #(5,3)
print(np.around((e[:,1:]-e[:,0:1])*100/np.abs(e[:,0:1]),2))
print(np.around((f[:,1:]-f[:,0:1])*100/np.abs(f[:,0:1]),2))

plt.figure()
plt.subplot(1,2,1)
plt.bar(x=[0.1,0.35,0.6],height=a1[0], width=0.25)
plt.bar(x=[1.1,1.35,1.6],height=a2[-1], width=0.25)
plt.subplot(1,2,2)
plt.bar(x=[0.1,0.35,0.6],height=a3[0], width=0.25)
plt.bar(x=[1.1,1.35,1.6],height=a4[-1], width=0.25)
plt.savefig('figureS5.pdf')
"""

# cal physical consistency
phy_group = np.full((ngrid,nens,nmodel), np.nan)
for j in range(nens):
    for k in range(nmodel):
        phy_group[:,j,k],_ = cal_phy_cons(pred_group_bk[:,:,:,:,j,k],aux)

# cal extreme performance over different p
nnse_p_group = np.full((ngrid,nfeat,nens,nmodel,3), np.nan)
for i in range(ngrid):
    for m in range(nfeat):
        for j in range(nens):
            for k in range(nmodel):
                tmp = obs[i,:,m]
                idx1,idx2,idx3 = np.where(tmp<10)[0],np.where((tmp>10)&(tmp<50))[0],np.where(tmp>50)[0]
                nnse_p_group[i,m,j,k,0] = nnse(obs[i,idx1,m],pred_group[i,idx1,m,j,k])
                nnse_p_group[i,m,j,k,1] = nnse(obs[i,idx2,m],pred_group[i,idx2,m,j,k])
                nnse_p_group[i,m,j,k,2] = nnse(obs[i,idx3,m],pred_group[i,idx3,m,j,k])


# --------------------------------------------------------------------
# Figure S2.
# --------------------------------------------------------------------
a = phy_group[:,:,-4:] # 100%
a1 = np.nanmean(a[0:50],axis=(0)) 
a2 = np.nanmean(a[50:100],axis=(0))
a3 = np.nanmean(a[100:150],axis=(0))
a4 = np.nanmean(a[150:200],axis=(0))
a5 = np.nanmean(a, axis=(0))

plt.figure()
mean_rmse_mean = np.nanmean(a1, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a1,axis=0) #(12)
mean_rmse_max = np.nanmax(a1,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(231)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=7,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title('0-25% P', fontsize=13)

mean_rmse_mean = np.nanmean(a2, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a2,axis=0) #(12)
mean_rmse_max = np.nanmax(a2,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(232)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=7,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title('25-50% P', fontsize=13)

mean_rmse_mean = np.nanmean(a3, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a3,axis=0) #(12)
mean_rmse_max = np.nanmax(a3,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(234)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=7,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title('50-75% P', fontsize=13)

mean_rmse_mean = np.nanmean(a4, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a4,axis=0) #(12)
mean_rmse_max = np.nanmax(a4,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(235)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=7,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title('75-100% P', fontsize=13)

mean_rmse_mean = np.nanmean(a5, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a5,axis=0) #(12)
mean_rmse_max = np.nanmax(a5,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(236)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=7, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=7, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=7, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=7,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title('All Regions', fontsize=13)
plt.savefig('figureS2.pdf')
print('Figure S2 completed!')


# --------------------------------------------------------------------------------------
# Fig 7. 
# --------------------------------------------------------------------------------------
mean_rmse = np.nanmean(nnse_group, axis=(0,1)) #(5,12)
mean_rmse_mean = np.nanmean(mean_rmse, axis=(0)) #(12)
mean_rmse_min = np.nanmin(mean_rmse,axis=0) #(12)
mean_rmse_max = np.nanmax(mean_rmse,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)

fig = plt.figure(figsize=(9,6))
ax = plt.subplot(111)
ax.errorbar(np.arange(0,1,0.2),mean_rmse_mean[::4],yerr=d[:,::4],ms=4, marker='o', mfc='w', color='gray',capsize=5, linestyle='none')
ax.errorbar(np.arange(1,2,0.2),mean_rmse_mean[1::4],yerr=d[:,1::4],ms=4, marker='s', mfc='w', color='red',capsize=5,linestyle='none')
ax.errorbar(np.arange(2,3,0.2),mean_rmse_mean[2::4],yerr=d[:,2::4],ms=4, marker='D', mfc='w', color='green',capsize=5,linestyle='none')
ax.errorbar(np.arange(3,4,0.2),mean_rmse_mean[3::4],yerr=d[:,3::4],ms=4, marker='h', mfc='w', color='blue',capsize=5,linestyle='none')
ax.plot(np.arange(0,1,0.2),mean_rmse_mean[::4],color='gray',linestyle='--')
ax.plot(np.arange(1,2,0.2),mean_rmse_mean[1::4],color='red',linestyle='--')
ax.plot(np.arange(2,3,0.2),mean_rmse_mean[2::4],color='green',linestyle='--')
ax.plot(np.arange(3,4,0.2),mean_rmse_mean[3::4],color='blue',linestyle='--')
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
plt.axhline(mean_rmse_mean[4], color='gray',linestyle='--',linewidth=0.5)
plt.axhline(mean_rmse_mean[8], color='gray',linestyle='--',linewidth=0.5)
plt.axhline(mean_rmse_mean[12], color='gray',linestyle='--',linewidth=0.5)
plt.axhline(mean_rmse_mean[16], color='gray',linestyle='--',linewidth=0.5)
ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8])
ax.set_xticklabels(['1%','10%','20%','50%','100%','1%','10%','20%','50%','100%','1%','10%','20%','50%','100%','1%','10%','20%','50%','100%'])
plt.axvline(0.9, color='gray',linestyle='--',linewidth=0.5)
plt.axvline(1.9, color='gray',linestyle='--',linewidth=0.5)
plt.axvline(2.9, color='gray',linestyle='--',linewidth=0.5)
plt.axvline(3.9, color='gray',linestyle='--',linewidth=0.5)
ax.set_ylabel('Test performance')
ax.set_xlabel('Training data profiles')
plt.xlim(-0.1,3.9)
plt.savefig('figure7.pdf')
print('Figure 7 completed!')
print(mean_rmse_mean[-4:]-mean_rmse_mean[:4])


# --------------------------------------------------------------------------------------
# Fig 8. 
# --------------------------------------------------------------------------------------
fig = plt.figure(figsize=(9.9,13.2))
ms = 9
kk = 0.03

a = nnse_group[:,:,:,-4:]
a1 = np.nanmean(a[0:50],axis=(0,1)) #(5,4)
a2 = np.nanmean(a[50:100],axis=(0,1))
a3 = np.nanmean(a[100:150],axis=(0,1))
a4 = np.nanmean(a[150:200],axis=(0,1))
a5 = np.nanmean(a, axis=(0,1))

mean_rmse_mean = np.nanmean(a1, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a1,axis=0) #(12)
mean_rmse_max = np.nanmax(a1,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(431)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title('0-25% P', fontsize=13)

mean_rmse_mean = np.nanmean(a2, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a2,axis=0) #(12)
mean_rmse_max = np.nanmax(a2,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(432)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title("25-50% P", fontsize=13)

mean_rmse_mean = np.nanmean(a3, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a3,axis=0) #(12)
mean_rmse_max = np.nanmax(a3,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(434)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title("50-75% P", fontsize=13)

mean_rmse_mean = np.nanmean(a4, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a4,axis=0) #(12)
mean_rmse_max = np.nanmax(a4,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(435)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.3)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title("75-100% P", fontsize=13)

mean_rmse_mean = np.nanmean(a5, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a5,axis=0) #(12)
mean_rmse_max = np.nanmax(a5,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(436)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.3)
ax.set_xticks([])
plt.title("All regions", fontsize=13)

a = nnse_group[:,:,:,:4] #(200,5,5,1)
a1 = np.nanmean(a[0:50],axis=(0,1)) #(5,4)
a2 = np.nanmean(a[50:100],axis=(0,1))
a3 = np.nanmean(a[100:150],axis=(0,1))
a4 = np.nanmean(a[150:200],axis=(0,1))
a5 = np.nanmean(a, axis=(0,1))

mean_rmse_mean = np.nanmean(a1, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a1,axis=0) #(12)
mean_rmse_max = np.nanmax(a1,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(437)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title('0-25% P', fontsize=13)

mean_rmse_mean = np.nanmean(a2, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a2,axis=0) #(12)
mean_rmse_max = np.nanmax(a2,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(438)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title('25-50% P', fontsize=13)

mean_rmse_mean = np.nanmean(a3, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a3,axis=0) #(12)
mean_rmse_max = np.nanmax(a3,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(4,3,10)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title('50-75% P', fontsize=13)

mean_rmse_mean = np.nanmean(a4, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a4,axis=0) #(12)
mean_rmse_max = np.nanmax(a4,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(4,3,11)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.3)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title('75-100% P', fontsize=13)

mean_rmse_mean = np.nanmean(a5, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a5,axis=0) #(12)
mean_rmse_max = np.nanmax(a5,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(4,3,12)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.3)
ax.set_xticks([])
plt.title('All regions', fontsize=13)

plt.savefig('figure8.pdf')
print('Figure 8 completed!')


# --------------------------------------------------------------------------------------
# Fig 9. 
# --------------------------------------------------------------------------------------
fig = plt.figure(figsize=(9.9,13.2))
ms = 9
kk = 0.03

a = nnse_p_group[:,:,:,-4:,0]
a1 = np.nanmean(a, axis=(0,1))
mean_rmse_mean = np.nanmean(a1, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a1,axis=0) #(12)
mean_rmse_max = np.nanmax(a1,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(4,3,1)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
#ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title('(a) Light rain (<10mm)', fontsize=13)

a = nnse_p_group[:,:,:,-4:,1]
a1 = np.nanmean(a, axis=(0,1))
mean_rmse_mean = np.nanmean(a1, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a1,axis=0) #(12)
mean_rmse_max = np.nanmax(a1,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(4,3,2)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xlim(-0.05, 0.2)
#ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xticks([])
plt.title('(b) Moderate rain (10-50mm)', fontsize=13)

a = nnse_p_group[:,:,:,-4:,2]
a1 = np.nanmean(a, axis=(0,1))
mean_rmse_mean = np.nanmean(a1, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a1,axis=0) #(12)
mean_rmse_max = np.nanmax(a1,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(4,3,3)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xlim(-0.05, 0.2)
#ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xticks([])
plt.title('(c) Heavy rain (>50mm)', fontsize=13)

a = nnse_p_group[:,:,:,:4,0]
a1 = np.nanmean(a, axis=(0,1))
mean_rmse_mean = np.nanmean(a1, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a1,axis=0) #(12)
mean_rmse_max = np.nanmax(a1,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(4,3,4)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xlim(-0.05, 0.2)
#ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xticks([])
plt.title('(d) Light rain (<10mm)', fontsize=13)

a = nnse_p_group[:,:,:,:4,1]
a1 = np.nanmean(a, axis=(0,1))
mean_rmse_mean = np.nanmean(a1, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a1,axis=0) #(12)
mean_rmse_max = np.nanmax(a1,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(4,3,5)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xlim(-0.05, 0.2)
#ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xticks([])
plt.title('(e) Moderate rain (10-50mm)', fontsize=13)

a = nnse_p_group[:,:,:,:4,2]
a1 = np.nanmean(a, axis=(0,1))
mean_rmse_mean = np.nanmean(a1, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a1,axis=0) #(12)
mean_rmse_max = np.nanmax(a1,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(4,3,6)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xlim(-0.05, 0.2)
#ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xticks([])
plt.title('(f) Heavy rain (>50mm)', fontsize=13)
plt.savefig('figure9.pdf')
print('Figure 9 completed!')


# --------------------------------------------------------------------------------------
# Fig S3. 
# --------------------------------------------------------------------------------------
fig = plt.figure(figsize=(20,12.5))
ms = 9
kk = 0.02
kk1 = 0.03

a = nnse_10th_group[:,0:1,:,-4:]
a1 = np.nanmean(a[0:50],axis=(0,1)) #(5,4)
a2 = np.nanmean(a[50:100],axis=(0,1))
a3 = np.nanmean(a[100:150],axis=(0,1))
a4 = np.nanmean(a[150:200],axis=(0,1))
a5 = np.nanmean(a, axis=(0,1))

mean_rmse_mean = np.nanmean(a1, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a1,axis=0) #(12)
mean_rmse_max = np.nanmax(a1,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,1)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title('0-25% P', fontsize=13)

mean_rmse_mean = np.nanmean(a2, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a2,axis=0) #(12)
mean_rmse_max = np.nanmax(a2,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,2)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title('25-50% P', fontsize=13)

mean_rmse_mean = np.nanmean(a3, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a3,axis=0) #(12)
mean_rmse_max = np.nanmax(a3,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,3)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.3)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title('50-75% P', fontsize=13)

mean_rmse_mean = np.nanmean(a4, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a4,axis=0) #(12)
mean_rmse_max = np.nanmax(a4,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,4)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title('75-100% P', fontsize=13)

mean_rmse_mean = np.nanmean(a5, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a5,axis=0) #(12)
mean_rmse_max = np.nanmax(a5,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,5)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xticks([])
plt.title('All regions', fontsize=13)

a = nnse_10th_group[:,1:2,:,-4:] #(200,5,5,1)
a1 = np.nanmean(a[0:50],axis=(0,1)) #(5,4)
a2 = np.nanmean(a[50:100],axis=(0,1))
a3 = np.nanmean(a[100:150],axis=(0,1))
a4 = np.nanmean(a[150:200],axis=(0,1))
a5 = np.nanmean(a, axis=(0,1))

mean_rmse_mean = np.nanmean(a1, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a1,axis=0) #(12)
mean_rmse_max = np.nanmax(a1,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,6)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])

mean_rmse_mean = np.nanmean(a2, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a2,axis=0) #(12)
mean_rmse_max = np.nanmax(a2,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,7)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xticks([])

mean_rmse_mean = np.nanmean(a3, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a3,axis=0) #(12)
mean_rmse_max = np.nanmax(a3,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,8)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xticks([])

mean_rmse_mean = np.nanmean(a4, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a4,axis=0) #(12)
mean_rmse_max = np.nanmax(a4,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,9)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xticks([])

mean_rmse_mean = np.nanmean(a5, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a5,axis=0) #(12)
mean_rmse_max = np.nanmax(a5,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,10)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])

a = nnse_90th_group[:,4:5,:,-4:] #(200,5,5,1)
a1 = np.nanmean(a[0:50],axis=(0,1)) #(5,4)
a2 = np.nanmean(a[50:100],axis=(0,1))
a3 = np.nanmean(a[100:150],axis=(0,1))
a4 = np.nanmean(a[150:200],axis=(0,1))
a5 = np.nanmean(a, axis=(0,1))

mean_rmse_mean = np.nanmean(a1, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a1,axis=0) #(12)
mean_rmse_max = np.nanmax(a1,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,11)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_ylim(mean_rmse_mean[0]-kk1, mean_rmse_mean[0]+kk1)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])

mean_rmse_mean = np.nanmean(a2, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a2,axis=0) #(12)
mean_rmse_max = np.nanmax(a2,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,12)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk1, mean_rmse_mean[0]+kk1)
ax.set_xlim(-0.05, 0.2)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xticks([])

mean_rmse_mean = np.nanmean(a3, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a3,axis=0) #(12)
mean_rmse_max = np.nanmax(a3,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,13)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk1, mean_rmse_mean[0]+kk1)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])

mean_rmse_mean = np.nanmean(a4, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a4,axis=0) #(12)
mean_rmse_max = np.nanmax(a4,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,14)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk1, mean_rmse_mean[0]+kk1)
ax.set_xlim(-0.05, 0.2)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xticks([])

mean_rmse_mean = np.nanmean(a5, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a5,axis=0) #(12)
mean_rmse_max = np.nanmax(a5,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,15)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk1, mean_rmse_mean[0]+kk1)
ax.set_xlim(-0.05, 0.2)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xticks([])

plt.savefig('figureS3.pdf')
print('Figure S3 completed!')


# --------------------------------------------------------------------------------------
# Fig S4. 
# --------------------------------------------------------------------------------------
fig = plt.figure(figsize=(20,12.5))
ms = 9

a = nnse_10th_group[:,0:1,:,:4] #(200,5,5,1)
a1 = np.nanmean(a[0:50],axis=(0,1)) #(5,4)
a2 = np.nanmean(a[50:100],axis=(0,1))
a3 = np.nanmean(a[100:150],axis=(0,1))
a4 = np.nanmean(a[150:200],axis=(0,1))
a5 = np.nanmean(a, axis=(0,1))

mean_rmse_mean = np.nanmean(a1, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a1,axis=0) #(12)
mean_rmse_max = np.nanmax(a1,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,1)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title('0-25% P', fontsize=13)

mean_rmse_mean = np.nanmean(a2, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a2,axis=0) #(12)
mean_rmse_max = np.nanmax(a2,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,2)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title('25-50% P', fontsize=13)

mean_rmse_mean = np.nanmean(a3, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a3,axis=0) #(12)
mean_rmse_max = np.nanmax(a3,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,3)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.3)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title('50-75% P', fontsize=13)

mean_rmse_mean = np.nanmean(a4, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a4,axis=0) #(12)
mean_rmse_max = np.nanmax(a4,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,4)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])
plt.title('75-100% P', fontsize=13)

mean_rmse_mean = np.nanmean(a5, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a5,axis=0) #(12)
mean_rmse_max = np.nanmax(a5,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,5)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xticks([])
plt.title('All regions', fontsize=13)

a = nnse_10th_group[:,1:2,:,:4] #(200,5,5,1)
a1 = np.nanmean(a[0:50],axis=(0,1)) #(5,4)
a2 = np.nanmean(a[50:100],axis=(0,1))
a3 = np.nanmean(a[100:150],axis=(0,1))
a4 = np.nanmean(a[150:200],axis=(0,1))
a5 = np.nanmean(a, axis=(0,1))

mean_rmse_mean = np.nanmean(a1, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a1,axis=0) #(12)
mean_rmse_max = np.nanmax(a1,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,6)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])

mean_rmse_mean = np.nanmean(a2, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a2,axis=0) #(12)
mean_rmse_max = np.nanmax(a2,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,7)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xticks([])

mean_rmse_mean = np.nanmean(a3, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a3,axis=0) #(12)
mean_rmse_max = np.nanmax(a3,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,8)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xticks([])

mean_rmse_mean = np.nanmean(a4, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a4,axis=0) #(12)
mean_rmse_max = np.nanmax(a4,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,9)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xticks([])

mean_rmse_mean = np.nanmean(a5, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a5,axis=0) #(12)
mean_rmse_max = np.nanmax(a5,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,10)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])

a = nnse_90th_group[:,4:5,:,:4] #(200,5,5,1)
a1 = np.nanmean(a[0:50],axis=(0,1)) #(5,4)
a2 = np.nanmean(a[50:100],axis=(0,1))
a3 = np.nanmean(a[100:150],axis=(0,1))
a4 = np.nanmean(a[150:200],axis=(0,1))
a5 = np.nanmean(a, axis=(0,1))

mean_rmse_mean = np.nanmean(a1, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a1,axis=0) #(12)
mean_rmse_max = np.nanmax(a1,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,11)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])

mean_rmse_mean = np.nanmean(a2, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a2,axis=0) #(12)
mean_rmse_max = np.nanmax(a2,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,12)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xticks([])

mean_rmse_mean = np.nanmean(a3, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a3,axis=0) #(12)
mean_rmse_max = np.nanmax(a3,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,13)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xlim(-0.05, 0.2)
ax.set_xticks([])

mean_rmse_mean = np.nanmean(a4, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a4,axis=0) #(12)
mean_rmse_max = np.nanmax(a4,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,14)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xticks([])

mean_rmse_mean = np.nanmean(a5, axis=(0)) #(12)
mean_rmse_min = np.nanmin(a5,axis=0) #(12)
mean_rmse_max = np.nanmax(a5,axis=0) #(12)
d1 = mean_rmse_mean-mean_rmse_min
d2 = mean_rmse_max-mean_rmse_mean
d = np.stack([d1,d2], axis=0)
ax = plt.subplot(3,5,15)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.errorbar(0,mean_rmse_mean[0],yerr=d[:,0:1],ms=ms, marker='o', mfc='w', color='gray',capsize=5)#fmt='o',capsize=5,c='gray',)
ax.errorbar(0.05,mean_rmse_mean[1],yerr=d[:,1:2], ms=ms, marker='s', mfc='w', color='red',capsize=5)
ax.errorbar(0.1,mean_rmse_mean[2],yerr=d[:,2:3],ms=ms, marker='D', mfc='w', color='green',capsize=5)
ax.errorbar(0.15,mean_rmse_mean[3],yerr=d[:,3:4],ms=ms,  marker='h', mfc='w', color='blue',capsize=5)
ax.set_ylim(mean_rmse_mean[0]-kk, mean_rmse_mean[0]+kk)
ax.set_xlim(-0.05, 0.2)
plt.axhline(mean_rmse_mean[0], color='gray',linestyle='--',linewidth=0.5)
ax.set_xticks([])

plt.savefig('figureS4.pdf')
print('Figure S4 completed!')








