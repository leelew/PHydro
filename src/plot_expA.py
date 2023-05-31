import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import parse_args
from utils import nnse


# parameters
cfg = parse_args()
path = cfg["outputs_path"]+'forecast/'

# load simulate (ngrid,nt,nfeat,nens)
pred_st             = np.load(path+"expA/single_task/single_task_ens_gd_9km.npy")
pred_mt_uncertainty = np.load(path+"expA/multi_tasks_v2/multi_tasks_v2_ens_gd_9km.npy")
pred_mt_lam0p5      = np.load(path+"expA/multi_tasks_v1_factor_0.5/multi_tasks_v1_ens_gd_9km.npy")
pred_mt_lam1        = np.load(path+"expA/multi_tasks_v1_factor_1/multi_tasks_v1_ens_gd_9km.npy")
pred_mt_lam2p5      = np.load(path+"expA/multi_tasks_v1_factor_2.5/multi_tasks_v1_ens_gd_9km.npy")
pred_mt_lam5        = np.load(path+"expA/multi_tasks_v1_factor_5/multi_tasks_v1_ens_gd_9km.npy")
pred_mt_lam10       = np.load(path+"expA/multi_tasks_v1_factor_10/multi_tasks_v1_ens_gd_9km.npy")
pred_mt_lam20       = np.load(path+"expA/multi_tasks_v1_factor_20/multi_tasks_v1_ens_gd_9km.npy")

# concat & mean (ngrid,nt,nfeat,nens,nmodel)
pred_group = np.stack([pred_st,pred_mt_uncertainty,
                       pred_mt_lam0p5,pred_mt_lam1,pred_mt_lam2p5,
                       pred_mt_lam5,pred_mt_lam10,pred_mt_lam20], axis=-1)[:,-731:,1] 
pred_group_mean = np.nanmean(pred_group,axis=-2)

# load obs (ngrid,nt,nfeat)
obs = np.load(path+'obs_gd_9km.npy')[:,-731:]

# get shape
ngrid, nt, nfeat, nens, nmodel = pred_group.shape

# cal perf 
nnse_group = np.full((ngrid,nfeat,nmodel), np.nan)
r_group = np.full((ngrid,nfeat,nmodel), np.nan)
for i in range(ngrid):
    for m in range(nfeat):
        for k in range(nmodel):
            nnse_group[i,m,k] = nnse(obs[i,:,m],pred_group_mean[i,:,m,k])
            r_group[i,m,k] = np.corrcoef(obs[i,:,m],pred_group_mean[i,:,m,k])[0,1]


# --------------------------------------------------------------------------------------
# Table 1.
# --------------------------------------------------------------------------------------
nnse_group_mean_ngrid = np.nanmean(nnse_group,axis=(0)) 
nnse_group_mean_grid_mean_feat = np.nanmean(nnse_group_mean_ngrid,axis=(0),keepdims=True)
nnse_table = np.transpose(np.concatenate([nnse_group_mean_ngrid,nnse_group_mean_grid_mean_feat],axis=0),(1,0))
print(pd.DataFrame(nnse_table).round(3))
r_group_mean_ngrid = np.nanmean(r_group,axis=(0))
r_group_mean_grid_mean_feat = np.nanmean(r_group_mean_ngrid,axis=(0),keepdims=True) 
r_table = np.transpose(np.concatenate([r_group_mean_ngrid,r_group_mean_grid_mean_feat],axis=0),(1,0))
print(pd.DataFrame(r_table).round(3))
print('Table 1 finished!')


# --------------------------------------------------------------------------------------
# Figure 5.
# --------------------------------------------------------------------------------------
# init
rank_nnse = np.full((ngrid,nfeat+1),np.nan)
rank_r = np.full((ngrid,nfeat+1),np.nan)

# each feat
for i in range(ngrid):
    for k in range(nfeat):
        tmp_nnse,tmp_r = nnse_group[i,k,1:],r_group[i,k,1:]
        rank_nnse[i,k],rank_r[i,k] = np.argsort(tmp_nnse)[-1],np.argsort(tmp_r)[-1]

# mean
nnse_group_mean_feat = np.nanmean(nnse_group[:,:,1:], axis=1) # remove STL
r_group_mean_feat = np.nanmean(r_group[:,:,1:], axis=1) 
for i in range(ngrid):
    rank_nnse[i,-1] = np.argsort(nnse_group_mean_feat[i])[-1]
    rank_r[i,-1] = np.argsort(r_group_mean_feat[i])[-1]

# percentage
num_rank_nnse = np.full((nfeat+1,nmodel-1),np.nan)
num_rank_r = np.full((nfeat+1,nmodel-1),np.nan)
for k in range(nfeat+1):
    for m in range(nmodel-1):
        num_rank_nnse[k,m] = len(np.where(rank_nnse[:,k]==m)[0])/2 # /2 for %
        num_rank_r[k,m] = len(np.where(rank_r[:,k]==m)[0])/2

category_names = ['MTL UW','MTL(0.5)','MTL(1)','MTL(2.5)','MTL(5)','MTL(10)','MTL(20)']
results_nnse = {
    'SM_L1': num_rank_nnse[0],
    'SM_L2': num_rank_nnse[1],
    'SM_L3': num_rank_nnse[2],
    'ET': num_rank_nnse[3],
    'Q': num_rank_nnse[4],
    'Mean': num_rank_nnse[5]
}
results_r = {
    'SM_L1': num_rank_r[0],
    'SM_L2': num_rank_r[1],
    'SM_L3': num_rank_r[2],
    'ET': num_rank_r[3],
    'Q': num_rank_r[4],
    'Mean': num_rank_r[5]
}

fig = plt.figure(figsize=(30,10))
ax = plt.subplot(221)
labels = list(results_nnse.keys())
data = np.array(list(results_nnse.values()))
data_cum = data.cumsum(axis=1)
category_colors = plt.colormaps['RdYlGn'](np.linspace(0.15, 0.85, data.shape[1]))
ax.invert_yaxis()
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xticklabels(['','','','','',''])
ax.set_xlim(0, np.sum(data, axis=1).max())
for i, (colname, color) in enumerate(zip(category_names, category_colors)):
    widths = data[:, i]
    starts = data_cum[:, i] - widths
    rects = ax.barh(labels, widths, left=starts, height=0.5,
                    label=colname, color=color)
    r, g, b, _ = color
    text_color = 'black' if r * g * b < 0.5 else 'black'
    ax.bar_label(rects, label_type='center', color=text_color, fontsize=13)
ax.legend(bbox_to_anchor=(0.8, 0, 0.5, 0.5),loc='right', fontsize='small')
ax.set_xlabel('The percentage of grids perform the best (%)')
plt.savefig('figure5.pdf')
print('Figure 5 finished!')


# --------------------------------------------------------------------------------------
# Figure 6
# --------------------------------------------------------------------------------------
num = []
r_group_mean_feat = np.nanmean(nnse_group, axis=1)
for i in range(nmodel-1):
    a = r_group_mean_feat[:,i+1]
    b = r_group_mean_feat[:,0]
    num.append(len(np.where(a>=b)[0])/2)

fig = plt.figure(figsize=(15,5))
ax = plt.subplot(121)
rects = ax.bar(np.arange(7), np.array(num), color='gray')
ax.bar_label(rects, label_type='center', color='white', fontsize=13)
plt.axhline(50,linestyle='--',color='black')
ax.set_xticks([0,1,2,3,4,5,6])
ax.set_xticklabels(['']*7)
plt.ylabel('The percentage of grids (%)', fontsize=13)

d = r_group[:,:,-3]-r_group[:,:,0]
ax = plt.subplot(122)
for j in range(nfeat):
    tmp = d[:,j]
    tmp = np.delete(tmp, np.isnan(tmp))
    ax.boxplot(tmp,  positions=[j*0.4], notch=True,
            widths=0.2, whis=0.2, patch_artist=True, showfliers=False,
            boxprops=dict(facecolor='gray', color='gray'))
ax.boxplot(np.nanmean(d,axis=-1),positions=[2.0], notch=True,
            widths=0.2, whis=0.2, patch_artist=True, showfliers=False,
            boxprops=dict(facecolor='red', color='red'))
plt.axhline(0,linestyle='--',color='black')
ax.set_xticks([0,0.4,0.8,1.2,1.6,2.0])
ax.set_xticklabels(['']*6)
plt.ylabel('The difference of performance', fontsize=13)
plt.savefig('figure6.pdf')
print('Figure 6 finished!')


# --------------------------------------------------------------------------------------
# Figure S1.
# --------------------------------------------------------------------------------------
til = ['(a) Lambda = 1','(b) Lambda = 2.5','(c) Lambda = 5','(d) Lambda = 10']
fig = plt.figure(figsize=(15,15))
for i in range(4):
    ax = plt.subplot(3,2,i+1)
    d = r_group[:,:,i+3]-r_group[:,:,0]
    for j in range(5):
        tmp = d[:,j]
        tmp = np.delete(tmp, np.isnan(tmp))
        ax.boxplot(tmp,  positions=[j*0.4], notch=True,
                widths=0.2, whis=0.2, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='gray', color='gray'))
    ax.boxplot(np.nanmean(d,axis=-1),positions=[2.0], notch=True,
                widths=0.2, whis=0.2, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='red', color='red'))
    plt.axhline(0,linestyle='--',color='black')
    ax.set_xticks([0,0.4,0.8,1.2,1.6,2.0])
    ax.set_xticklabels(['']*6)
    plt.ylabel('The difference of performance', fontsize=13)
    plt.title(til[i])
plt.savefig('figureS1.pdf')
print('Figure S1 finished!')