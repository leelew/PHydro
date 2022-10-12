import imp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from config import parse_args
import os
from utils import unbiased_RMSE
from sklearn.metrics import r2_score
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from mpl_toolkits.basemap import Basemap

# get config
cfg = parse_args()

# load forecast and observation (ngrids, 366, 6)
path = cfg["outputs_path"]+'forecast/'+cfg["model_name"]+'/'
y_pred = np.load(path+cfg["model_name"]+'_guangdong_9km.npy')
path = cfg["inputs_path"]
y_test = np.load(path+'y_test.npy')

# get shape
ngrids, nt, nfeat = y_pred.shape

# load (ngrids, 2) lon,lat
attr = np.load(path+"guangdong_9km_attribute.npy")
lon_min, lat_min = np.nanmin(attr, axis=0)
lon_max, lat_max = np.nanmax(attr, axis=0)
ease_9km_grids = np.load(path+"EASE_9km_attribute.npy") #(1800, 3600, 2)
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
            urmse[i, t] = unbiased_RMSE(y_test[i, :, t], y_pred[i, :, t])
            r2[i, t] = r2_score(y_test[i, :, t], y_pred[i, :, t])

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

# save
np.save('r2_'+cfg["model_name"]+'.npy', r2)
np.save('urmse_'+cfg["model_name"]+'.npy', urmse)
path = cfg["outputs_path"]+'forecast/'+cfg["model_name"]+'/'
os.system('mv {} {}'.format('r2_'+cfg["model_name"]+'.npy', path))
os.system('mv {} {}'.format('urmse_'+cfg["model_name"]+'.npy', path))

# cal physical consistency


# preliminary plot
# figure 1 (countour)
r2_grid = site2grid(r2, attr[:,1], attr[:,0], lat_gd, lon_gd)
var_list = ["0-7cm soil moisture", "7-28cm soil moisture", "28-100cm soil moisture", "100-286.46cm soil moisture","evapotranspiration","total runoff"]
text = ["(a)","(b)","(c)","(d)","(e)","(f)"]
fig = plt.figure(figsize=(8,10))

for i in range(cfg["num_out"]):
    ax = plt.subplot(3, 2, i+1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)

    m = Basemap(projection='mill',
            llcrnrlat=lat_min, urcrnrlat=lat_max,
            llcrnrlon=lon_min, urcrnrlon=lon_max,
            ax=ax)  # mill projection
    #m.drawcoastlines(linewidth=1)
    x, y = m(grid_lon, grid_lat)
    sc = m.pcolormesh(x, y, r2_grid[:,:,i] ,vmin=0, vmax=1, cmap='jet')
    plt.title(text[i]+' '+var_list[i])
cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
fig.colorbar(sc, cax=cbar_ax, orientation="horizontal",spacing='proportional')
plt.suptitle("R2 of single task model")
plt.savefig("figure1.pdf")

# figure 2 (boxplot)

# figure 3 (spatial mean) 
fig = plt.figure(figsize=(10,18))
for i in range(cfg["num_out"]):
    ax = plt.subplot(6, 1, i+1)
    plt.plot(np.nanmean(y_pred[:,:,i], axis=0))
    plt.plot(np.nanmean(y_test[:,:,i], axis=0))
    plt.title(text[i]+' '+var_list[i])
    plt.legend(['single_task','CoLM'])

plt.savefig("figure3.pdf")
# figure 4 (temporal mean)



               


