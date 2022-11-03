import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from mpl_toolkits.basemap import Basemap
import numpy as np
from config import parse_args

cfg = parse_args()
# ---------------------------------
# Figure 1
# ---------------------------------
path = cfg["outputs_path"]+'forecast/'
r2_single_task = np.load(path+"single_task_epoch_200/r2_single_task.npy")
r2_multi_tasks = np.load(path+"multi_tasks_epoch_200/r2_multi_tasks.npy")

phy_single_task = np.load(path+"single_task_epoch_200/phy_cons_single_task.npy")
phy_multi_tasks = np.load(path+"multi_tasks_epoch_200/phy_cons_multi_tasks.npy")

r2_st, r2_mt = [], []
phy_st, phy_mt = [], []
for i in range(6):
    temp = r2_multi_tasks[:,:,i].reshape(-1, )
    temp = np.delete(temp, np.isnan(temp))
    print(temp)
    r2_mt.append(temp)
    temp = r2_single_task[:,:,i].reshape(-1, )
    temp = np.delete(temp, np.isnan(temp))    
    r2_st.append(temp)

temp = phy_multi_tasks.reshape(-1, )
temp = np.delete(temp, np.isnan(temp))
phy_mt.append(temp)
temp = phy_single_task.reshape(-1, )
temp = np.delete(temp, np.isnan(temp))    
phy_st.append(temp)

#figure
fig = plt.figure()
ax = plt.subplot(211)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.boxplot(r2_st,
            notch=True,
            patch_artist=True,
            showfliers=False,
            positions=[0, 1, 2, 3, 4, 5],
            boxprops=dict(facecolor='lightblue', color='black'))
ax.boxplot(r2_mt,notch=True,
            patch_artist=True,
            showfliers=False,
            positions=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            boxprops=dict(facecolor='red', color='black'))

ax = plt.subplot(212)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.boxplot(phy_st,
            notch=True,
            patch_artist=True,
            showfliers=False,
            positions=[0],
            boxprops=dict(facecolor='lightblue', color='black'))
ax.boxplot(phy_mt,notch=True,
            patch_artist=True,
            showfliers=False,
            positions=[0.5],
            boxprops=dict(facecolor='red', color='black'))
plt.savefig('figure1.pdf')
print('Figure 1 completed!')



"""
m = Basemap(projection='mill',
        llcrnrlat=lat_min-1, urcrnrlat=lat_max+1,
        llcrnrlon=lon_min-1, urcrnrlon=lon_max+1,
        ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(grid_lon, grid_lat)
sc = m.pcolormesh(x, y, phy_cons_grid[:,:,0], cmap='jet')
m.readshapefile(cfg["inputs_path"]+"guangdong_shp/guangdong", "guangdong", drawbounds=True)
cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
fig.colorbar(sc, cax=cbar_ax, orientation="horizontal",spacing='proportional')
plt.suptitle("Physical consistency of model")
plt.savefig("figure2.pdf")


# preliminary plot
# figure 1 (countour)
print(r2.shape, attr.shape, lat_gd.shape, lon_gd.shape)
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
            llcrnrlat=lat_min-1, urcrnrlat=lat_max+1,
            llcrnrlon=lon_min-1, urcrnrlon=lon_max+1,
            ax=ax)  # mill projection
    #m.drawcoastlines(linewidth=1)
    x, y = m(grid_lon, grid_lat)
    sc = m.pcolormesh(x, y, r2_grid[:,:,i] ,vmin=0.6, vmax=1, cmap='jet')
    m.readshapefile(cfg["inputs_path"]+"guangdong_shp/guangdong", "guangdong", drawbounds=True)
    plt.title(text[i]+' '+var_list[i])
cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
fig.colorbar(sc, cax=cbar_ax, orientation="horizontal",spacing='proportional')
plt.suptitle("R2 of single task model")
plt.savefig("figure1.pdf")

# figure 3 (spatial mean) 
fig = plt.figure(figsize=(10,18))
for i in range(cfg["num_out"]):
    ax = plt.subplot(6, 1, i+1)
    plt.plot(np.nanmean(y_pred[:,:,i], axis=0))
    plt.plot(np.nanmean(y_test[:,:,i], axis=0))
    plt.title(text[i]+' '+var_list[i])
    plt.legend(['single_task','CoLM'])

plt.savefig("figure3.pdf")
"""