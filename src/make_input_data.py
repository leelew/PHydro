import os

import netCDF4 as nc
import numpy as np
import shapefile
import shapely.geometry as geometry
import xarray as xr

from utils import h2osoi_colm2era


# ------------------------------------------------------------------------------
# params
# ------------------------------------------------------------------------------
years = np.arange(2017, 2020)  # 2000-2016
months = np.arange(1,12)  # 1-12
inputs_path = "/tera05/lilu/PHydro/input/"  # path contain inputs
mode = 'train'


# ------------------------------------------------------------------------------
# extract guangdong points
# ------------------------------------------------------------------------------
"""
# read lat/lon in CoLM outputs
with xr.open_dataset(PATH+"global_grid_ERA5_pft_GD_hist_2015-01.nc", decode_times=False) as f:
    lat, lon = np.array(f.lat), np.array(f.lon)
    rnof = np.array(f.f_rnof)

# shapefile input
gd_shp = shapefile.Reader("/tera05/lilu/PHydro/data/shapefile/guangdong.shp")
feat = gd_shp.shapeRecords()[0]
first = feat.shape
grid_lon, grid_lat = np.meshgrid(lon, lat)
flat_lon, flat_lat = grid_lon.flatten(), grid_lat.flatten()
flat_points = np.column_stack((flat_lon, flat_lat))

in_shape_idx = np.zeros_like(flat_lat)
in_shape_points = []
count = 0
for i, pt in enumerate(flat_points):
    print(i)
    if geometry.Point(pt).within(geometry.shape(first)):
        in_shape_points.append(pt)
        in_shape_idx[i] = 1
        count += 1
idx = np.where(in_shape_idx == 1)[0]

# remove NaN points in runoff outputs
rnof = rnof.reshape(rnof.shape[0], -1)[:, idx] # time, grid
remove_idx = np.unique(np.where(np.isnan(rnof))[-1])
idx = np.delete(idx, [remove_idx])
print(idx.shape)

np.save("gd_9km_shapefile.npy", idx)
"""
idx = np.load(inputs_path+"gd_9km_shapefile.npy")


# ------------------------------------------------------------------------------
# read ancillary variables
# ------------------------------------------------------------------------------
PATH = "/tera05/lilu/PHydro/data/"
with xr.open_dataset(PATH+'ancillary/Beck_KG_V1_present_0p1.nc') as f:
    lat2, lon2 = np.array(f.lat), np.array(f.lon)
    lat_idx = np.where((lat2>18) & (lat2<28))[0]
    lon_idx = np.where((lon2>108) & (lon2<119))[0]
    climate_zone = np.array(f.Band1)
with xr.open_dataset(PATH+'ancillary/DEM_0p1.nc') as f:
    dem = np.array(f.elv)
with xr.open_dataset(PATH+'ancillary/kosugi_0p1.nc') as f:
    kosugi = np.array(f.kosugi)
with xr.open_dataset(PATH+'ancillary/LC_0p1.nc') as f:
    land_cover = np.array(f["class"])
with xr.open_dataset(PATH+'ancillary/soilgrid_0p1.nc') as f:
    soilgrid = np.array(f.soilgrids)
tmp = np.stack([climate_zone, dem, land_cover, soilgrid[0], soilgrid[1],
                soilgrid[7], soilgrid[8], soilgrid[14], soilgrid[15]], axis=0)
static = np.concatenate([tmp, kosugi], axis=0)
static = static[:,lat_idx, :][:,:,lon_idx]
static = np.transpose(static, (1, 2, 0))  # (lat, lon, feats)
static = np.flip(static, axis=0)
Nlat, Nlon, Nfeat = static.shape
static = static.reshape(-1, Nfeat)[idx, :]
print(static.shape)

for i in range(Nfeat):
    tmp = static[:,i]
    tmp[np.isnan(tmp)] = np.nanmean(tmp)
    static[:,i] = tmp

print(np.isnan(static).any())
np.save("gd_9km_ancillary.npy", static)


# ------------------------------------------------------------------------------
# read CoLM hydrology & forcing variables
# ------------------------------------------------------------------------------
PATH = "/tera04/zhwei/cases/global_grid_ERA5_pft_GD/history/"  # path contain raw data
forcing = []
hydrology = []
for year in years:
    for month in months:
        print(year, month)
        file_name = "global_grid_ERA5_pft_GD_hist_{year}-{month:02}.nc".format(
            year=year, month=month)
        if os.path.exists(PATH+file_name):
            with xr.open_dataset(PATH+file_name, decode_times=False) as f:
                # hydrology
                evpa = np.array(f.f_fevpa)  # mm/s
                rnof = np.array(f.f_rnof)  # mm/s
                h2osoi = np.array(f.f_h2osoi)  # m3/m3
                wa = np.array(f.f_wa) #mm
                ldew = np.array(f.f_ldew) #mm
                scv = np.array(f.f_scv) #mm
                xerr = np.array(f.f_xerr) # mm/s
                wice_soisno = np.array(f.f_wice_soisno)[:,:,:,5:] #kg/m2
                wliq_soisno = np.array(f.f_wliq_soisno)[:,:,:,5:] #kg/m2
                swvl1, swvl2, swvl3, swvl4, zi = h2osoi_colm2era(wice_soisno, wliq_soisno) # m3/m3

                # forcing
                prc = np.array(f.f_xy_prc)  # mm/s convective
                prl = np.array(f.f_xy_prl)  # mm/s large scale
                us = np.array(f.f_xy_us)  # m/s
                vs = np.array(f.f_xy_vs)  # m/s
                t = np.array(f.f_xy_t)  # kelvin
                q = np.array(f.f_xy_q)  # kg/kg
                pbot = np.array(f.f_xy_pbot)  # pa
                fr = np.array(f.f_xy_frl)  # W/m2
                solarin = np.array(f.f_xy_solarin)  # W/m2

                #p = swvl1*70+swvl2*210+swvl3*720+swvl4*(zi[9]-100)*10+wa+scv+ldew
            # turn all hydrology variables to mm/hour
            evpa = evpa*60*60  # mm
            rnof = rnof*60*60  # mm
            prc = prc*60*60  # mm
            prl = prl*60*60  # mm
            xerr = xerr*60*60

            tmp = np.stack([prc, prl, t, q, pbot, fr, solarin, us, vs], axis=-1)
            print(tmp.shape)
            forcing.append(tmp)
            tmp = np.stack([swvl1, swvl2, swvl3, swvl4, evpa, rnof], axis=-1)
            hydrology.append(tmp)

# check physical consistency
#m = np.diff(p, axis=0)
#d = m-(prc[1:]+prl[1:]-evpa[1:]-rnof[1:]+xerr[1:])
#mean_phy = np.nanmean(np.abs(d), axis=0)
#mean_phy = mean_phy.reshape(-1,)[idx]
#mask_unphy = np.ones_like(mean_phy)
#mask_unphy[mean_phy>0.1] = 0
#idx = np.delete(idx, np.where(mask_unphy==0)[0])
print(len(forcing))
forcing = np.concatenate(forcing, axis=0)
Nt, Nlat, Nlon, Nfeat = forcing.shape
forcing = forcing.reshape(Nt, -1, Nfeat)[:, idx, :]
np.save("gd_9km_forcing_{}.npy".format(mode), forcing)
print(forcing.shape)
del forcing
hydrology = np.concatenate(hydrology, axis=0)
Nt, Nlat, Nlon, Nfeat = hydrology.shape
hydrology = hydrology.reshape(Nt, -1, Nfeat)[:, idx, :]
np.save("gd_9km_hydrology_{}.npy".format(mode), hydrology)
print(hydrology.shape)
del hydrology


# ------------------------------------------------------------------------------
# move to input path
# ------------------------------------------------------------------------------
np.save("gd_9km_shapefile.npy", idx)
os.system("mv {} {}".format('gd_9km*npy', inputs_path))



