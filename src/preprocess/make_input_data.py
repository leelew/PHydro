import glob
import os

import netCDF4 as nc
import numpy as np
import shapefile
import shapely.geometry as geometry
import xarray as xr

from colm2era import h2osoi_colm2era

# ------------------------------------------------------------------------------
# params
# ------------------------------------------------------------------------------
years = np.arange(2019, 2021)  # 2000-2016
months = np.arange(1, 13)  # 1-12
PATH = "/tera05/lilu/PHydro/data/"  # path contain raw data
inputs_path = "/tera05/lilu/PHydro/input/"  # path contain inputs
mode = 'test'
# ------------------------------------------------------------------------------
# extract guangdong points
# ------------------------------------------------------------------------------
in_shape_idx = np.load(inputs_path+"guangdong_9km_shapefile.npy")
idx = np.where(in_shape_idx == 1)[0]
# FIXME: Remove grids contains all NaN automatically
idx1 = np.delete(idx, [1220,1544,1550,1563])


# codes for extract guangdong points
with xr.open_dataset(PATH+"CoLM/global_MSWX_9km_hist_2000-01.nc", decode_times=False) as f:
    lat, lon = np.array(f.lat), np.array(f.lon)

gd_shp = shapefile.Reader(PATH+"shapefile/guangdong.shp")
feat = gd_shp.shapeRecords()[0]
first = feat.shape
grid_lon, grid_lat = np.meshgrid(lon, lat)
flat_lon, flat_lat = grid_lon.flatten(), grid_lat.flatten()
flat_points = np.column_stack((flat_lon, flat_lat))

in_shape_idx = np.zeros_like(flat_lat)
in_shape_points = []
count = 0
for i, pt in enumerate(flat_points):
    if geometry.Point(pt).within(geometry.shape(first)):
        in_shape_points.append(pt)
        in_shape_idx[i] = 1
        count += 1
idx = np.where(in_shape_idx == 1)[0]
np.save("guangdong_9km_shapefile.npy", in_shape_idx)



# ------------------------------------------------------------------------------
# read CoLM hydrology & forcing variables
# ------------------------------------------------------------------------------
forcing = []
hydrology = []
for year in years:
    for month in months:
        print(year, month)
        file_name = "global_MSWX_9km_hist_{year}-{month:02}.nc".format(
            year=year, month=month)
        if os.path.exists(PATH+'CoLM/'+file_name):
            with xr.open_dataset(PATH+'CoLM/'+file_name, decode_times=False) as f:
                # hydrology
                evpa = np.array(f.f_fevpa)  # mm/s
                rnof = np.array(f.f_rnof)  # mm/s
                h2osoi = np.array(f.f_h2osoi)  # m3/m3
                swvl1, swvl2, swvl3, swvl4 = h2osoi_colm2era(h2osoi)  # m3/m3

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

            # turn all hydrology variables to mm/day
            evpa = evpa*24*60*60  # mm
            rnof = rnof*24*60*60  # mm
            prc = prc*24*60*60  # mm
            prl = prl*24*60*60  # mm
        tmp = np.stack([prc, prl, t, q, pbot, fr, solarin, us, vs], axis=-1)
        Nt, Nlat, Nlon, Nfeat = tmp.shape
        tmp = tmp.reshape(Nt, -1, Nfeat)[:, idx, :]
        forcing.append(tmp)
        tmp = np.stack([swvl1, swvl2, swvl3, swvl4, evpa, rnof], axis=-1)
        Nt, Nlat, Nlon, Nfeat = tmp.shape
        tmp = tmp.reshape(Nt, -1, Nfeat)[:, idx, :]
        hydrology.append(tmp)

forcing = np.concatenate(forcing, axis=0)
np.save("guangdong_9km_forcing_{}.npy".format(mode), forcing)
print(forcing.shape)
del forcing
hydrology = np.concatenate(hydrology, axis=0)
np.save("guangdong_9km_hydrology_{}.npy".format(mode), hydrology)
print(hydrology.shape)
del hydrology


# ------------------------------------------------------------------------------
# read ancillary variables
# ------------------------------------------------------------------------------
with xr.open_dataset(PATH+'ancillary/Beck_KG_V1_present_0p1.nc') as f:
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
static = np.transpose(static, (1, 2, 0))  # (lat, lon, feats)
static = np.flip(static, axis=0)
Nlat, Nlon, Nfeat = static.shape
static = static.reshape(-1, Nfeat)[idx, :]
print(static.shape)

for i in range(Nfeat):
    tmp = static[:,i]
    tmp[np.isnan(tmp)] = np.nanmean(tmp)
    static[:,i] = tmp

np.save("guangdong_9km_ancillary.npy", static)


# ------------------------------------------------------------------------------
# move to input path
# ------------------------------------------------------------------------------
os.system("mv {} {}".format('guangdong*npy', inputs_path))



