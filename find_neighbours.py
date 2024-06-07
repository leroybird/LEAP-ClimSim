#%%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
#%%
grid_path = "/mnt/storage/kaggle/ClimSim_low-res_grid-info.nc"
#%%
with xr.open_dataset(grid_path) as ds:
    ds = ds.load()
    
ds
#%%
lats, lons = ds["lat"].values, ds["lon"].values
#%%
plt.plot(lats,)
plt.plot(lons)
#%%
def calc_dist(t_lat, t_lon, lats, lons):
    # calculate the distance between the target point and all other points
    # convert degrees to radians
    t_lat, t_lon = np.radians(t_lat), np.radians(t_lon)
    lats, lons = np.radians(lats), np.radians(lons)
    
    

#%%