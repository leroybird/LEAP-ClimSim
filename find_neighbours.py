# %%
from math import dist
from typing import Tuple
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from geopy.distance import geodesic
import pyproj

# %%
grid_path = "/mnt/storage/kaggle/ClimSim_low-res_grid-info.nc"
# %%
with xr.open_dataset(grid_path) as ds:
    ds = ds.load()

# %%
lats, lons = ds["lat"].values, ds["lon"].values
# %%
plt.plot(
    lats,
)
plt.plot(lons)


def calculate_relative_distances(point1, point2, center_point):
    # Define the Lambert Conformal Conic projection centered on the given latitude and longitude
    proj_string = (
        f"+proj=lcc +lat_1={center_point[0]} +lat_2={center_point[0]} "
        f"+lat_0={center_point[0]} +lon_0={center_point[1]} "
        f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    custom_proj = pyproj.CRS.from_proj4(proj_string)

    # Define the WGS84 coordinate system
    wgs84 = pyproj.CRS("EPSG:4326")

    # Create a transformer object
    transformer = pyproj.Transformer.from_crs(wgs84, custom_proj, always_xy=True)

    # Transform the points to the custom projection coordinates
    easting1, northing1 = transformer.transform(point1[1], point1[0])
    easting2, northing2 = transformer.transform(point2[1], point2[0])

    # Calculate the relative distances
    x_distance = easting2 - easting1
    y_distance = northing2 - northing1

    return x_distance, y_distance


def get_distances(t_lat, t_lon, lats, lons):
    distances = []
    for lat2, lon2 in zip(lats, lons):
        d = geodesic((t_lat, t_lon), (lat2, lon2)).kilometers
        distances.append(d)

    return np.array(distances)


# %%
output_ds = {
    "lats": ("idx", lats),
    "lons": ("idx", lons),
    # Store the 9 closest points for each target point
    "distances": (("idx", "closest_points"), []),
    "x_distances": (("idx", "closest_points"), []),
    "y_distances": (("idx", "closest_points"), []),
    "idxs": (("idx", "closest_points"), []),
}

for idx in range(len(lats)):

    distances = get_distances(lats[idx], lons[idx], lats, lons)
    idxs = distances.argsort()[0:9]
    closes_distances = distances[idxs]

    x_dist, y_dist = calculate_relative_distances(
        (lats[idx], lons[idx]), (lats[idxs], lons[idxs]), (lats[idx], lons[idx])
    )

    output_ds["distances"][-1].append(closes_distances)
    output_ds["x_distances"][-1].append(x_dist)
    output_ds["y_distances"][-1].append(y_dist)
    output_ds["idxs"][-1].append(idxs)

    # plt.scatter(x_dist, y_dist, color=["red" if i == idx else "blue" for i in idxs],
    #             )
    # plt.grid()
    # plt.show()
# %%
ds_out = xr.Dataset(
    output_ds,
    coords={"idx": range(len(lats)), "closest_points": range(9)},
)
#%%
ds_out
# %%
ds_out.to_netcdf("neighbours.nc")
#%%
