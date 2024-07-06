from re import I
import torch
from pathlib import Path
import xarray as xr
import dataloader
import numpy as np
import pandas as pd
import config
import logging
import polars as pl
from torch.utils.data import DataLoader
import norm


def get_static(grid_info):
    lat, lon = grid_info["lat"].values, grid_info["lon"].values
    lon1, lon2 = np.cos(np.deg2rad(lon)), np.sin(np.deg2rad(lon))
    lat1, lat2 = np.cos(np.deg2rad(2 * lat)), np.sin(np.deg2rad(2 * lat))
    area_weight = grid_info["area"].values / grid_info["area"].values.mean()
    area = 10 * (area_weight - 1.0)

    static_data = np.stack([lon1, lon2, lat1, lat2, area], axis=1)
    return static_data


class LeapLoader:
    def __init__(
        self,
        root_folder: Path,
        grid_info_path,
        df,
        grid_neighbours_path=Path("__file__").parent / "neighbours.nc",
        x_transform=None,
        y_transform=None,
        add_static=False,
        muti_step=True,
    ):
        self.root_folder = root_folder
        self.df = df
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.add_static = add_static
        self.multi_step = muti_step

        self.data_path = None
        self.input_vars = []
        self.target_vars = []
        self.input_feature_len = None
        self.target_feature_len = None

        self.grid_info = xr.open_dataset(grid_info_path).load()
        self.grid_info.close()

        self.neighbours = xr.open_dataset(grid_neighbours_path).load()
        self.neighbours.close()

        self.level_name = "lev"
        self.sample_name = "sample"
        self.num_levels = len(self.grid_info["lev"])
        self.num_latlon = len(
            self.grid_info["ncol"]
        )  # number of unique lat/lon grid points

        # make area-weights
        self.grid_info["area_wgt"] = self.grid_info["area"] / self.grid_info["area"].mean(
            dim="ncol"
        )
        self.area_wgt = self.grid_info["area_wgt"].values
        # map ncol to nsamples dimension
        # to_xarray = {'area_wgt':(self.sample_name,np.tile(self.grid_info['area_wgt'], int(n_samples/len(self.grid_info['ncol']))))}
        # to_xarray = xr.Dataset(to_xarray)
        self.normalize = True
        self.lats, self.lats_indices = np.unique(
            self.grid_info["lat"].values, return_index=True
        )
        self.lons, self.lons_indices = np.unique(
            self.grid_info["lon"].values, return_index=True
        )
        self.sort_lat_key = np.argsort(
            self.grid_info["lat"].values[np.sort(self.lats_indices)]
        )
        self.sort_lon_key = np.argsort(
            self.grid_info["lon"].values[np.sort(self.lons_indices)]
        )
        self.indextolatlon = {
            i: (
                self.grid_info["lat"].values[i % self.num_latlon],
                self.grid_info["lon"].values[i % self.num_latlon],
            )
            for i in range(self.num_latlon)
        }

        def find_keys(dictionary, value):
            keys = []
            for key, val in dictionary.items():
                if val[0] == value:
                    keys.append(key)
            return keys

        indices_list = []
        for lat in self.lats:
            indices = find_keys(self.indextolatlon, lat)
            indices_list.append(indices)
        indices_list.sort(key=lambda x: x[0])
        self.lat_indices_list = indices_list

        self.hyam = self.grid_info["hyam"].values
        self.hybm = self.grid_info["hybm"].values
        self.p0 = 1e5  # code assumes this will always be a scalar
        self.ps_index = None

        self.pressure_grid_train = None
        self.pressure_grid_val = None
        self.pressure_grid_scoring = None
        self.pressure_grid_test = None

        self.dp_train = None
        self.dp_val = None
        self.dp_scoring = None
        self.dp_test = None

        self.train_regexps = None
        self.train_stride_sample = None
        self.train_filelist = None
        self.val_regexps = None
        self.val_stride_sample = None
        self.val_filelist = None
        self.scoring_regexps = None
        self.scoring_stride_sample = None
        self.scoring_filelist = None
        self.test_regexps = None
        self.test_stride_sample = None
        self.test_filelist = None

        self.grid_size = 384

        self.full_vars = False

        # physical constants from E3SM_ROOT/share/util/shr_const_mod.F90
        self.grav = 9.80616  # acceleration of gravity ~ m/s^2
        self.cp = 1.00464e3  # specific heat of dry air   ~ J/kg/K
        self.lv = 2.501e6  # latent heat of evaporation ~ J/kg
        self.lf = 3.337e5  # latent heat of fusion      ~ J/kg
        self.lsub = self.lv + self.lf  # latent heat of sublimation ~ J/kg
        self.rho_air = (
            101325 / (6.02214e26 * 1.38065e-23 / 28.966) / 273.15
        )  # density of dry air at STP  ~ kg/m^3
        # ~ 1.2923182846924677
        # SHR_CONST_PSTD/(SHR_CONST_RDAIR*SHR_CONST_TKFRZ)
        # SHR_CONST_RDAIR   = SHR_CONST_RGAS/SHR_CONST_MWDAIR
        # SHR_CONST_RGAS    = SHR_CONST_AVOGAD*SHR_CONST_BOLTZ
        self.rho_h20 = 1.0e3  # density of fresh water     ~ kg/m^ 3

        self.v1_inputs = [
            "state_t",
            "state_q0001",
            "state_ps",
            "pbuf_SOLIN",
            "pbuf_LHFLX",
            "pbuf_SHFLX",
        ]

        self.v1_outputs = [
            "ptend_t",
            "ptend_q0001",
            "cam_out_NETSW",
            "cam_out_FLWDS",
            "cam_out_PRECSC",
            "cam_out_PRECC",
            "cam_out_SOLS",
            "cam_out_SOLL",
            "cam_out_SOLSD",
            "cam_out_SOLLD",
        ]

        self.v2_inputs = [
            "state_t",
            "state_q0001",
            "state_q0002",
            "state_q0003",
            "state_u",
            "state_v",
            "state_ps",
            "pbuf_SOLIN",
            "pbuf_LHFLX",
            "pbuf_SHFLX",
            "pbuf_TAUX",
            "pbuf_TAUY",
            "pbuf_COSZRS",
            "cam_in_ALDIF",
            "cam_in_ALDIR",
            "cam_in_ASDIF",
            "cam_in_ASDIR",
            "cam_in_LWUP",
            "cam_in_ICEFRAC",
            "cam_in_LANDFRAC",
            "cam_in_OCNFRAC",
            # "cam_in_SNOWHICE",
            "cam_in_SNOWHLAND",
            "pbuf_ozone",  # outside of the upper troposphere lower stratosphere (UTLS, corresponding to indices 5-21), variance in minimal for these last 3
            "pbuf_CH4",
            "pbuf_N2O",
        ]

        self.v2_outputs = [
            "ptend_t",
            "ptend_q0001",
            "ptend_q0002",
            "ptend_q0003",
            "ptend_u",
            "ptend_v",
            "cam_out_NETSW",
            "cam_out_FLWDS",
            "cam_out_PRECSC",
            "cam_out_PRECC",
            "cam_out_SOLS",
            "cam_out_SOLL",
            "cam_out_SOLSD",
            "cam_out_SOLLD",
        ]

        self.var_lens = {  # inputs
            "state_t": self.num_levels,
            "state_q0001": self.num_levels,
            "state_q0002": self.num_levels,
            "state_q0003": self.num_levels,
            "state_u": self.num_levels,
            "state_v": self.num_levels,
            "state_ps": 1,
            "pbuf_SOLIN": 1,
            "pbuf_LHFLX": 1,
            "pbuf_SHFLX": 1,
            "pbuf_TAUX": 1,
            "pbuf_TAUY": 1,
            "pbuf_COSZRS": 1,
            "cam_in_ALDIF": 1,
            "cam_in_ALDIR": 1,
            "cam_in_ASDIF": 1,
            "cam_in_ASDIR": 1,
            "cam_in_LWUP": 1,
            "cam_in_ICEFRAC": 1,
            "cam_in_LANDFRAC": 1,
            "cam_in_OCNFRAC": 1,
            "cam_in_SNOWHICE": 1,
            "cam_in_SNOWHLAND": 1,
            "pbuf_ozone": self.num_levels,
            "pbuf_CH4": self.num_levels,
            "pbuf_N2O": self.num_levels,
            # outputs
            "ptend_t": self.num_levels,
            "ptend_q0001": self.num_levels,
            "ptend_q0002": self.num_levels,
            "ptend_q0003": self.num_levels,
            "ptend_u": self.num_levels,
            "ptend_v": self.num_levels,
            "cam_out_NETSW": 1,
            "cam_out_FLWDS": 1,
            "cam_out_PRECSC": 1,
            "cam_out_PRECC": 1,
            "cam_out_SOLS": 1,
            "cam_out_SOLL": 1,
            "cam_out_SOLSD": 1,
            "cam_out_SOLLD": 1,
        }

        self.var_short_names = {
            "ptend_t": "$dT/dt$",
            "ptend_q0001": "$dq/dt$",
            "cam_out_NETSW": "NETSW",
            "cam_out_FLWDS": "FLWDS",
            "cam_out_PRECSC": "PRECSC",
            "cam_out_PRECC": "PRECC",
            "cam_out_SOLS": "SOLS",
            "cam_out_SOLL": "SOLL",
            "cam_out_SOLSD": "SOLSD",
            "cam_out_SOLLD": "SOLLD",
        }

        self.target_energy_conv = {
            "ptend_t": self.cp,
            "ptend_q0001": self.lv,
            "ptend_q0002": self.lv,
            "ptend_q0003": self.lv,
            "ptend_wind": None,
            "cam_out_NETSW": 1.0,
            "cam_out_FLWDS": 1.0,
            "cam_out_PRECSC": self.lv * self.rho_h20,
            "cam_out_PRECC": self.lv * self.rho_h20,
            "cam_out_SOLS": 1.0,
            "cam_out_SOLL": 1.0,
            "cam_out_SOLSD": 1.0,
            "cam_out_SOLLD": 1.0,
        }

        # for metrics

        self.input_train = None
        self.target_train = None
        self.preds_train = None
        self.samplepreds_train = None
        self.target_weighted_train = {}
        self.preds_weighted_train = {}
        self.samplepreds_weighted_train = {}
        self.metrics_train = []
        self.metrics_idx_train = {}
        self.metrics_var_train = {}

        self.input_val = None
        self.target_val = None
        self.preds_val = None
        self.samplepreds_val = None
        self.target_weighted_val = {}
        self.preds_weighted_val = {}
        self.samplepreds_weighted_val = {}
        self.metrics_val = []
        self.metrics_idx_val = {}
        self.metrics_var_val = {}

        self.input_scoring = None
        self.target_scoring = None
        self.preds_scoring = None
        self.samplepreds_scoring = None
        self.target_weighted_scoring = {}
        self.preds_weighted_scoring = {}
        self.samplepreds_weighted_scoring = {}
        self.metrics_scoring = []
        self.metrics_idx_scoring = {}
        self.metrics_var_scoring = {}

        self.input_test = None
        self.target_test = None
        self.preds_test = None
        self.samplepreds_test = None
        self.target_weighted_test = {}
        self.preds_weighted_test = {}
        self.samplepreds_weighted_test = {}
        self.metrics_test = []
        self.metrics_idx_test = {}
        self.metrics_var_test = {}

        self.model_names = []
        self.metrics_names = []

        self.num_CRPS = 32

        self.set_to_v2_vars()

    def set_to_v2_vars(self):
        """
        This function sets the inputs and outputs to the V2 subset.
        It also indicates the index of the surface pressure variable.
        """
        self.input_vars = self.v2_inputs
        self.target_vars = self.v2_outputs
        self.ps_index = 360
        self.input_feature_len = 557
        self.target_feature_len = 368
        self.full_vars = True

    def get_xrdata(self, file, file_vars=None):
        """
        This function reads in a file and returns an xarray dataset with the variables specified.
        file_vars must be a list of strings.
        """
        ds = xr.open_dataset(file, engine="netcdf4")
        if file_vars is not None:
            ds = ds[file_vars]
        ds = ds.load()
        ds.close()

        ds = ds.merge(self.grid_info[["lat", "lon"]])
        ds = ds.where((ds["lat"] > -999) * (ds["lat"] < 999), drop=True)
        ds = ds.where((ds["lon"] > -999) * (ds["lon"] < 999), drop=True)
        return ds

    def get_input(self, input_file):
        """
        This function reads in a file and returns an xarray dataset with the input variables for the emulator.
        """
        # read inputs
        return self.get_xrdata(input_file, self.input_vars)

    def get_pair(self, input_file):
        """
        This function reads in a file and returns an xarray dataset with the target variables for the emulator.
        """
        # read inputs
        ds_input = self.get_input(input_file)
        ds_target = self.get_xrdata(
            input_file.parent / input_file.name.replace(".mli.", ".mlo.")
        )

        # each timestep is 20 minutes which corresponds to 1200 seconds
        ds_target["ptend_t"] = (
            ds_target["state_t"] - ds_input["state_t"]
        ) / 1200  # T tendency [K/s]
        ds_target["ptend_q0001"] = (
            ds_target["state_q0001"] - ds_input["state_q0001"]
        ) / 1200  # Q tendency [kg/kg/s]
        if self.full_vars:
            ds_target["ptend_q0002"] = (
                ds_target["state_q0002"] - ds_input["state_q0002"]
            ) / 1200  # Q tendency [kg/kg/s]
            ds_target["ptend_q0003"] = (
                ds_target["state_q0003"] - ds_input["state_q0003"]
            ) / 1200  # Q tendency [kg/kg/s]
            ds_target["ptend_u"] = (
                ds_target["state_u"] - ds_input["state_u"]
            ) / 1200  # U tendency [m/s/s]
            ds_target["ptend_v"] = (
                ds_target["state_v"] - ds_input["state_v"]
            ) / 1200  # V tendency [m/s/s]
        ds_target = ds_target[self.target_vars]

        return ds_input, ds_target

    def __len__(self):
        return len(self.df)

    def get_data(self, idx, key="path"):
        row = self.df.iloc[idx]
        if key == "path":
            ds_input, ds_target = self.get_pair(self.root_folder / row[key])
        else:
            ds_input = self.get_input(self.root_folder / row[key])
            ds_target = None

        # # normalization, scaling
        # if self.normalize:
        #     ds_input = (ds_input - self.input_mean) / (self.input_max - self.input_min)
        #     ds_target = ds_target * self.output_scale
        # else:

        # lat, lon = self.grid_info["lat"].values, self.grid_info["lon"].values
        # lon1, lon2 = np.cos(np.deg2rad(lon)), np.sin(np.deg2rad(lon))
        # lat1, lat2 = np.cos(np.deg2rad(2 * lat)), np.sin(np.deg2rad(2 * lat))
        # area = self.grid_info["area_wgt"].values

        ds_input = ds_input.drop(["lat", "lon"])

        # stack
        # ds = ds.stack({'batch':{'sample','ncol'}})
        ds_input = ds_input.stack({"batch": {"ncol"}})
        ds_input = ds_input.to_stacked_array(
            "mlvar", sample_dims=["batch"], name="mli"
        ).values

        # dso = dso.stack({'batch':{'sample','ncol'}})
        if ds_target is not None:
            ds_target = ds_target.stack({"batch": {"ncol"}})
            ds_target = ds_target.to_stacked_array(
                "mlvar", sample_dims=["batch"], name="mlo"
            ).values

        if self.add_static:
            static_data = get_static(self.grid_info)
            ds_input = np.concatenate([ds_input, static_data], axis=1)

        if self.x_transform:
            ds_input = self.x_transform(ds_input)
        if self.y_transform and ds_target is not None:
            ds_target = self.y_transform(ds_target)

        return ds_input, ds_target

    def get_data_neighbours(self, idx, key="path"):
        x_all, y_all = self.get_data(idx, key=key)
        x_nei = x_all[self.neighbours["idxs"].values, :]
        assert (x_nei[:, 0] == x_all).all()

        # Set the ranges between -1 and 1
        y_dist = self.neighbours["y_distances"].values[..., None] / 1975267.0
        x_dist = self.neighbours["x_distances"].values[..., None] / 1975267.0
        dist_norm = self.neighbours["distances"].values[..., None] / 2035.0

        match key:
            case "path":
                time_emb = np.zeros_like(dist_norm)
            case "prev_path":
                time_emb = -np.ones_like(dist_norm)
            case "next_path":
                time_emb = np.ones_like(dist_norm)

        x_nei = np.concatenate(
            [x_nei, y_dist, x_dist, dist_norm, time_emb], axis=-1
        ).astype(np.float32)
        return x_nei, y_all

    def __getitem__(self, idx, dtype=np.float32):
        x, y = self.get_data(idx)
        return x, y.astype(dtype)

    # def __getitem__(self, idx):

    #     if self.multi_step:
    #         x0, _ = self.get_data_neighbours(idx, key="prev_path")
    #         x1, y = self.get_data_neighbours(idx, key="path")
    #         x2, _ = self.get_data_neighbours(idx, key="next_path")

    #         # x1 first so x1[:,0] is the target
    #         return np.concatenate([x1, x0, x2], axis=1), y
    #     else:
    #         return self.get_data(idx)


def get_idxs(num, num_workers, seed=42):
    idxs = np.arange(num)
    np.random.seed(seed)
    idxs = np.random.permutation(idxs)[0 : num - num % num_workers]
    idxs = np.array_split(idxs, num_workers)
    return idxs


# class IterableDataset(torch.utils.data.IterableDataset):
#     def __init__(self, inner_ds, num_workers=24, seed=42, sample_size=16):
#         self.num_workers = num_workers
#         self.total_iterations = -1
#         self.seed = seed
#         self.inner_ds = inner_ds
#         self.num_samples = len(inner_ds)
#         self.sample_size = sample_size
#         self.grid_points = 384
#         assert self.grid_points % self.sample_size == 0
#         self.inner_rep = self.grid_points // self.sample_size

#     def gen(
#         self,
#     ):
#         self.total_iterations += 1
#         worker_info = torch.utils.data.get_worker_info()

#         if worker_info is None:
#             logging.warning("No worker info")
#             iter_idx = 0
#         else:
#             iter_idx = worker_info.id

#         idxs = get_idxs(
#             len(self.inner_ds), self.num_workers, self.seed + self.total_iterations
#         )

#         for idx in idxs[iter_idx]:
#             # Each inner dataset contains 384 unique grid points
#             ds_x_inner, ds_y_inner = self.inner_ds[idx]
#             # ds_x_inner = ds_x_inner.values
#             # ds_y_inner = ds_y_inner.values

#             random_sample = np.random.permutation(self.grid_points)
#             for n in range(self.inner_rep):
#                 ds_x = ds_x_inner[
#                     random_sample[n * self.sample_size : (n + 1) * self.sample_size]
#                 ]
#                 ds_y = ds_y_inner[
#                     random_sample[n * self.sample_size : (n + 1) * self.sample_size]
#                 ]
#                 yield ds_x, ds_y

#     def __iter__(self):
#         return self.gen()


class InnerDataLoader(torch.utils.data.IterableDataset):
    def __init__(self, inner_ds, num_workers=12, seed=42, batch_size=128):
        self.inner_ds = inner_ds
        self.num_workers = num_workers
        self.seed = seed
        self.gen = np.random.RandomState(seed)
        self.dl = DataLoader(
            inner_ds,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            batch_size=num_workers,
            prefetch_factor=2,
            collate_fn=concat_collate,
            shuffle=True,
        )
        assert (384 * num_workers) % batch_size == 0

        self.batch_size = batch_size
        self.dl_iter = None

    def __iter__(self):
        print("Starting generator")
        self.dl_iter = iter(self.dl)

        return self.generator() 

    def generator(self):
        while True:
            assert self.dl_iter is not None
            x, y = next(self.dl_iter)
            sample = self.gen.permutation(y.shape[0])
            sample = sample.reshape(-1, self.batch_size,)

            for i in range(sample.shape[0]):
                yield [a[sample[i]] for a in x], y[sample[i]]

    def __len__(self):
        return len(self.inner_ds) -  (len(self.inner_ds) % self.batch_size)

    # reset
    def reset(self):
        self.dl_iter = iter(self.dl)


def concat_collate(batch):
    x_all = [[], [], []]
    y_all = []
    for (x1, x2, x3), y in batch:
        x_all[0].append(torch.from_numpy(x1))
        x_all[1].append(torch.from_numpy(x2))
        x_all[2].append(torch.from_numpy(x3))
        y_all.append(torch.from_numpy(y))

    x_all = [torch.cat(x, dim=0) for x in x_all]
    y_all = torch.cat(y_all, dim=0)
    return x_all, y_all


def get_datasets(loader_cfg: config.LoaderConfig, data_cfg: config.DataConfig):
    x_norm, y_norm = norm.get_stats(loader_cfg, data_cfg)

    df_index = pd.read_parquet(loader_cfg.index_path)

    df_index_tr = df_index[df_index["year"] <= 8]
    df_index_val = df_index[df_index["year"] == 9]

    assert len(df_index) == len(df_index_tr) + len(df_index_val)

    inner_train_ds = LeapLoader(
        root_folder=Path(loader_cfg.root_folder),
        grid_info_path=loader_cfg.grid_info_path,
        df=df_index_tr,
        x_transform=x_norm if loader_cfg.apply_norm else None,
        y_transform=y_norm if loader_cfg.apply_norm else None,
    )

    valid_ds = LeapLoader(
        root_folder=Path(loader_cfg.root_folder),
        grid_info_path=loader_cfg.grid_info_path,
        df=df_index_val,
        x_transform=x_norm if loader_cfg.apply_norm else None,
        y_transform=y_norm if loader_cfg.apply_norm else None,
    )

    return inner_train_ds, valid_ds


def single_batch_collate(batch):
    return batch[0]


def setup_dataloaders(
    loader_cfg: config.LoaderConfig,
    data_cfg: config.DataConfig,
):
    inner_train_ds, valid_ds = get_datasets(loader_cfg, data_cfg)

    if loader_cfg.use_iterable_train:
        train_ds = InnerDataLoader(
            inner_train_ds, num_workers=12, batch_size=loader_cfg.batch_size
        )

        dl_kwargs = dict(
            num_workers=0,
            batch_size=1,
            collate_fn=single_batch_collate,
            pin_memory=False,
        )
    else:
        train_ds = inner_train_ds
        dl_kwargs = dict(
            num_workers=8, batch_size=1, pin_memory=True, shuffle=True
        )  # effective batch size -> 384

    train_dl = torch.utils.data.DataLoader(train_ds, **dl_kwargs)

    x, y = next(iter(train_dl))
    print(f"y.shape: {y.shape} y_std: {y.std()}")
    for n in range(len(x)):
        print(
            f"x[{n}].shape: {x[n].shape} x_std: {x[n].std()} x_max: {x[n].max()} x_min: {x[n].min()}"
        )

    if isinstance(train_ds, InnerDataLoader):
        train_ds.reset()

    # effective batch size -> 384
    valid_loader = torch.utils.data.DataLoader(
        valid_ds,
        num_workers=12,
        batch_size=1,
        collate_fn=concat_collate,
        pin_memory=True,
    )

    return train_dl, valid_loader
