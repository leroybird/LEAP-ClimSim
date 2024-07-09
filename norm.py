import json
import numpy as np
import config
import pandas as pd
from einops import rearrange


def split_data(x, split_idx, num_2d_in):
    # Data contains 1d vars, point vars, then 1d, then static vars...
    x_1d, x_point = x[:, :split_idx], x[:, split_idx:]

    x_1d_2, x_point = x_point[:, num_2d_in:], x_point[:, :num_2d_in]

    x_1d = np.concatenate([x_1d, x_1d_2], axis=1)
    x_1d_re = rearrange(x_1d.copy(), "n (f v) -> n v f", v=60)

    return x_point, x_1d, x_1d_re


def load_from_json(fname):
    with open(fname) as f:
        data = json.loads(f.read())
    for key, val in data.items():
        data[key] = np.asarray(val)
    return data


class Norm2:
    def __init__(
        self,
        stds,
        means,
        tanh_mults=None,
        zero_mask=None,
        ndim=1,
    ):
        def add_dims(arr, ndim):
            for _ in range(ndim):
                arr = arr[None, :]
            return arr

        self.stds = stds.copy()
        self.means = means.copy()

        self.means = add_dims(self.means, ndim)
        self.stds = add_dims(self.stds, ndim)

        self.zero_mask = zero_mask
        # self.stds[0] <= eps if zero_mask is None else zero_mask
        if self.zero_mask is not None:
            self.stds[..., self.zero_mask] = 1.0

        self.use_tanh = tanh_mults is not None
        if self.use_tanh:
            self.tanh_mults = add_dims(tanh_mults, ndim)

    def __call__(self, data):
        out = (data - self.means) / self.stds
        if self.zero_mask is not None:
            out[:, self.zero_mask] = 0

        if self.use_tanh:
            out = np.tanh(out * self.tanh_mults)

        return out.astype(np.float32)

    def denorm(self, data):
        data = data.astype(np.float64)

        if self.use_tanh:
            data = np.arctanh(data)
            data = data / self.tanh_mults

        out = data * self.stds + self.means

        assert not self.use_tanh

        if self.zero_mask is not None:
            out[:, self.zero_mask] = 0  # self.means[:, self.zero_mask]

        return out


class Norm:
    def __init__(
        self, fname=None, stds=None, means=None, zero_mask=None, dataset=None, eps=1e-14
    ):
        if dataset is not None:
            self.means, self.stds = np.mean(dataset, axis=0), np.std(dataset, axis=0)
            with open(fname, "w") as f:
                f.write(
                    json.dumps({"means": self.means.tolist(), "stds": self.stds.tolist()})
                )
        elif means is not None and stds is not None:
            self.stds = stds.copy()
            self.means = means.copy()
        else:
            with open(fname) as f:
                stats_dict = json.loads(f.read())

            self.means = np.asarray(stats_dict["means"])
            self.stds = np.asarray(stats_dict["stds"])

        self.means = self.means[None, :]
        self.stds = self.stds[None, :]

        self.zero_mask = self.stds[0] <= eps if zero_mask is None else zero_mask

        self.stds[:, self.zero_mask] = 1.0

        self.eps = eps
        # self.df = pd.DataFrame({'col' : names, 'std' : self.stds, 'mean' : self.means})

    def __call__(self, data):
        out = (data - self.means) / self.stds
        out[:, self.zero_mask] = 0

        return out.astype(np.float32)

    def denorm(self, data):
        data = data.astype(np.float64)
        out = data * self.stds + self.means

        out[:, self.zero_mask] = 0  # self.means[:, self.zero_mask]
        return out


class NormSplitCmb:
    def __init__(self, norm_x, norm_1dx, data_cfg: config.DataConfig):
        self.norm_x = norm_x
        self.norm_1dx = norm_1dx
        self.data_cfg = data_cfg

    def __call__(self, data):
        _, _, x_1d_re = split_data(
            data, self.data_cfg.split_index, self.data_cfg.num_2d_feat
        )
        x_n = self.norm_x(data)
        x_p, x_1d, _ = split_data(
            x_n, self.data_cfg.split_index, self.data_cfg.num_2d_feat
        )

        x_1d_re = self.norm_1dx(x_1d_re)

        return x_p, x_1d, x_1d_re


def get_stats(loader_cfg: config.LoaderConfig, data_cfg: config.DataConfig):
    # Y stds are the weights
    #weights = pd.read_csv(loader_cfg.weights_path, nrows=1)
    #weights = weights.iloc[0, 1:].values.astype(np.float32)

    # Use the weightings as y_std
    #std_weights = 1.0 / (weights)
    #std_weights[weights == 0] = 0
    #assert np.isfinite(std_weights).all()

    stats_x = load_from_json(loader_cfg.x_stats_path)
    if loader_cfg.x_tanh:
        print("Using tanh")
        tanh_mults = 2.5 / (stats_x["x_range"] + 0.1)
    else:
        print("Disabling tanh")
        tanh_mults = None
    

    if loader_cfg.x_mask_thresh:
        x_mask = stats_x['x_range'] > loader_cfg.x_mask_thresh
        assert data_cfg.x_names is not None
        for n, name in enumerate(data_cfg.x_names):
            if name.startswith("state_q0001") and int(name.split("_")[-1]) <= 10:
                x_mask[n] = True
            if name.startswith("state_q0002") and int(name.split("_")[-1]) <= 15:
                x_mask[n] = True
            if name.startswith("state_q0003") and int(name.split("_")[-1]) <= 15:
                x_mask[n] = True

        print(f"Masking {x_mask.sum()} features")
        print(f"Masked features: {np.array(data_cfg.x_names)[x_mask]}")


    else:
        x_mask = None

    norm_x = Norm2(stds=stats_x["stds"], means=stats_x["means"], tanh_mults=tanh_mults,
                   zero_mask=x_mask)
    # norm_x = Norm(fname=loader_cfg.x_stats_path, eps=1e-7)
    # Set means to zero for q vars so we can predict a multiplier
    # norm_x.means[:, data_cfg.fac_idxs[0] : data_cfg.fac_idxs[1]] = 0.0

    norm_1dx = Norm2(stds=stats_x["x1d_std"], means=stats_x["x1d_mean"], ndim=2)    
    norm_x_cmb = NormSplitCmb(norm_x, norm_1dx, data_cfg)

    stats_y = load_from_json(loader_cfg.y_stats_path)

    std_weights = stats_y["stds"]
    y_means = stats_y["means"]
    y_zero_mask = stats_y['y_zero']
    
    norm_y = Norm2(stds=std_weights, means=y_means, zero_mask=y_zero_mask)
    
    print(f"Zero mask: {y_zero_mask.sum()} of {len(y_zero_mask)}")


    # This variable still seems to have norm issues.
    #indxs = [data_cfg.y_names.index(a) for a in ["ptend_q0002_26", "ptend_q0002_25"]]
    #norm_y.zero_mask[indxs] = True

    return norm_x_cmb, norm_y
