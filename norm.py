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
        self, stds, means, tanh_mults=None, zero_mask=None, ndim=1, dict_key=None
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

        self.dict_key = dict_key

    def __call__(self, data, x=None):
        out = (data - self.means) / self.stds
        if self.zero_mask is not None:
            out[:, self.zero_mask] = 0

        if self.use_tanh:
            out = np.tanh(out * self.tanh_mults)

        out = out.astype(np.float32)
        if self.dict_key is not None:
            return {self.dict_key: out}
        else:
            return out

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

        return {
            "x_p": x_p.astype(np.float32),
            "x_1d": x_1d.astype(np.float32),
            "x_1d_re": x_1d_re.astype(np.float32),
        }

    def denorm(self, data):
        return self.norm_x.denorm(data)


def get_classification_mask(y_zero_mask):
    y_zero_mask = ~y_zero_mask.copy().squeeze()
    y_zero_mask[0 : 60 * 2] = False
    y_zero_mask[60 * 4 :] = False
    return y_zero_mask


def get_classification_ratio_labels(x_raw, y_raw, mask, y_std, thresh=0.01):
    x_data = x_raw[:, 0 : y_raw.shape[1]][:, mask].copy()
    y_data = y_raw[:, mask].copy()
    y_std = y_std[mask]

    ratio = (1200 * y_data) / (x_data + 1e-99)
    x_data_z = x_data <= 1e-99
    ratio[x_data_z] = 0

    assert ratio.min() >= -1 - 1e-6

    out = np.zeros_like(ratio, dtype=np.int64)
    out_ratios = np.zeros_like(ratio, dtype=np.float32)

    thresh_r = 1 - thresh

    assert (x_data >= 0).all()
    non_zeros = np.abs(y_data) >= y_std * thresh

    mask_one = ratio < -thresh_r
    mask_two = (ratio >= -thresh_r) & (ratio <= thresh_r)

    mask_there = ratio > thresh_r

    assert not (mask_one & mask_two).any()
    assert not (mask_one & mask_there).any()
    assert not (mask_two & mask_there).any()
    mask_one[~non_zeros] = False
    mask_two[~non_zeros] = False
    mask_there[~non_zeros] = False

    out[mask_one] = 1
    out[mask_two] = 2
    out[mask_there] = 3

    out_ratios[mask_two] = ratio[mask_two]
    assert out_ratios.min() >= -thresh_r
    assert out_ratios.max() <= thresh_r

    return out, out_ratios


class ClassWrapper:
    def __init__(self, y_norm: Norm2, class_mask, thresh=0.01):
        self.y_norm = y_norm
        self.class_mask = class_mask
        self.thresh = thresh

    def __call__(self, y, x=None):
        assert x is not None

        y_norm = self.y_norm(y)["y"]
        y_cls, y_ratios = get_classification_ratio_labels(
            x, y, self.class_mask, self.y_norm.stds.squeeze(), self.thresh
        )

        return {
            "y": y_norm.astype(np.float32),
            "y_cls": y_cls,
            "y_ratios": y_ratios.astype(np.float32),
            "y_raw": y,
            "x_raw": x,
        }

    def denorm(self, data):
        return self.y_norm.denorm(data)


def get_stats(loader_cfg: config.LoaderConfig, data_cfg: config.DataConfig):
    # Y stds are the weights
    # weights = pd.read_csv(loader_cfg.weights_path, nrows=1)
    # weights = weights.iloc[0, 1:].values.astype(np.float32)

    # Use the weightings as y_std
    # std_weights = 1.0 / (weights)
    # std_weights[weights == 0] = 0
    # assert np.isfinite(std_weights).all()

    stats_x = load_from_json(loader_cfg.x_stats_path)
    if loader_cfg.x_tanh:
        print("Using tanh")
        tanh_mults = 2.5 / (stats_x["x_range"] + 0.1)
    else:
        print("Disabling tanh")
        tanh_mults = None

    if loader_cfg.x_mask_thresh:
        x_mask = stats_x["x_range"] > loader_cfg.x_mask_thresh
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

    norm_x = Norm2(
        stds=stats_x["stds"],
        means=stats_x["means"],
        tanh_mults=tanh_mults,
        zero_mask=x_mask,
    )
    # norm_x = Norm(fname=loader_cfg.x_stats_path, eps=1e-7)
    # Set means to zero for q vars so we can predict a multiplier
    # norm_x.means[:, data_cfg.fac_idxs[0] : data_cfg.fac_idxs[1]] = 0.0

    norm_1dx = Norm2(stds=stats_x["x1d_std"], means=stats_x["x1d_mean"], ndim=2)
    norm_x_cmb = NormSplitCmb(norm_x, norm_1dx, data_cfg)

    stats_y = load_from_json(loader_cfg.y_stats_path)

    std_weights = stats_y["stds"]
    y_means = stats_y["means"]
    y_zero_mask = stats_y["y_zero"]

    norm_y = Norm2(stds=std_weights, means=y_means, zero_mask=y_zero_mask, dict_key="y")
    if loader_cfg.y_class:
        y_class_mask = get_classification_mask(y_zero_mask)
        norm_y = ClassWrapper(norm_y, y_class_mask)

        print(f"Classification mask: {y_class_mask.sum()} of {len(y_class_mask)}")

    print(f"Zero mask: {y_zero_mask.sum()} of {len(y_zero_mask)}")

    return norm_x_cmb, norm_y
