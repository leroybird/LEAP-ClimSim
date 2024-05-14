import json
import numpy as np
import config
import pandas as pd


class Norm:
    def __init__(self, fname=None, stds=None, means=None, zero_mask=None, dataset=None, eps=1e-14):
        if dataset is not None:
            self.means, self.stds = np.mean(dataset, axis=0), np.std(dataset, axis=0)
            with open(fname, "w") as f:
                f.write(json.dumps({"means": self.means.tolist(), "stds": self.stds.tolist()}))
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


def get_stats(loader_cfg: config.LoaderConfig, data_cfg: config.DataConfig):
    weights = pd.read_csv(loader_cfg.weights_path, nrows=1)
    weights = weights.iloc[0, 1:].values.astype(np.float32)

    # Use the weightings as y_std
    std_weights = 1.0 / (weights)
    std_weights[weights == 0] = 0
    assert np.isfinite(std_weights).all()

    norm_x = Norm(fname=loader_cfg.x_stats_path, eps=1e-7)

    # Set means to zero for q vars so we can predict a multiplier
    norm_x.means[:, data_cfg.fac_idxs[0] : data_cfg.fac_idxs[1]] = 0.0

    norm_y = Norm(stds=std_weights, means=np.zeros_like(std_weights), zero_mask=std_weights < 1e-13)

    # This variable still seems to have norm issues.
    indxs = [data_cfg.y_names.index(a) for a in ["ptend_q0002_26", "ptend_q0002_25"]]
    norm_y.zero_mask[indxs] = True

    return norm_x, norm_y
