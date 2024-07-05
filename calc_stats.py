# %%
import numpy as np
import pandas as pd
import polars as pl
from config import LoaderConfig
import matplotlib.pyplot as plt

# %%
cfg = LoaderConfig()
cfg
# %%
test_kaggle_path = "/mnt/ssd/kaggle/new_data/test.csv"
sample_submission_path = "/mnt/ssd/kaggle/new_data/sample_submission.csv"
# %%
pl.Config(tbl_cols=-1)
train_df = pl.read_csv(cfg.train_kaggle_csv, n_rows=4_000_000)
# %%
test_df = pl.read_csv(test_kaggle_path)
# %%
mask = pd.read_csv(sample_submission_path, nrows=1).iloc[0, 1:].values.astype(bool)
mask
# %%
test_df
# %%
NUM_VERT = 60
NUM_VERT_FEAT = 9
NUM_VERT_FEAT_Y = 6

FEAT_COLS = train_df.columns[1:557]
TARGET_COLS = train_df.columns[557:]


NUM_2D_FEAT = len(FEAT_COLS) - NUM_VERT * NUM_VERT_FEAT
NUM_2D_FEAT_Y = len(TARGET_COLS) - NUM_VERT * NUM_VERT_FEAT_Y

# Predict a multiplier of q for q_tends
FRAC_IDXS = (NUM_VERT, NUM_VERT * 4)
# %%
NUM_VERT_FEAT_FIRST = 6
NUM_2D_FEAT = 16


def split_data(x, split_idx, num_2d_in):
    # x = rearrange(x, "b t c -> b c t").contiguous()
    # Data contains 1d vars, point vars, then 1d, then static vars...
    x_1d, x_point = x[:, :split_idx], x[:, split_idx:]

    x_1d_2, x_point = x_point[:, num_2d_in:], x_point[:, :num_2d_in]

    x_1d = np.concatenate([x_1d, x_1d_2], axis=1)

    return x_point, x_1d


# %%
x_train = train_df.select(FEAT_COLS).to_numpy()
y_train = train_df.select(TARGET_COLS).to_numpy()
# %%
x_test = test_df.select(FEAT_COLS).to_numpy()
# %%
x_log_std = np.log(x_train.std(axis=0))
x_test_log_std = np.log(x_test.std(axis=0))
plt.plot(x_log_std, label="train")
plt.plot(x_test_log_std, label="test")
plt.legend()
plt.show()
# %%
# %%
x_std = x_train.std(axis=0)
# %%
x_mean = x_train.mean(axis=0)
# %%
x_min = np.min(x_train, axis=0)
x_max = np.max(x_train, axis=0)
x_range = (x_max - x_min) / x_std
# %%
plt.plot(x_range)

# %%
plt.plot(np.log(x_std))


# %%
def norm_tanh(x, std, mean, mult):
    t_norm = (x - mean) / std
    return np.tanh(t_norm * mult).astype(np.float32)


# %%
x_sub = x_train[np.random.choice(x_train.shape[0], 500_000), :]
# %%
import json


def save_to_json(fname, data):
    data_out = {}
    for key, val in data.items():
        data_out[key] = val.tolist()
    with open(fname, "w") as f:
        f.write(json.dumps(data_out, indent=4))


def load_from_json(fname):
    with open(fname) as f:
        data = json.loads(f.read())
    for key, val in data.items():
        data[key] = np.asarray(val)
    return data


# %%
class Norm:
    def __init__(
        self,
        stds,
        means,
        tanh_mults=None,
        zero_mask=None,
    ):
        self.stds = stds.copy()
        self.means = means.copy()

        self.means = self.means[None, :]
        self.stds = self.stds[None, :]

        self.zero_mask = zero_mask
          # self.stds[0] <= eps if zero_mask is None else zero_mask
        if self.zero_mask is not None:
            self.stds[:, self.zero_mask] = 1.0

        self.use_tanh = tanh_mults is not None
        if self.use_tanh:
            self.tanh_mults = tanh_mults[None, :]

    def __call__(self, data):
        out = (data - self.means) / self.stds
        if self.zero_mask is not None:
            out[:, self.zero_mask] = 0

        if self.use_tanh:
            out = np.tanh(out * self.tanh_mults)

        return out.astype(np.float32)

    def denorm(self, data):
        data = data.astype(np.float64)
        out = data * self.stds + self.means
        
        assert not self.use_tanh
        
        if self.zero_mask is not None:
            out[:, self.zero_mask] = 0  # self.means[:, self.zero_mask]
        
        return out
#%%

norm = Norm(x_std, x_mean, tanh_mults=2.5 / (x_range + 0.1))
#%%
x_train

#%%
x_sub_norm = norm(x_sub)
# %%
x_sub_norm[:, 0:400].max()
# %%
plt.plot(x_sub_norm.std(axis=0))
plt.plot(x_sub_norm.min(axis=0))
plt.plot(x_sub_norm.max(axis=0))
plt.ylim(-1.0, 1.0)
# %%
x_out = {
    'stds': x_std,
    'means': x_mean,
    'x_range' : x_range,
    'x_max' : x_max,
    'x_min' : x_min,
}

# %%
save_to_json("x_stats_v2_1.json", x_out)
#%%


# %%
norm_y = Norm(
    stds=np.ones(y_train.shape[1]),
    means=np.zeros_like(std_weights),
    zero_mask=std_weights < 1e-13,
)  # dataset=y_train)

indxs = [TARGET_COLS.index(a) for a in ["ptend_q0002_26", "ptend_q0002_25"]]
col = "ptend_q0002_26"
norm_y.zero_mask[indxs] = True
# %%
