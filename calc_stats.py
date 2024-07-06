# %%
%load_ext autoreload
%autoreload 2
#%%
from einops import rearrange
import numpy as np
import pandas as pd
import polars as pl
from config import LoaderConfig
import matplotlib.pyplot as plt
from norm import split_data, Norm2, load_from_json

# %%
cfg = LoaderConfig()
cfg
# %%
test_kaggle_path = "/mnt/ssd/kaggle/new_data/test.csv"
sample_submission_path = "/mnt/ssd/kaggle/new_data/sample_submission.csv"
# %%
pl.Config(tbl_cols=-1)
train_df = pl.read_csv(cfg.train_kaggle_csv, n_rows=4_000_000)
#%%
y_weightings = pd.read_csv(cfg.weights_path, nrows=1).iloc[0, 1:].values.astype(np.float32)
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
VERT_SPLIT = NUM_VERT * NUM_VERT_FEAT_FIRST

# %%
FEAT_COLS[0:VERT_SPLIT],
# %%
FEAT_COLS[VERT_SPLIT : VERT_SPLIT + NUM_2D_FEAT + 2]
# %%
x_train = train_df.select(FEAT_COLS).to_numpy()
y_train = train_df.select(TARGET_COLS).to_numpy()
# %%
x_test = test_df.select(FEAT_COLS).to_numpy()
#%%
del train_df
import gc
gc.collect()

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

# %%
norm = Norm(x_std, x_mean, tanh_mults=2.5 / (x_range + 0.1))
# %%
x_train

# %%
x_sub_norm = norm(x_sub)
# %%
x_sub_norm[:, 0:400].max()
# %%
plt.plot(x_sub_norm.std(axis=0))
plt.plot(x_sub_norm.min(axis=0))
plt.plot(x_sub_norm.max(axis=0))
plt.ylim(-1.0, 1.0)


# %%
# %%
# import gc
# del train_df
# %%
def split_data(x, split_idx=VERT_SPLIT, num_2d_in=NUM_2D_FEAT):
    # Data contains 1d vars, point vars, then 1d, then static vars...
    x_1d, x_point = x[:, :split_idx], x[:, split_idx:]

    x_1d_2, x_point = x_point[:, num_2d_in:], x_point[:, :num_2d_in]

    x_1d = np.concatenate([x_1d, x_1d_2], axis=1)
    x_1d = rearrange(x_1d, "n (f v) -> n v f", v=60)


    return x_point, x_1d


# %%
x_point, x_1d = split_data(x_train, split_idx=VERT_SPLIT, num_2d_in=NUM_2D_FEAT)
# %%
x_1d.shape
# %%
x_1d_std = x_1d.std(axis=(0, 1))
# %%
x_1d_mean = x_1d.mean(axis=(0, 1))
x_1d_min = np.min(x_1d, axis=(0, 1))
x_1d_max = np.max(x_1d, axis=(0, 1))
x_1d_range = (x_1d_max - x_1d_min) / x_1d_std
# %%
plt.plot(x_1d_range)
# %%
x_out = {
    "stds": x_std,
    "means": x_mean,
    "x_range": x_range,
    "x_max": x_max,
    "x_min": x_min,
    "x1d_std": x_1d_std,
    "x1d_mean": x_1d_mean,
    "x1d_range": x_1d_range,
    "x1d_max": x_1d_max,
    "x1d_min": x_1d_min,
}
# %%
save_to_json("x_stats_v2_1.json", x_out)
# %%
y_weightings
#%%
y_std = y_train.std(axis=0)
y_mean = y_train.mean(axis=0)
y_min = np.min(y_train, axis=0)
y_max = np.max(y_train, axis=0)
y_range = (y_max - y_min)/y_std #/ (y_std + 1e-14)
#%%
y_t = y_range.copy()
y_t[~mask] = np.nan
#%%
yt2 = (y_max - y_min)*y_weightings
yt2[~mask] = np.nan
#%%
plt.figure(figsize=(20, 20))
plt.plot(y_t)
plt.plot(yt2)
#plt.xlim(140, 150)
#%%
plt.plot()

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


#%%