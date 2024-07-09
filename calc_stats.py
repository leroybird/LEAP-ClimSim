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
train_df_cols = pl.read_csv(cfg.train_kaggle_csv, n_rows=1)
#%%

#%%
#train_df.write_parquet("/mnt/ssd/kaggle/train3m.parquet")
train_df = pl.read_parquet("/mnt/ssd/kaggle/train3m.parquet")
#%%
train_df.columns = train_df_cols.columns
#%%
y_weightings = pd.read_csv(cfg.weights_path, nrows=1).iloc[0, 1:].values.astype(np.float32)
# %%
test_df = pl.read_csv(test_kaggle_path)
#%%
train_df
# %%
mask = pd.read_csv(sample_submission_path, nrows=1).iloc[0, 1:].values.astype(bool)
mask
# %%
y_zero_mask = pd.read_csv(sample_submission_path, nrows=1).iloc[0, 1:].values
y_zero_mask = y_zero_mask == 0
#%%
plt.plot(y_zero_mask)
#%%
train_df
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
test_df
#%%
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
y_log_std = np.log(y_train.std(axis=0))
#%%
y_log_std[mask == 0] = 0
plt.plot(y_log_std)
plt.show()
#%%
mask.shape
# %%
y_log_std.shape
#%%
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
#%%

y_std = y_train.std(axis=0)
y_std[mask == 0] = 1.0

y_mean = y_train.mean(axis=0)
y_min = np.min(y_train, axis=0)
y_max = np.max(y_train, axis=0)
y_range = (y_max - y_min) / y_std
#%%
# %%
start = 120
end = start + 60
max_range = 500
#%%
plt.plot(y_std[start:end])
plt.axvline(27.0, color='black')
plt.yscale('log')
#%%
y_zero_mask[120:120+27] = True
#%%
y_std[~y_zero_mask].min()
#%%
# Remove mean for q vars
y_mean[60:60*4] = 0.0
#%%
#%%
y_out = {
    'stds': y_std,
    'means': y_mean,
    'y_range': y_range,
    'y_max': y_max,
    'y_min': y_min,
    'y_zero' : y_zero_mask
}
#%%
save_to_json('y_stats_v2_1.json', y_out)
#%%
y_zero_mask
#%%

range_sub = y_range.copy()
range_sub[range_sub > max_range] = max_range
#%%
y_std *= 
#%% 
plt.plot(y_range[start:end])

#%%
a = (1200 * y_train[:, 0:6*60])/(x_train[:, 0:6*60])
a[x_train[:, 0:60] == 0] = 0
#%%
a[x_train[:, 0:6*60] == 0] = 0
#%%
plt.plot(mask)
#%%
from collections import defaultdict
output = defaultdict(list)
output_pos_ranges = defaultdict(list)

!mkdir -p plots
for n, (col, m) in enumerate(zip(TARGET_COLS, mask)):
    if n > 6*60:
        break
    
    if m == 1 and 'q' in col:
        zero_mask = np.abs(x_train[:, n]) >= y_std[n]*0.01
        # if zero_mask.sum() == 0:
        #     print(f'{col} has no zeros')
        y_data = y_train[zero_mask, n].copy()
        
        ratio = a[zero_mask, n].copy()*1200                
        max_value = np.abs(np.min(ratio))
        
        y_neg_m = ratio < -0.99*max_value
        y_middle_m = (ratio > (-0.99*max_value)) & (ratio <= 2*max_value)
        y_pos_m = ratio > max_value*2
        assert y_neg_m.sum() + y_middle_m.sum() + y_pos_m.sum() == y_data.shape[0]
        
        y_neg_v = 100*np.abs(y_data[y_neg_m]).sum() / np.abs(y_data).sum()
        y_middle_v = 100*np.abs(y_data[y_middle_m]).sum() / np.abs(y_data).sum()
        y_pos_v = 100*np.abs(y_data[y_pos_m]).sum() / np.abs(y_data).sum()
        
        y_pos_max = y_data[y_pos_m].max() if y_pos_m.sum() > 0 else 0
        y_pos_min = y_data[y_pos_m].min() if y_pos_m.sum() > 0 else 0
        output_pos_ranges['pos_range'].append((y_pos_max - y_pos_min)/y_std[n])
        output_pos_ranges['index'].append(col)
        
        y_zero_t = 100*np.abs(y_train[~zero_mask, n]).sum() / np.abs(y_data).sum()
        output['index'].append(col)
        output['non_zero'].append(100*zero_mask.sum()/zero_mask.shape[0])
        
        output['neg'].append(y_neg_v)
        output['middle'].append(y_middle_v)
        output['pos'].append(y_pos_v)
        output['zero'].append(y_zero_t)
        
        
        zero_prec = 100*zero_mask.sum()/zero_mask.shape[0]
        print(f'{col}: non_zero:{zero_prec:.5f} {y_neg_v:.5f}, {y_middle_v:.5f}, {y_pos_v:.5f} {y_zero_t:.5f}')        
        
        sub_mask = ratio > (max_value * 2)
        ratio[sub_mask] = max_value*2

        plt.figure(figsize=(12, 12))
        plt.title(f'{zero_prec:.5f} {col} {y_neg_v:.3f}, {y_middle_v:.3f}, {y_pos_v:.3f} {y_zero_t:.3f}')
        plt.hist(ratio, bins=100)
        plt.savefig(f'plots/{col}.png')
        plt.close()
#%%
output = pd.DataFrame(output)
output.index = output['index']
#%%
output_pos_ranges = pd.DataFrame(output_pos_ranges)
output_pos_ranges.index = output_pos_ranges['index']
#%%
output_pos_ranges
#%%
subset = output_pos_ranges[output_pos_ranges['index'].str.contains('q0003')]
subset.plot(figsize=(12,12))
plt.axvline(18-12, color='black')
 #%%
output.to_csv('q_metadata.csv', index=True)
#%%
#plt.figure()
target = 'q0003'
subset = output[output['index'].str.contains(target)]
subset.plot(figsize=(16,16))

#%%
!mkdir -p plots2
for n, (col, m) in enumerate(zip(TARGET_COLS, mask)):
    if m == 1:
        plt.figure(figsize=(12, 12))
        plt.hist(x_train[:, n] - x_mean[n], bins=100)
        plt.hist(y_train[:, n]*1200, bins=100, alpha=0.5)
        
        plt.savefig(f'plots2/{col}.png')
        plt.close()

#%%
