#%%
%load_ext autoreload
%autoreload 2
#%%
from itertools import chain
import pandas as pd
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from arch import NetTr
import torch
#%%
weightings = pd.read_csv('/mnt/ssd/kaggle/sample_submission.csv', nrows=1)
#%%

train_df = pl.read_parquet('/mnt/ssd/kaggle/train2.parquet', n_rows=10_000)
#%%
test_df = pl.read_parquet('/mnt/ssd/kaggle/test.parquet', n_rows=100_000)

#%%
weighting = weightings.iloc[0, 1:].values.astype(np.float32)
weighting
#%%
pl.Config(tbl_cols=-1, tbl_rows=20)

FEAT_COLS = train_df.columns[1:557]
TARGET_COLS= train_df.columns[557:]

NUM_VERT = 60
NUM_VERT_FEAT = 9
NUM_VERT_FEAT_Y = 6

FEAT_COLS = train_df.columns[61:60*4+1]
TARGET_COLS= train_df.columns[557+60:557+60*4]


NUM_2D_FEAT = len(FEAT_COLS) - NUM_VERT*NUM_VERT_FEAT
NUM_2D_FEAT_Y = len(TARGET_COLS) - NUM_VERT*NUM_VERT_FEAT_Y

NUM_2D_FEAT, NUM_2D_FEAT_Y
#%% 

#%%
60*6
#%%
833/1e-6
#%%
y = train_df.select(TARGET_COLS).to_numpy()
x = train_df.select(FEAT_COLS).to_numpy()
#%%
# net = NetTr(NUM_2D_FEAT,NUM_VERT_FEAT , NUM_2D_FEAT_Y, NUM_VERT_FEAT_Y)
# #%%

# out = net((torch.from_numpy(x[0:64, :]), None))
# #%%
# out.shape
# #%%
# net.check_emb_idxs(FEAT_COLS, TARGET_COLS).dtypes
# #%%
# # Display all rows
# pd.set_option('display.max_rows', 1000)
# net.check_emb_idxs(FEAT_COLS, TARGET_COLS).head(1000)
# #%%
# len(net.var_idxs)
# #%%
# len(FEAT_COLS + TARGET_COLS)
#%%
a = y/(x + 1e-15)
min_fact = np.abs(a.min())
min_fact
#%%
1/min_fact

#%%
a[a > min_fact*2] = 2*min_fact
#%%
subset.shape
#%%
a[a > min_fact*2] = min_fact
#%%
x[1]
#%%
a[1]*1200
#%%
#a[a == 0] = np.nan

for n, col in enumerate(TARGET_COLS):
    if n % 60 > 10:
        plt.figure(figsize=(12, 12))
        plt.hist(a[:, n]*1200, bins=100)
        plt.savefig(f'plots/{col}.png')
        plt.close()
#%%
import torch
#%%
num_vert = 60
num_3d_start = 6
num_in_2d = NUM_2D_FEAT
num_2d_out = NUM_2D_FEAT_Y
total_3d = num_vert*num_3d_start
num_3d_in = NUM_VERT_FEAT
num_3d_out = NUM_VERT_FEAT_Y
#%%
total_3d = num_3d_in + num_3d_out

var_idxs = torch.Tensor([[n]*num_vert for n in range(total_3d)]).flatten().long()
var_idxs_2d = torch.arange(total_3d, total_3d+ num_in_2d + num_2d_out, step=1).long()


#%%
len(var_idxs)
#%%
# 3d_in, 2d_in, 3d_in_2, 2d_out, 3d_out
var_idxs = torch.cat([var_idxs[0:num_3d_start*num_vert], var_idxs_2d[:num_in_2d], 
                        var_idxs[num_3d_start*num_vert:],
                        var_idxs_2d[num_in_2d:]])
#%%
len(var_idxs)
#%%
((df_y/(df_x + 1e-15))*1e4).max()

#%%
# Show summary
desc = weightings.describe()
#%%
df_x = train_df.select(FEAT_COLS)
df_y = train_df.select(TARGET_COLS)
#%%
vars_1d = train_df.columns[60*6+1:60*6+17]
vars_1d
#%%
(df_y/(df_x + 1e-15)).fill_nan(0).max()
#%%
train_df.select(list(chain.from_iterable(zip(FEAT_COLS, TARGET_COLS))))
#%%
list(chain(zip(FEAT_COLS, TARGET_COLS)))
#%%
train_df
#%%
new_col = test_df['sample_id'].map_elements(lambda x: int(x.split('_')[1]), int).rename('id2')
#%%
test_df = test_df.with_columns(new_col)
#%%
test_df = test_df.sort('id2')
#%%
#%%
train_df.select(train_df.columns[1:557])
# %%

plt.plot(train_df[['cam_in_LANDFRAC', 'cam_in_OCNFRAC']].to_numpy(),)
plt.xlim(0, 384*2)
#%%
len(test_df) - (len(test_df)%384)
#%%
data = test_df['cam_in_LANDFRAC']#[0:len(test_df) - (len(test_df)%384)].to_numpy()
# %%
data[::1]
#%%
data[1:2]
#%%

data[:, 0:1] == data
#%%
#%%
test_df.columns
#%%
def plot_pattern(column, df, n, k=384):
    plt.figure(figsize=(12, 12))
    
    data_sub = df[column][0:k*n].to_numpy().reshape(n,k)
    if n < 20:
        plt.plot(data_sub.T)
    else:
        plt.imshow(data_sub.T)
#%%
def plot_pattern_diff(column, df, n, k=384):
    plt.figure(figsize=(12, 12))
    
    data_sub = df[column][0:k*n].to_numpy().reshape(n,k)
    data_s = np.diff(data_sub, axis=0)
    if n < 20:
        plt.plot(data_s.T)
    else:
        plt.imshow(data_s.T)

#%%
def plot_pattern_single(column, df, n, k=384):
    plt.figure(figsize=(12, 12))
    
    data_sub = df[column][0:k*n].to_numpy().reshape(n,k)

    data_s = data_sub[:, 100]
    plt.plot(data_s)

#%%
plot_pattern('pbuf_SOLIN', test_df, 2, k=384)
plt.show()
#%%
plot_pattern_single('pbuf_COSZRS', test_df, 200, k=384)
plt.xlim(0.0, 200)
plt.show()
#%%
test_df
# %%
test_df
# %%
