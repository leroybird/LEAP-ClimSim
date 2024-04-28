#%%
from itertools import chain
import pandas as pd
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
#%%
weightings = pd.read_csv('/mnt/ssd/kaggle/sample_submission.csv', nrows=1)
#%%

train_df = pl.read_parquet('/mnt/ssd/kaggle/train2.parquet', n_rows=10_000)
#%%
test_df = pl.read_parquet('/mnt/ssd/kaggle/test.parquet')

#%%
train_df
#%%
weightings
#%%
pl.Config(tbl_cols=-1, tbl_rows=20)

FEAT_COLS = train_df.columns[1:557-60*3]
TARGET_COLS= train_df.columns[557:]

NUM_VERT = 60
NUM_VERT_FEAT = 6
NUM_VERT_FEAT_Y = 6

FEAT_COLS = train_df.columns[61:60*4+1]
TARGET_COLS= train_df.columns[557+60:557+60*4]


NUM_2D_FEAT = len(FEAT_COLS) - NUM_VERT*NUM_VERT_FEAT
NUM_2D_FEAT_Y = len(TARGET_COLS) - NUM_VERT*NUM_VERT_FEAT_Y

NUM_2D_FEAT, NUM_2D_FEAT_Y
#%%
60*6
#%%
833/1e-6
#%%
y = train_df.select(TARGET_COLS).to_numpy()
x = train_df.select(FEAT_COLS).to_numpy()
#%%
a = y/(x + 1e-15)
min_fact = np.abs(a.min())
min_fact
#%%
subset = y[a > min_fact*2]
#%%
subset.shape
#%%
a[a > min_fact*2] = min_fact
#a[a == 0] = np.nan

for n, col in enumerate(TARGET_COLS):
    if n % 60 > 10:
        plt.figure(figsize=(12, 12))
        plt.hist(a[:, n], bins=100)
        plt.savefig(f'plots/{col}.png')
        plt.close()
#%%

#%%
((df_y/(df_x + 1e-15))*1e4).max()

#%%
# Show summary
desc = weightings.describe()
#%%
df_x = train_df.select(FEAT_COLS)
df_y = train_df.select(TARGET_COLS)
#%%

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
train_df.select(train_df.columns[1:557])
# %%

plt.plot(train_df[['cam_in_LANDFRAC', ]].to_numpy(),)
plt.xlim(0, 400)
#%%

# %%
