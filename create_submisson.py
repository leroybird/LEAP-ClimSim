# %%
from cgi import test
import polars as pl
import torch
import norm
import config
import numpy as np
import pandas as pd
# %%
input_fname = "output_time.pt"
fill_df = pl.read_csv("output_base_v1.csv").to_pandas()
weights = torch
offset = 384
# %%
preds = torch.load(input_fname)
preds.shape
# %%
test_df = pl.read_parquet("/mnt/ssd/kaggle/test.parquet")
test_df
#%%
test_data = test_df[:, 1:preds.shape[1]+1].to_numpy()
test_data.shape
#%%
test_data_mod = test_data.shape[0] % 384
test_data_mod
#%%
weightings = pd.read_csv('/mnt/ssd/kaggle/sample_submission.csv', nrows=1)
weighting = weightings.iloc[0, 1:].values.astype(np.float32)
weightings.shape
# %%
cfg_loader = config.LoaderConfig()
cfg_data = config.get_data_config(cfg_loader)

norm_x, norm_y = norm.get_stats(cfg_loader, cfg_data)
#%%
norm_y.zero_mask
#%%
# %%
preds.shape
# %%

assert preds.shape[0] == fill_df.shape[0] - offset*2 - test_data_mod
# %%
# %%
# Set all predictions from offset to the end, apart from the first column
fill_df.iloc[offset:-offset-test_data_mod, 1:] = preds
# %%
diff_mask = norm_y.zero_mask.copy()
diff_mask.shape
#%%
diff_mask[60*4:] = 0
#%%
test_data.shape
#%%
fill_df.shape
#%%
diff_mask_pd = np.concatenate([np.zeros(1, dtype=bool), diff_mask])
#%%
diff_mask_pd.shape
#%%
test_data.shape
#%%
fill_df.iloc[:, diff_mask_pd] = -(test_data[:, diff_mask]*weighting[diff_mask])/1200
#%%
test_data.shape
#%%

# %%
fill_df.to_parquet("change_vars.parquet", index=False)
# %%
