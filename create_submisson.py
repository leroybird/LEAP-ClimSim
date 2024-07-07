# %%
import polars as pl
import torch
import norm
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
input_fname = "out_n4.pt"
offset = 384
# %%
preds = torch.load(input_fname)
preds.shape
# %%
preds.std(axis=0)
# %%
test_df = pl.read_csv("/mnt/ssd/kaggle/new_data/test.csv")
test_df
#%%
#test_df.write_parquet("/mnt/ssd/kaggle/new_data/test.parquet")
# %%
test_data = test_df[:, 1 : preds.shape[1] + 1].to_numpy()
test_data.shape
# %%
assert len(test_data) == len(preds)
# %%
weightings = pd.read_csv("/mnt/storage/kaggle/new_data/sample_submission.csv", nrows=1)
weighting = weightings.iloc[0, 1:].values.astype(np.float32)
weighting
# %%
assert len(weighting) == preds.shape[1]
# %%
cfg_loader = config.LoaderConfig()
cfg_data = config.get_data_config(cfg_loader)

norm_x, norm_y = norm.get_stats(cfg_loader, cfg_data)
del norm_x
# %%
norm_y.zero_mask
# %%
preds = norm_y.denorm(preds)
#%%
preds
# %%
diff_mask = norm_y.zero_mask.copy()
diff_mask.shape
#%%
diff_mask
#%%
diff_mask
#%%
weighting == 1
#%%
missing_mask = ((weighting == 1) & diff_mask)
missing_mask
#%%
missing_mask_pd = np.concatenate([np.zeros(1, dtype=bool), missing_mask])
# %%
out_df = pd.DataFrame({'sample_id' : test_df["sample_id"]})
out_df[list(weightings.columns[1:])] = preds 
#%%
assert np.isclose(out_df[list(weightings.columns[1:])].to_numpy(), preds).all()
#%%
out_df
#%%
missing_mask_pd
#%%
out_df.iloc[:, missing_mask_pd] = -(test_data[:, missing_mask])/1200
# %%
out_df
# %%
assert out_df.iloc[:, 1:].iloc[:, weighting == 0].sum().sum() == 0
out_df.rename(columns={0 : "sample_id"}, inplace=True)
#%%
out_df.columns
# %%
out_df.to_parquet("norm_test_200.parquet", index=False)
#%%
#%%
