# %%
import polars as pl
import torch
import norm
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
# %%
input_fname = Path("output/output_new_norm_bf.pt")

# %%
preds = torch.load(input_fname)
preds.shape
# %%
preds.std(axis=0)
# %%
test_df = pl.read_csv("/mnt/ssd/kaggle/new_data/test.csv")
test_df
# %%
# test_df.write_parquet("/mnt/ssd/kaggle/new_data/test.parquet")
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
norm_y.zero_mask.sum()
# %%
preds = norm_y.denorm(preds)
# %%
preds
# %%
diff_mask = norm_y.zero_mask.copy()
diff_mask.shape
# %%
assert (diff_mask | (weighting == 0)).sum() == diff_mask.sum()
# %%
ratio_mask = diff_mask & (weighting == 1)
#%%
ratio_mask_pd = np.concatenate([np.zeros(1, dtype=bool), ratio_mask])
# %%
out_df = pd.DataFrame({"sample_id": test_df["sample_id"]})
out_df[list(weightings.columns[1:])] = preds
# %%
assert np.isclose(out_df[list(weightings.columns[1:])].to_numpy(), preds).all()
# %%
out_df
# %%
out_df.iloc[:, ratio_mask_pd] = -(test_data[:, ratio_mask]) / 1200
# %%

# %%
assert out_df.iloc[:, 1:].iloc[:, weighting == 0].sum().sum() == 0
out_df.rename(columns={0: "sample_id"}, inplace=True)
# %%
out_df
# %%
out_df.to_parquet(input_fname.parent / input_fname.name.replace(".pt", "_base.parquet"), index=False)
#%%
out_df = pd.read_parquet(input_fname.parent / input_fname.name.replace(".pt", "_base.parquet"))

# %%
q_start, q_end = 120, 60*4
q_mask = ~diff_mask.copy()
q_mask[0:q_start] = False
q_mask[q_end:] = False
q_mask_pd = np.concatenate([np.zeros(1, dtype=bool), q_mask], axis=0)
#%%
q_mask.shape
#%%
assert (test_data[:, q_mask] >= 0).all() 
#%%
smaller_r = out_df.iloc[:, q_mask_pd].values < (-test_data[:, q_mask]) / 1200
# %%
av_smaller_r = smaller_r.sum(axis=0) / smaller_r.shape[0]
plt.plot(av_smaller_r)
#%%
out_df_2 = out_df.copy()
#%%
for n, idx_q in enumerate(np.where(q_mask_pd)[0]):
    s_m = smaller_r[:, n]
    out_df_2.values[s_m, idx_q] = -(test_data[s_m, idx_q-1]) / 1200
#%%
out_df_2.to_parquet(input_fname.parent / input_fname.name.replace(".pt", "_min_ratio.parquet"), index=False)
#%%
(out_df == out_df_2).all()

#%%