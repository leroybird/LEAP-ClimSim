# %%
import polars as pl
from scipy import stats
import torch

# %%
input_fname = "output_time_v2.pt"
fill_df = pl.read_csv("output_base_v1.csv").to_pandas()
weights = torch
offset = 384
# %%
preds = torch.load(input_fname)
# %%
# %%
preds.shape
# %%
assert preds.shape[0] == fill_df.shape[0] - offset*2
# %%
# %%
# Set all predictions from offset to the end, apart from the first column
fill_df.iloc[offset:-offset, 1:] = preds

# %%


# %%
fill_df.to_parquet(input_fname.replace(".pt", ".parquet"), index=False)
# %%
