# %%
import lightgbm as lgb
import polars as pl
from pathlib import Path
import pandas as pd
import numpy as np
import sklearn.linear_model
from sklearn.metrics import r2_score
import dataloader
import norm
import config
import torch
import tqdm

# %%
cfg_loader = config.LoaderConfig()
cfg_data = config.get_data_config(cfg_loader)

norm_x, norm_y = norm.get_stats(cfg_loader, cfg_data)

# %%
input_test = list(Path("model_preds/").glob("*test.pt"))
input_val = [
    fname.with_name(fname.name.replace("test.pt", "valid.pt")) for fname in input_test
]
# %%
for fname in input_test:
    assert fname.exists()
# %%
all_test = []
for fname in input_test:
    all_test.append(torch.load(fname)["reg"])
# %%
all_val = []
for fname in input_val:
    all_val.append(torch.load(fname)["reg"])

# %%
_, val_dl = dataloader.setup_dataloaders(cfg_loader, cfg_data)
# %%
# val_data_y = []
# val_data_x = []
# for batch in val_dl:
#     val_data_y.append(batch["y"].numpy())

#     val_data_x.append(batch[0].numpy())
# %%
# val_data = np.concatenate(val_data, axis=0)
# # %%
# torch.save(val_data, "val_data_y.pt")
# %%
# Take average# %%
val_data = torch.load("val_data_y.pt")
# %%
test_df = pl.read_csv("/mnt/ssd/kaggle/new_data/test.csv")
test_df
# %%
# test_df.write_parquet("/mnt/ssd/kaggle/new_data/test.parquet")
# %%
test_data = test_df[:, 1 : val_data.shape[1] + 1].to_numpy()
test_data.shape
# %%

# # %%
# x_test, y_test, preds_test, _ = load_data(test_path)
# %%
lgbm_params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": ["l2"],
    "num_iterations": 100,
    # "reg_alpha": 0.1,
    # "reg_lambda": 3.25,
    "device": "gpu",
    "random_state": 42,
}
# %%
output_preds = []
# %%
import logging

# %%
logging.basicConfig(
    level=logging.INFO,
)
#
import sklearn.linear_model

# %%

# %%
# y_xgb = y[:, i] - preds[:, i]
# y_xgb_test = y_test[:, i] - preds_test[:, i]
# x_train_cat = np.concatenate([x, preds[:, i : i + 1]], axis=1)
# x_test_cat = np.concatenate([x_test, preds_test[:, i : i + 1]], axis=1)

val_data.shape
all_val[0].shape
# %%
np.random.seed(42)
x_train_mask = np.random.sample(len(val_data)) < 0.8
x_val_mask = ~x_train_mask
# %%
all_val_av = np.stack(all_val, axis=0).mean(axis=0)

# %%
i = 0
y_val = val_data[x_val_mask, i] - all_val_av[x_val_mask, i]
y_train = val_data[x_train_mask, i] - all_val_av[x_train_mask, i]

x_val = np.stack([a[x_val_mask, i] for a in all_val], axis=1)
x_train = np.stack([a[x_train_mask, i] for a in all_val], axis=1)

x_test = np.stack([a[:, i] for a in all_test], axis=1)
# %%
y_val.shape, y_train.shape, x_val.shape, x_train.shape, x_test.shape
# %%
model = lgb.LGBMRegressor(**lgbm_params)

lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)

model.fit(
    x_train,
    y_train,
    eval_set=[(x_val, y_val)],
    eval_metric="mean_squared_error",
    callbacks=[lgb.log_evaluation(10), lgb.early_stopping(10)],
)
# %%
((y_val) ** 2).mean()
# %%

# %%
preds_model = []
ratios = []

r2_base_lst = []
r2_gb_lst = []

for i, w in enumerate(norm_y.zero_mask):
    if w == False:
        y_val = val_data[x_val_mask, i] - all_val_av[x_val_mask, i]
        y_train = val_data[x_train_mask, i] - all_val_av[x_train_mask, i]

        x_val = np.stack([a[x_val_mask, i] for a in all_val], axis=1)
        x_train = np.stack([a[x_train_mask, i] for a in all_val], axis=1)

        x_test = np.stack([a[:, i] for a in all_test], axis=1)

        # model = sklearn.linear_model.LinearRegression()
        # model.fit(x_train_cat, y_xgb)

        model = lgb.LGBMRegressor(**lgbm_params)

        result = model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="mean_squared_error",
            callbacks=[lgb.log_evaluation(10), lgb.early_stopping(10)],
        )

        preds_gbm = model.predict(x_test)

        mse_gb = ((y_val - model.predict(x_val)) ** 2).mean()
        mse_base = (y_val**2).mean()

    else:
        print(f"Skipping {i}")
        mse_base = 0  # r2_score(y_test[:, i], preds_test[:, i])
        mse_gb = 0
        preds_gbm = np.zeros(x_test.shape[0])

    r2_base_lst.append(mse_base)
    r2_gb_lst.append(mse_gb)
    # r2_ratio_lst.append(r2_ratio)
    preds_model.append(preds_gbm)
    print(f"Base: {mse_base:.5f}, GB: {mse_gb:.5f} diff: {mse_gb - mse_base:.5f} {i}")
#%%
# Add average preds 
preds_gbm = np.stack(preds_model, axis=1)
# %%
test_av = np.stack(all_test, axis=0).mean(axis=0)
preds_final = preds_gbm + test_av
#%%
torch.save(preds_final, "sub.pt")
#%%
import matplotlib.pyplot as plt
plt.plot(r2_base_lst)
plt.plot(r2_gb_lst)
#%%
np.mean(r2_base_lst)
#%%
np.mean(r2_gb_lst)
#%%
1 - np.mean(r2_gb_lst) / 0.886
#%%
1 - np.mean(r2_base_lst) / 0.886
#%%

ratio = -(x_test[:, : len(weighting)] * weighting[None, :]) / 1200
# r2_ratio = r2_score(y_test[:, i], ratio)
# %%
mask = weighting != 0
# %%
ratio.shape
r2_score(y_test, ratio, multioutput="raw_values")
# %%
norm_y.zero_mask
# %%
preds_test[:, norm_y.zero_mask] = 0
# %%

r2_score(y_test[:, mask], preds_test[:, mask])
# %%
y_test[0] - preds_test[0]
# %%

# %%
diff = np.array(r2_gb_lst) - np.array(r2_base_lst)
diff[np.isclose(diff, -1.0)] = 0
plt.plot(diff)
plt.ylim(-0.01, 0.01)
# %%
