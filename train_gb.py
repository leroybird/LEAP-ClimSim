# %%
import lightgbm as lgb
import polars as pl
from pathlib import Path
import pandas as pd
import numpy as np
import sklearn.linear_model
from sklearn.metrics import r2_score

import norm
import config
#%%
cfg_loader = config.LoaderConfig()
cfg_data = config.get_data_config(cfg_loader)

norm_x, norm_y = norm.get_stats(cfg_loader, cfg_data)

# %%
base_path = Path("/mnt/storage/kaggle/raw_preds")
test_path = Path("/mnt/storage/kaggle/raw_preds_val")


# %%
def load_data(path):
    x = pl.read_parquet(path / "x_all.parquet").to_numpy()
    y = pl.read_parquet(path / "y_all.parquet").to_numpy()
    preds = pl.read_parquet(path / "pred_all.parquet").to_numpy()

    # x = np.concatenate([x, preds], axis=1)

    weightings = pd.read_csv("/mnt/ssd/kaggle/sample_submission.csv", nrows=1)
    weighting = weightings.iloc[0, 1:].values  # .astype(np.float32)

    y = y * weighting

    return x, y, preds, weighting


# %%
x, y, preds, weighting = load_data(base_path)
# %%
x_test, y_test, preds_test, _ = load_data(test_path)
# %%
lgbm_params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": ["l2"],
    "learning_rate": 0.15,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.7,
    "bagging_freq": 10,
    "verbose": -10,
    "max_depth": 10,
    "num_leaves": 128,
    "max_bin": 63,
    "num_iterations": 200,
    # "reg_alpha": 0.1,
    # "reg_lambda": 3.25,
    "device": "gpu",
}
# %%
preds.std(), y.std()
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
preds_model = []
ratios = []

# r2_base_lst = []
# r2_gb_lst = []
for i, w in enumerate(weighting):
    if w != 0:
        y_xgb = y[:, i] - preds[:, i]
        y_xgb_test = y_test[:, i] - preds_test[:, i]
        x_train_cat = np.concatenate([x, preds[:, i : i + 1]], axis=1)
        x_test_cat = np.concatenate([x_test, preds_test[:, i : i + 1]], axis=1)

        model = sklearn.linear_model.LinearRegression()
        model.fit(x_train_cat, y_xgb)

        # model = lgb.LGBMRegressor(**lgbm_params)

        # model.fit(
        #     x_train_cat,
        #     y_xgb,
        #     eval_set=[(x_test_cat, y_xgb_test)],
        #     eval_metric="mean_squared_error",
        #     callbacks=[lgb.log_evaluation(10), lgb.early_stopping(10)],
        # )

        # lgb_train = lgb.Dataset(x_train_cat, y_xgb)
        # lgb_eval = lgb.Dataset(x_test_cat, y_xgb_test, reference=lgb_train)

        # gbm = lgb.train(
        #     lgbm_params,
        #     lgb_train,
        #     valid_sets=[lgb_eval],
        # )

        preds_gbm = model.predict(x_test_cat)

        r2_gb = r2_score(y_test[:, i], preds_gbm + preds_test[:, i])
        r2_base = r2_score(y_test[:, i], preds_test[:, i])

    else:
        r2_base = 0  # r2_score(y_test[:, i], preds_test[:, i])
        r2_gb = 0

    # r2_base_lst.append(r2_base)
    # r2_gb_lst.append(r2_gb)
    # r2_ratio_lst.append(r2_ratio)
    preds_model.append(preds_gbm)
    print(f"Base: {r2_base:.5f}, GB: {r2_gb:.5f} diff: {r2_gb - r2_base:.5f} {i}")

# %%
import matplotlib.pyplot as plt
    
ratio = -(x_test[:, :len(weighting)] * weighting[None, :]) / 1200
#r2_ratio = r2_score(y_test[:, i], ratio)
#%%
mask = weighting != 0
# %%
ratio.shape
r2_score(y_test, ratio, multioutput="raw_values")
#%%
norm_y.zero_mask
#%%
preds_test[:, norm_y.zero_mask] = 0
#%% 

r2_score(y_test[:, mask], preds_test[:, mask])
#%%
y_test[0] - preds_test[0]
#%%

# %%
diff = np.array(r2_gb_lst) - np.array(r2_base_lst)
diff[np.isclose(diff, -1.0)] = 0
plt.plot(diff)
plt.ylim(-0.01, 0.01)
# %%
