# %%
%load_ext autoreload
%autoreload 2
#%%
from collections import defaultdict
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
from eval import EvalLoader
import optuna

# %%
cfg_loader = config.LoaderConfig()
cfg_data = config.get_data_config(cfg_loader)
# %%
norm_x, norm_y = norm.get_stats(cfg_loader, cfg_data)

# %%
input_test = list(Path("model_preds/").glob("*test.pt"))
input_val = [
    fname.with_name(fname.name.replace("test.pt", "valid.pt")) for fname in input_test
]
#%%
input_val
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
#%%
all_val_av = np.stack(all_val, axis=0).mean(axis=0)
# %%
val_data_y = torch.load("val_data_y.pt")


# %%
def r2_score(y_true, y_pred):
    return 1 - ((y_true - y_pred) ** 2).mean() / 0.886


# %%
for fname, data in zip(input_val, all_val):
    print(fname)
    print(r2_score(val_data_y, data))


# %%
def get_dataset(dl):
    outputs = defaultdict(list)
    for batch in dl:
        for key, value in batch.items():
            outputs[key].append(value.numpy())
    outputs = {k: np.concatenate(v) for k, v in outputs.items()}
    return outputs


# %%
_, val_dl = dataloader.setup_dataloaders(cfg_loader, cfg_data)
# %%
test_df = pl.read_csv("/mnt/ssd/kaggle/new_data/test.csv")
test_df
# %%
test_data = test_df[:, 1:].to_numpy()
test_data.shape
# %%
test_ds = EvalLoader({"x": test_data}, {"x": norm_x})

test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=384,
    drop_last=False,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)
# %%
val_data_dict = get_dataset(val_dl)
# %%
val_data_dict.keys(), {k: v.shape for k, v in val_data_dict.items()}
# %%
test_data_dict = get_dataset(test_loader)
# %%
test_data_dict.keys(), {k: v.shape for k, v in test_data_dict.items()}
# %%
from sklearn.decomposition import PCA

# %%
n_components = 16
# %%

x_val_cat = np.concatenate([val_data_dict["x_1d"][:, 0:360], val_data_dict["x_p"]], axis=1
)
# %%
x_test_cat = np.concatenate([test_data_dict["x_1d"][:, 0:360], test_data_dict["x_p"]], axis=1)
# %%
pca = PCA(n_components=n_components)
pca.fit(x_val_cat)
# %%
pca.explained_variance_ratio_.cumsum()
#%%
import gc
gc.collect()
# %%
x_val_pca = pca.transform(x_val_cat)
x_test_pca = pca.transform(x_test_cat)
# %%

# %%
# val_data_y = []
# val_data_x = []
# for batch in val_dl:
#     val_data_y.append(batch["y"].numpy())

#     val_data_x.append(batch["x"].numpy())
# %%
# val_data = np.concatenate(val_data, axis=0)
# # %%
# torch.save(val_data, "val_data_y.pt")
# %%
# Take average# %%

# %%
fixed_params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": ["l2"],
    "num_iterations": 200,
    # "reg_alpha": 0.1,
    # "reg_lambda": 3.25,
    "device": "gpu",
    "random_state": 42,
}

# %%
import logging

# %%
# logging.basicConfig(
#     level=logging.INFO,
# )
#
# %%

# %%
# y_xgb = y[:, i] - preds[:, i]
# y_xgb_test = y_test[:, i] - preds_test[:, i]
# x_train_cat = np.concatenate([x, preds[:, i : i + 1]], axis=1)
# x_test_cat = np.concatenate([x_test, preds_test[:, i : i + 1]], axis=1)

# val_data_y.shape
# all_val[0].shape
# %%
np.random.seed(42)
x_train_mask = np.ones(val_data_y.shape[0], dtype=bool)
x_train_mask[int(val_data_y.shape[0]*0.8):] = False
x_val_mask = ~x_train_mask
#%%
len(val_data_y)
#%%
x_val_mask.sum(), x_train_mask.sum()
# %%
# torch.save(x_train_mask, "x_train_mask.pt")
# %%
test_av = np.stack(all_test, axis=0).mean(axis=0)
# %%
all_val_av.shape
# %%
r2_score(val_data_y, all_val_av)
#%%
# Set loglevel to warning
import logging
logging.basicConfig(level=logging.WARNING)
#%%
# %%
def run_xgb(i, params):
    y_val = val_data_y[x_val_mask, i] - all_val_av[x_val_mask, i]
    y_train = val_data_y[x_train_mask, i] - all_val_av[x_train_mask, i]

    x_val = np.concatenate(
        [a[x_val_mask, i : i + 1] for a in all_val]
        + [x_val_cat[x_val_mask, i : i + 1]],
        #+ [x_val_pca[x_val_mask, :]],
        axis=1,
    )
    x_train = np.concatenate(
        [a[x_train_mask, i : i + 1] for a in all_val]
        + [x_val_cat[x_train_mask, i : i + 1]],
        #+ [x_val_pca[x_train_mask, :]],
        axis=1,
    )

    x_test = np.concatenate(
        [a[:, i : i + 1] for a in all_test]
        + [x_test_cat[:, i : i + 1]],
        #+ [x_test_pca],
        axis=1,
    )
    
    # x_mask_2 = np.ones(x_train.shape[0], dtype=bool)
    # x_mask_2[int(x_train.shape[0]*0.8):] = False
    # x_tr = x_train[x_mask_2]
    # y_tr = y_train[x_mask_2]
    
    # x_train_v = x_train[~x_mask_2]
    # y_train_v = y_train[~x_mask_2]

    # model = sklearn.linear_model.LinearRegression()
    # model.fit(x_train_cat, y_xgb)

    model = lgb.LGBMRegressor(**params)

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="mean_squared_error",
        callbacks=[lgb.log_evaluation(10), lgb.early_stopping(10)],
    )
    preds_gbm = model.predict(x_test)

    mse_gb = ((y_val - model.predict(x_val)) ** 2).mean()
    mse_base = (y_val**2).mean()

    return {
        "model": model,
        "preds_gbm": preds_gbm,
        "mse_gb": mse_gb,
        "mse_base": mse_base,
    }


# %%
x_val_pca.shape

#%%
#%%

# from https://github.com/optuna/optuna-examples/blob/main/xgboost/xgboost_simple.py



#%%

trail_idxs = [5, 20, 50, 80, 150, 170, 230, 330, 360]
#%%
norm_y.zero_mask[trail_idxs]
#%%
# fixed_params = {"boosting_type": "dart",
#                 "objective": "regression",
#                 "metric": ["l2"],
#                 "num_iterations": 100,
#                 "device": "gpu",
#                 "random_state": 42,
#                 "verbose" : -1}

def objective(trial):
    param = {
        **fixed_params,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        #"max_depth": trial.suggest_int("max_depth", 2, 20),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        #"bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        #"min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.2),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
    }
    
    total_mse = 0
    for idx in trail_idxs:
        total_mse += run_xgb(idx, param)['mse_gb']
        
    return total_mse / len(trail_idxs)
    



study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200, n_jobs=3)
#%%
print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

#%%
import json
with open("best_lgb_params.json", "w") as f:
    json.dump(trial.params, f, indent=4)
#%%
with open("best_lgb_params.json", "r") as f:
    best_params = json.load(f)

# %%

best_params
#%%
preds_model = []
ratios = []

r2_base_lst = []
r2_gb_lst = []

for i, w in enumerate(norm_y.zero_mask):
    if w == False:
        out_dict = run_xgb(i, {**fixed_params, **best_params})
        preds_gbm = out_dict["preds_gbm"]
        mse_gb = out_dict["mse_gb"]
        mse_base = out_dict["mse_base"]
    else:
        print(f"Skipping {i}")
        mse_base = 0  # r2_score(y_test[:, i], preds_test[:, i])
        mse_gb = 0
        preds_gbm = np.zeros(x_test_cat.shape[0])

    r2_base_lst.append(mse_base)
    r2_gb_lst.append(mse_gb)
    # r2_ratio_lst.append(r2_ratio)
    preds_model.append(preds_gbm)
    print(f"Base: {mse_base:.5f}, GB: {mse_gb:.5f} diff: {mse_gb - mse_base:.5f} {i}")
# %%
# Add average preds
1  - 0.00477  - np.mean([min(b, g) for b, g in zip(r2_base_lst, r2_gb_lst)]) / 0.8806
#%%
preds_gbm = np.stack(preds_model, axis=1)
# %%
preds_final = preds_gbm + test_av
#%%
torch.save(preds_final, "sub6.pt")

# %%
# Select the best out of avg
preds_cmb = []
for idx, (a,b) in enumerate(zip(r2_base_lst, r2_gb_lst)):
    if a < b:
        preds_cmb.append(test_av[:, idx])
    else:
        preds_cmb.append(preds_final[:, idx])
preds_cmb = np.stack(preds_cmb, axis=1)
#%%
preds_cmb.shape
# %%
torch.save(preds_cmb, "sub7_cmb.pt")
# %%
import matplotlib.pyplot as plt

plt.plot(r2_base_lst)
plt.plot(r2_gb_lst)
# %%
np.mean(r2_base_lst)
# %%
np.mean(r2_gb_lst)
# %%
# 0.7877
0.886 / (0.7877 / 0.78298)

# %%
(1.0 - 0.00477 - np.mean(r2_gb_lst) / 0.8806)
# %%
(1.0 - np.mean(r2_gb_lst) / 0.8806)

# %%
1 - np.mean(r2_base_lst) / 0.8806
# %%
# %%

# %%
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
