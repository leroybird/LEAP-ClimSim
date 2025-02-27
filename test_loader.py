#%%
%load_ext autoreload
%autoreload 2
#%%
import logging
from pathlib import Path

from matplotlib import pyplot as plt
from config import DataConfig, LoaderConfig, get_data_config
import dataloader
import pandas as pd
import numpy as np
import torch
import polars as pl

#%%
cfg_loader = LoaderConfig()
cfg_data = get_data_config(cfg_loader)
cfg_data, cfg_loader
#%%
df_index = pd.read_parquet('/mnt/ssd/kaggle/index.parquet')
grid_info_path = '/mnt/storage/kaggle/ClimSim_low-res_grid-info.nc'
root_folder = Path('/mnt/ssd/kaggle/train')
weights = pd.read_csv('/mnt/ssd/kaggle/sample_submission.csv', nrows=1)
weights = weights.iloc[0, 1:].values.astype(np.float32)

#%%
train_ds_sample = pl.read_parquet('/mnt/ssd/kaggle/train2.parquet', n_rows=100).to_pandas()
train_ds_sample.shape
#%%
len(weights)
#%%
df_index_tr = df_index[df_index['year'] <= 8]
df_index_val = df_index[df_index['year'] == 9]
assert len(df_index) == len(df_index_tr) + len(df_index_val)
#%%
train_dl, val_dl = dataloader.setup_dataloaders(loader_cfg=cfg_loader, data_cfg=cfg_data)
#%%
class_mask = train_dl.dataset.inner_ds.y_transform.class_mask
#%%
batch = next(iter(train_dl))
#%%
out = batch['y_cls'].numpy()
#%%
y0_amount = (out==0).sum(axis=0).astype(np.float32)/out.shape[0]
y1_amount = (out==1).sum(axis=0).astype(np.float32)/out.shape[0]
y2_amount = (out==2).sum(axis=0).astype(np.float32)/out.shape[0]
y3_amount = (out==3).sum(axis=0).astype(np.float32)/out.shape[0]


#%%
plt.figure(figsize=(12, 12))
plt.ylim(0, 1)
plt.plot(y0_amount, label='0')
plt.plot(y1_amount, label='1')
plt.plot(y2_amount, label='2')
plt.plot(y3_amount, label='3')
plt.legend()
#%%
y_norm = batch['y'][:, class_mask]
y_total = (y_norm**2).sum(axis=0)
#%%
data_all = []
for n in range(y_norm.shape[1]):
    print(n)
    s_tot = y_norm[:, n]**2
    s_0 = y_norm[out[:, n]==0, n]**2
    s_1 = y_norm[out[:, n]==1, n]**2
    s_2 = y_norm[out[:, n]==2, n]**2
    s_3 = y_norm[out[:, n]==3, n]**2
    
    data_all.append([s_0.sum()/s_tot.sum(), s_1.sum()/s_tot.sum(), s_2.sum()/s_tot.sum(), s_3.sum()/s_tot.sum()])


#%%
plt.plot(data_all, label=['0', '1', '2', '3'])
plt.legend()

#%%
plt.scatter(0, (y_norm[out==0]**2).sum(axis=0)/y_total.sum(), label='0')
plt.scatter(0, (y_norm[out==1]**2).sum(axis=0)/y_total.sum(), label='1')
plt.scatter(0, (y_norm[out==2]**2).sum(axis=0)/y_total.sum(), label='2')
plt.scatter(0, (y_norm[out==3]**2).sum(axis=0)/y_total.sum(), label='3')
plt.legend()

#%%
(y_norm[out==0]**2).sum(axis=0)/y_total.sum(), (y_norm[out==1]**2).sum(axis=0)/y_total.sum(), (y_norm[out==2]**2).sum(axis=0)/y_total.sum(), (y_norm[out==3]**2).sum(axis=0)/y_total.sum()
#%%
batch = next(iter(val_dl))

#%%
plt.figure(figsize=(20, 20))
plt.plot(x.max(dim=0).values[0:450])
plt.plot(x.min(dim=0).values[0:450])
plt.ylim(1, -1)
#%%


#%%
ds_train = dataloader.LeapLoader(root_folder, grid_info_path, df_index_tr,)
ds_train
#%%
ds_train.neighbours
#%%

#%%
x, y = ds_train[0]
#%%
x.shape
#%%
x.max()
#%%
train_ds_sample
#%%
x,y = ds_train[0]
#%%
x.shape[1] + y.shape[1]
#%%
x.coords['variable'].values
#%%
train_ds_sample.columns[60*6+1:60*6+1+16] == x.coords['variable'].values[60*6:60*6+16]
#%%
num_workers=24

#%%
def get_idxs(num, num_workers, seed=42):
    idxs = np.arange(num)
    np.random.seed(seed)
    idxs = np.random.permutation(idxs)[0:num - num % num_workers]
    idxs = np.array_split(idxs, num_workers)
    return idxs
#%%
len(get_idxs(100, 24))

#%%
class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, inner_ds, num_workers=24, seed=42, sample_size=16):
        self.num_workers = num_workers
        self.total_iterations = -1
        self.seed = seed
        self.inner_ds = inner_ds
        self.num_samples = len(inner_ds)
        self.sample_size = sample_size
        self.grid_points = 384
        assert self.grid_points % self.sample_size == 0
        self.inner_rep = self.grid_points // self.sample_size
        
    def gen(self, ):
        self.total_iterations += 1
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            logging.warning("No worker info")
            iter_idx = 0
        else:
            iter_idx = worker_info.id
        
        idxs = get_idxs(len(self.inner_ds), self.num_workers, 
                             self.seed + self.total_iterations)
        
        for idx in idxs[iter_idx]:
            # Each inner dataset contains 384 unique grid points
            ds_x_inner, ds_y_inner = self.inner_ds[idx]
            ds_x_inner = ds_x_inner.values
            ds_y_inner = ds_y_inner.values
            
            random_sample = np.random.permutation(self.grid_points)
            for n in range(self.inner_rep):
                ds_x = ds_x_inner[random_sample[n*self.sample_size:(n+1)*self.sample_size]]
                ds_y = ds_y_inner[random_sample[n*self.sample_size:(n+1)*self.sample_size]]
                yield ds_x, ds_y
        
    def __iter__(self):
        return self.gen()
    
#%%
def concat_collate(batch):
    x = [torch.from_numpy(b[0]) for b in batch]
    y = [torch.from_numpy(b[1]) for b in batch]
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    return x, y

#%%
# def pooled_batches(loader):
#     loader_it = iter(loader)
#     while True:
#         samples = []
#         for _ in range(loader.num_workers):
#             try:
#                 samples.append(next(loader_it))
#             except StopIteration:
#                 pass
#         if len(samples) == 0:
#             break
#         else:
#             yield T.cat(samples, dim=0)

#%%
inner_train_ds = dataloader.LeapLoader(root_folder, grid_info_path, df_index_tr)
train_ds = IterableDataset(inner_train_ds, num_workers=4)        
        
#%%        
train_dl = torch.utils.data.DataLoader(train_ds, num_workers=4, batch_size=4, collate_fn=concat_collate,
                                       )
#%%
x,y = next(iter(train_dl))
#%%
x.shape, y.shape
#%%
train_ds.df.iloc[train_ds.idxs[0]]
#%%
out = next(iter(train_dl))            
#%%
out
#%%