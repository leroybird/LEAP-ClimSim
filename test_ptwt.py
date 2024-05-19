#%%
from ptwt._stationary_transform import _swt
import torch
import matplotlib.pyplot as plt
#%%
data_sz = 60
data = torch.arange(data_sz, )*0.2 + torch.sin(torch.arange(0, data_sz)*0.3) + torch.randn(data_sz)*0.5
#%%
plt.plot(data)
# %%
wavelets = torch.concatenate(_swt(data, 'db6', 2,) + [data[None, :]], dim=0)
#%%
wavelets.shape
# %%
#%%
for w in wavelets:
    plt.plot(w.numpy())

#%%