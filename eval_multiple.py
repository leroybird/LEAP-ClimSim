#%%
from pathlib import Path

#%%

input_dir = '/mnt/storage/kaggle/check_base'
fnames = list(Path(input_dir).glob('*.ckpt'))
#%%
#fnames = [str(fname) for fname in fnames if 'daily-valley' in str(fname)]
fnames = [str(fname) for fname in fnames]
fnames
#%%
configs = []
for fn in fnames:
    name = fn.split('/')[-1].split('-')[0] + '-' + fn.split('/')[-1].split('-')[1]
    print(name)
    targets = list(Path('model_configs').glob(f'{name}*.yaml'))
    assert len(targets) == 1
    configs.append(targets[0])
#%%
configs
#%%
for check, config in zip(fnames, configs):
    !CUDA_VISIBLE_DEVICES=0 python eval.py --test_df /mnt/ssd/kaggle/new_data/test.csv --preds_only --output model_preds/ --model {str(check)} --model_config={str(config)} --val_preds 
#%%
#for check, config in zip(fnames, configs):
for check in fnames:
    !CUDA_VISIBLE_DEVICES=0 python eval.py --test_df /mnt/ssd/kaggle/new_data/test.csv --preds_only --output model_preds/ --model {str(check)} --val_preds 
#%%
1
#%%