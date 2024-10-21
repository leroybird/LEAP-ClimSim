#%%
%load_ext autoreload
%autoreload 2
#%%
from pathlib import Path
import pandas as pd
import re
from collections import defaultdict

#%%
root_dir = Path('/mnt/storage2/models/hrrr')
output_path = root_dir / 'index.parquet'
re_pattern = r"(\d{4})/(\d{2})/(\d{2})/hrrr\.t(\d{2})z\.wrfnatf(\d{2})\.grib2"

#%%
input_files = list(root_dir.rglob('*.grib2'))
files_rel = [str(x.relative_to(root_dir)) for x in input_files]
len(input_files)
#%%
def get_dt_hour(fname, pattern=re_pattern):
    match = re.match(pattern, str(fname))
    year, month, day, init_time, forecast_timestep = match.groups()
    init_time = pd.Timestamp(f'{year}-{month}-{day} {init_time}:00:00')
    return init_time, int(forecast_timestep)
files_rel[0], get_dt_hour(files_rel[0])
#%%
df_index = pd.DataFrame(files_rel, columns=['path'])
df_index['init_time'], df_index['forecast_timestep'] = zip(*df_index['path'].map(get_dt_hour))
df_index['time'] = df_index['init_time'] + pd.to_timedelta(df_index['forecast_timestep'], unit='h')
#%%
def get_next_step(df, max_steps=4):
    next_paths = {f"next_path_{n}": ['']*len(df) for n in range(1, max_steps+1)}
    num_steps = []
    
    for idx, (_, row) in enumerate(df.iterrows()):
        df_init = df[df['init_time'] == row['init_time']]
        fc_hour = row['forecast_timestep']
        steps = 0
        for n in range(1, max_steps+1):
            next_fc = fc_hour + n
            
            if next_fc in df_init['forecast_timestep'].values:
                next_paths[f'next_path_{n}'][idx] = df_init[df_init['forecast_timestep'] == next_fc]['path'].values[0]                   
            else:
                break
            steps += 1
            
        num_steps.append(steps)
        
    df = df.copy()
    df['num_steps'] = num_steps
    for k, v in next_paths.items():
        df[k] = v
    
    return df    
    
#%%
df_index = get_next_step(df_index)
#%%
# drop rows with no next step
df_index = df_index[df_index['num_steps'] > 0]
#%%
df_index
#%%
df_index.to_parquet(output_path, index=False)
#%%
