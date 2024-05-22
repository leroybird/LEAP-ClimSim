#%%
%load_ext autoreload
%autoreload 2
#%%
from pathlib import Path
import pandas as pd
import re
from collections import defaultdict
#%%
root_dir = Path('/mnt/storage/kaggle/train')
output_path = Path('/mnt/ssd/kaggle/index_fwd.parquet')

#%%
files = list(root_dir.rglob('*.mli.*.nc'))
files_rel = [f.relative_to(root_dir) for f in files]
len(files_rel)
#%%
ens_year_map = defaultdict(list)
for fname in files_rel:
    ens_year_map[fname.parent.name].append(str(fname))

#%%
for k, path in ens_year_map.items():
    ens_year_map[k] = sorted(path)
#%%
ens_year_map[k]
#%%
output = defaultdict(list)

# output is every 20mins (3x per hour), y_test is every 10 hours
time_offset = 3*10
#%%
for k, path in ens_year_map.items():
    output['prev_path'].extend(path[0:-2*time_offset])
    output['next_path'].extend(path[2*time_offset:])
    
    p_sub = path[time_offset:-time_offset]
    
    output['path'].extend(p_sub)
    output['seconds'].extend([int(re.search(r'(\d{5}).nc', x).group(1)) for x in p_sub])
    output['day'].extend([int(re.search(r'-(\d{2})-(\d{5}).nc', x).group(1)) for x in p_sub])
    year, month = k.split('-')
    output['year'].extend([int(year)]*len(p_sub))
    output['month'].extend([int(month)]*len(p_sub))
    
#%%


{len(v) for v in output.values()}
#%% 
output = pd.DataFrame(output)
#%%
output
#%%
output['year'].value_counts()
#%%
output
#%%
output.to_parquet(output_path, index=False)
#%%

#%%
