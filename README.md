Occlusion Data Generate and view
---

suitable for: INTERACTION dataset v1 and v1_2 [v2 in code]

Remember to change the path in the code.
``` python
main_folder = '/home/kin/DATA_HDD/yy/INTERACTION-Dataset-DR-multi-v1_2/'

# will generate new csv files here
reformat_folder = '/home/kin/DATA_HDD/yy/INTERACTION-Dataset-DR-multi-v1_2/occ_files/'
```

Run generate data will have `occluded` label in the new csv files.

```bash
python3 src/generate_data_v2.py
```