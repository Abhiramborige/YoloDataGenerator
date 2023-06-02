import os
import splitfolders
import yaml

with open("C:\ModifiedTouchAutomate\Datagenerator\config.yaml") as fp:
    config_params = yaml.load(fp, Loader=yaml.FullLoader)

old_path_dir = config_params.get("OUTPUT_PATH")
new_path_dir = config_params.get("DIR_TO_PASS_TO_MODEL")
os.makedirs(new_path_dir, exist_ok = True)
splitfolders.ratio(old_path_dir, output=new_path_dir, seed=16,
                   ratio=(0.90,0.10), group_prefix=None, move=True)
