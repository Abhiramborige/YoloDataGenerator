# YoloDataGenerator
Data generation with labelling made easy with CV2 and Albumentations
## Took help from:
1. https://github.com/srp-31/Data-Augmentation-for-Object-Detection-YOLO-
2. https://stackoverflow.com/questions/69620706/overlay-image-on-another-image-with-opencv-and-numpy

### config.yaml
```yaml
BACKGROUND_FILE_PATHS: ['path_to_background_images1', 'path_to_background_images2']
SAMPLE_FILES_PATH: 'path_to_png_alpha_channel_icons'
OUTPUT_PATH: 'path_for_storing_synthetic_data(both images and labels)'
OUTPUT_PER_SAMPLE: 'int: number of icons to put on background'
# include here below to be labelled, rest all will not be labelled
INCLUDE_FILE_NAMES: ['full file name to be labelled']
OUTPUT_PATH: 'path_to_save_synthetic_data'
FRACTION_OF_INCLUDE_FILE: 'float(0.0 < x < 1.0): fraction of total icons that has to be labelled'
DIR_TO_PASS_TO_MODEL: 'path_to_store_split_data'
BACKGROUND_FLIP_REQ: 'boolean: flip the background in all ways or not'
```
