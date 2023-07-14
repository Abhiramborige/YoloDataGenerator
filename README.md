# YoloDataGenerator
Data generation with labelling made easy with CV2 and Albumentations
## Took help from:
1. https://github.com/srp-31/Data-Augmentation-for-Object-Detection-YOLO-
2. https://stackoverflow.com/questions/69620706/overlay-image-on-another-image-with-opencv-and-numpy

### config.yaml
```yaml
BACKGROUND_FILE_PATHS: ['path_to_background_images1', 'path_to_background_images2']
STRICTLY_NOT_INCLUDE_PATH: 'path_to_png_alpha_channel_icons_which_signify_false_negatives'
STRICTLY_INCLUDE_PATH: 'path_to_png_alpha_channel_icons_which_signify_true_positives'
FRAC_INCLUDE: 'float(0.0 < x < 1.0): fraction of total icons which are from STRICTLY_INCLUDE_PATH'
OUTPUT_PATH: 'path_for_storing_synthetic_data(both images and labels)'
OUTPUT_PER_SAMPLE: 'int: number of icons to put on background'
DIR_TO_PASS_TO_MODEL: 'path_for_storing_splitted_dataset'
SPLIT_FRACTION: 'float(0.0 < x < 1.0): splitting ratio'
BACKGROUND_FLIPS: list:[1, 0, -1]  # background flip horizontal, vertical, both at a time
SCALE_RANGE: list:[1.0, 1.4]       # scales the icons between the range randomly
ROTATE_ANGLES: list:[90, 270]      # rotation of background image.
RESTRICT_SCALE: list:[60,60]       # if icon exceeds the width(px) or heightpx) passed here, scaling wont happen
```
