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
```
