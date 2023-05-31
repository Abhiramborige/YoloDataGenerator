import albumentations as A
import cv2
import os
import yaml

with open("./config.yaml") as fp:
    config_params = yaml.load(fp, Loader=yaml.FullLoader)

path_img_dir = config_params.get("OUTPUT_PATH")+"images/"
path_label_dir = config_params.get("OUTPUT_PATH")+"labels/"

def augument_data_dir(img_dir, label_dir):
    img_list = os.listdir(img_dir)
    label_list = os.listdir(label_dir)
    for path_img, path_label in zip(img_list, label_list):
        image = cv2.imread(os.path.join(path_img_dir, path_img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = []
        class_labels = []
        with open(os.path.join(path_label_dir, path_label), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = list(map(float, line.strip().split(" ")))
                line.append(int(line[0]))
                del line[0]
                bboxes.append(line)
        yield bboxes, image, path_img, path_label


def convert_list_to_yolo_file(bboxes, file_path):
    with open(file_path, 'w+') as f:
        for icon_pos in bboxes:
            temp_str = str(icon_pos[-1])
            for coord in icon_pos[:len(icon_pos) - 1]:
                temp_str += (" " + str(round(coord, 6)))
            temp_str += "\n"
            f.write(temp_str)

def transformation(image, bboxes, path_image, path_label, transformer, suffix):
    try:
        transformed = transformer(image=image, bboxes=bboxes)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(path_img_dir,
                                 f'{path_img.split(".")[0]}_{suffix}.png'),transformed_image)
        convert_list_to_yolo_file(transformed_bboxes,
                                  os.path.join(path_label_dir,
                                               f'{path_label.split(".")[0]}_{suffix}.txt'))
        print(f"Operation: {suffix} on: ", path_img)
    except Exception as e:
        print(e)
        print("Error occurred")


transform_flip = A.Compose([
    A.HorizontalFlip(p=1),
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.5))
transform_brightness_contrast = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.6,
                               brightness_by_max=True, always_apply=True, p=1),
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.5))
transform_scale = A.Compose([
    A.RandomScale(scale_limit=(0.3, 0.6), interpolation=cv2.INTER_AREA, p=1, always_apply=True),
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.5))
transform_affine = A.Compose([
    A.Affine(keep_ratio=True, translate_percent=(0.1, 0.2), p=1, scale=(0.6, 1.2)),
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.5))
transform_grid = A.Compose([
    A.GridDistortion(num_steps=10, distort_limit=0.2, interpolation=cv2.INTER_AREA,
                     normalized=True, always_apply=True, p=1),
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.5))
transform_optical = A.Compose([
    A.OpticalDistortion(distort_limit=0.3, shift_limit=0.2, interpolation=cv2.INTER_AREA,
                        always_apply=True, p=1),
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.5))
transform_perspective = A.Compose([
    A.Perspective(scale=0.05, fit_output=True, always_apply=True, p=1),
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.5))
transform_resize_pad = A.Compose([
    A.LongestMaxSize(p=1, max_size=1280, always_apply=True, interpolation=cv2.INTER_AREA),
    A.PadIfNeeded(p=1, min_height=1280, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), mask_value=(0, 0, 0))
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.5))
transform_safecrop = A.Compose([
    A.BBoxSafeRandomCrop(erosion_rate=0.3, always_apply=True, p=1)
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.5))
transform_safecropscale = A.Compose([
    A.RandomSizedBBoxSafeCrop(height=1280, width=1280, erosion_rate=0.3,
                              interpolation=cv2.INTER_AREA, always_apply=True, p=1)
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.5))
transform_contrast = A.Compose([
    A.RandomContrast(limit=0.4, p=1, always_apply=True),
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.5))
transform_brightness = A.Compose([
    A.RandomBrightness(limit=0.4, p=1, always_apply=True),
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.5))
transform_equalize = A.Compose([
    A.Equalize(mode='cv', by_channels=False, p=1, always_apply=True),
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.5))
transform_downscale = A.Compose([
    A.Downscale(scale_min=0.2, scale_max=0.2, interpolation=cv2.INTER_AREA, p=1, always_apply=True),
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.5))

generator = augument_data_dir(path_img_dir, path_label_dir)
while True:
    try:
        bboxes, image, path_img, path_label = generator.__next__()
        transformation(image, bboxes, path_img, path_label, transform_flip, 'flip')
        transformation(image, bboxes, path_img, path_label, transform_brightness_contrast, 'brightcontrast')
        transformation(image, bboxes, path_img, path_label, transform_scale, 'scale')
        transformation(image, bboxes, path_img, path_label, transform_affine, 'affine')
        transformation(image, bboxes, path_img, path_label, transform_grid, 'grid')
        transformation(image, bboxes, path_img, path_label, transform_optical, 'optical')
        transformation(image, bboxes, path_img, path_label, transform_perspective, 'perspective')
        transformation(image, bboxes, path_img, path_label, transform_resize_pad, 'resizenpad')
        transformation(image, bboxes, path_img, path_label, transform_safecrop, 'safecrop')
        transformation(image, bboxes, path_img, path_label, transform_safecropscale, 'safecropscale')
        transformation(image, bboxes, path_img, path_label, transform_brightness, 'brightness')
        transformation(image, bboxes, path_img, path_label, transform_equalize, 'equalize')
        transformation(image, bboxes, path_img, path_label, transform_downscale, 'downscale')
    except Exception as e:
        print(e)
        print("Generator empty")
        break
    finally:
        print("Completed")
