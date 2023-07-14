import numpy as np
import cv2 as cv2
import os
import glob
import yaml
# https://github.com/srp-31/Data-Augmentation-for-Object-Detection-YOLO-

with open("./config.yaml" ,"r") as fp:
    config_params = yaml.load(fp, Loader=yaml.FullLoader)

background_file_paths = config_params.get("BACKGROUND_FILE_PATHS")
include_icon_path = config_params.get("STRICTLY_INCLUDE_PATH")
not_include_icon_path = config_params.get("STRICTLY_NOT_INCLUDE_PATH")
fraction_of_include = config_params.get("FRAC_INCLUDE")

outputfolder = config_params.get("OUTPUT_PATH")
icon_per_sample = config_params.get("OUTPUT_PER_SAMPLE")
flip_arr = config_params.get("BACKGROUND_FLIPS")
scale_range = config_params.get("SCALE_RANGE")
rotate_arr = config_params.get("ROTATE_ANGLES")
restrict_range = config_params.get("RESTRICT_SCALE")

os.makedirs(outputfolder+"images/", exist_ok=True)
os.makedirs(outputfolder+"labels/", exist_ok=True)
include_icons = []
include_icon_names = []
not_include_icons = []
not_include_icon_names = []

# icons to be put on image which require labelling
for name in os.listdir(include_icon_path):
    include_icons.append(
        cv2.imread(os.path.join(include_icon_path, name), cv2.IMREAD_UNCHANGED))
    include_icon_names.append(name)
# icons to be put on image which doesnt require labelling
for name in os.listdir(not_include_icon_path):
    not_include_icons.append(
        cv2.imread(os.path.join(not_include_icon_path, name), cv2.IMREAD_UNCHANGED))
    not_include_icon_names.append(name)
print(include_icon_names)
print(not_include_icon_names)


def rotate_image(image, angle):
    img_center = tuple(np.array(image.shape[1::-1])/2)
    rot_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], 
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(55,28,8)
    )
    return result

def scaleImage(image, scale):
    return cv2.resize(image, None, fx=scale, fy=scale)

def place_distorted_sample(bkgImg, iconImg, occupancy_bits):
    bgHeight, bgWidth, _ = np.shape(bkgImg)
    iconHeight, iconWidth, _ = np.shape(iconImg)
    if iconHeight < bgHeight and iconWidth < bgWidth:
        posX = np.random.randint(0, bgWidth - iconWidth)
        posY = np.random.randint(0, bgHeight - iconHeight)
        print("Position set as: ", (posX, posY))

        # extract alpha channel from foreground image as mask and make 3 channels
        alpha = iconImg[:,:,3]/255.0
        alpha = cv2.merge([alpha,alpha,alpha])
        iconImg = iconImg[:,:,0:3]

        # Image ranges, position coordinates of icon on image
        y1, y2 = max(0, posY), min(bgHeight, posY + iconHeight)
        x1, x2 = max(0, posX), min(bgWidth, posX + iconWidth)

        # check for enough space to put icon
        occupancy_crop = occupancy_bits[y1:y2, x1:x2]
        if len(occupancy_crop[occupancy_crop == 1])/np.prod(occupancy_crop.shape)!=0:
            raise AssertionError("Icon doesnt have enough space.")
        occupancy_crop[occupancy_crop==0] = 1

        # Blend overlay within the determined ranges
        # https://stackoverflow.com/questions/69620706/overlay-image-on-another-image-with-opencv-and-numpy
        img_crop = bkgImg[y1:y2, x1:x2]
        alpha_inv = 1.0 - alpha
        img_crop[:] = alpha * iconImg + alpha_inv * img_crop

        # [[center_x, center_y],[width, height]]
        boundRectFin = np.zeros((2, 2), float)
        # The order of x and y have been reversed for yolo
        boundRectFin[1][1] = round((float(iconHeight)+10) / float(bgHeight) ,6)
        boundRectFin[1][0] = round((float(iconWidth)+10) / float(bgWidth) ,6)
        boundRectFin[0][1] = round((float(posY-5) / float(bgHeight) + boundRectFin[1][1] / float(2)) ,6)
        boundRectFin[0][0] = round((float(posX-5) / float(bgWidth) + boundRectFin[1][0] / float(2)) ,6)
        return bkgImg, boundRectFin
    else:
        raise AssertionError("Icon dimensions greater than image.")

def choose_and_place_sample(bkgImg, occupancy_bits, 
                         filenameWithExt, file_ptr, 
                         label_req = True):
    # randomly choose the icon from include_files/exclude_files to put on image
    if label_req:
        randnum = np.random.randint(0, len(include_icons))
        icon_img = include_icons[randnum]
        label = include_icon_names[randnum]
    else:
        randnum = np.random.randint(0, len(not_include_icons))
        icon_img = not_include_icons[randnum]
        label = not_include_icon_names[randnum]

    icon_height, icon_width, _ = np.shape(icon_img)
    if len(scale_range)==2:
        if len(restrict_range) == 2:
            if restrict_range[0] < icon_width or restrict_range[1] < icon_height:
                scale = np.random.uniform(scale_range[0], scale_range[1])
                icon_img = scaleImage(icon_img, scale)


    finalImg, finalBoundRect = place_distorted_sample(
        bkgImg, icon_img, occupancy_bits
    )
    bkgImg = finalImg
    if label_req:
        print("Labelled Icon "+ label +" put on image: ", filenameWithExt)
        details = file_ptr.read() + (
            f"{label.split('.')[0]} "
            + " ".join(
                str(coord) for coord in np.reshape(finalBoundRect, 4)
            )
            + "\n"
        )
        file_ptr.write(details)
    else:
        print("Unlabelled Icon "+ label +" put on image: ", filenameWithExt)
    

def augment_data():
    for backgroundFilePath in background_file_paths:
        for bkgImgPath in glob.glob(os.path.join(backgroundFilePath, "*.png")):
            print("Reading: ", bkgImgPath)
            filenameWithExt = os.path.split(bkgImgPath)[1]
            filename = os.path.splitext(filenameWithExt)[0]
            bkgImg = cv2.imread(bkgImgPath)

            # crop required, [rows, columns]
            # cropping isnt mandatory, related to my data, so an add-on.
            crop_req = True
            if crop_req:
                bkgImg = bkgImg[200:bkgImg.shape[0], 0:bkgImg.shape[1]]

            bgHeight, bgWidth, _ = np.shape(bkgImg)
            
            variants_bkg = [bkgImg]
            # rotating the data if required
            for angle in rotate_arr:
                variants_bkg.append(rotate_image(bkgImg, angle))
            # flipping the data if required
            for code in flip_arr:
                variants_bkg.append(cv2.flip(bkgImg, code))

            variant_index = 1
            for bkgImg in variants_bkg:
                # for avoiding the overlapp of icons
                occupancy_bits = np.zeros((bgHeight, bgWidth), dtype=np.int8)
                count = 0
                outputName = filename
                label_required_icon_count = icon_per_sample*fraction_of_include

                with open(os.path.join(outputfolder+"labels/", str(outputName + "_" + str(variant_index) + ".txt")), "w+") as f:
                    # for True icons, which require labels
                    while count < label_required_icon_count:
                        try:
                            choose_and_place_sample(bkgImg, occupancy_bits, filenameWithExt, f, True)
                            count += 1
                        except AssertionError as e:
                            print(e)
                        
                # for False icons, remaining count, which dont require labels
                while count < icon_per_sample:
                    try:
                        choose_and_place_sample(bkgImg, occupancy_bits, filenameWithExt, f, False)
                        count += 1
                    except AssertionError as e:
                        print(e)

                cv2.imwrite(os.path.join(outputfolder+"images/", str(outputName + "_" + str(variant_index) +".png")), bkgImg)
                variant_index += 1
            
if __name__ == "__main__":
    augment_data()
