import numpy as np
import cv2 as cv2
import os
import glob
import yaml
# https://github.com/srp-31/Data-Augmentation-for-Object-Detection-YOLO-

with open("./config.yaml" ,"r") as fp:
    config_params = yaml.load(fp, Loader=yaml.FullLoader)

backgroundFilePaths = config_params.get("BACKGROUND_FILE_PATHS")
iconFilePath = config_params.get("SAMPLE_FILES_PATH")
outputfolder = config_params.get("OUTPUT_PATH")
flip_req = config_params.get("BACKGROUND_FLIP_REQ")
# icons which require labelling
include_files = config_params.get("INCLUDE_FILE_NAMES")
fraction = config_params.get("FRACTION_OF_INCLUDE_FILE")
icon_per_sample = config_params.get("OUTPUT_PER_SAMPLE")
label_required_icon_count = fraction*icon_per_sample
label_not_required_icon_count = icon_per_sample - label_required_icon_count

if not (os.path.isdir(outputfolder+"images/")):
    os.makedirs(outputfolder+"images/")
if not (os.path.isdir(outputfolder+"labels/")):
    os.makedirs(outputfolder+"labels/")

iconImgList =[]
iconImgNameList = []
for name in glob.glob(os.path.join(iconFilePath, "*.png")):
    iconImgList.append(cv2.imread(name, cv2.IMREAD_UNCHANGED))
    iconImgNameList.append(os.path.basename(name))
# icons to be put on image which doesnt require labelling
exclude_files = list(set(iconImgNameList) - set(include_files))
print(iconImgNameList)
print(include_files)
print(exclude_files)


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
                         filenameWithExt, file_ptr, label_req = True):
    # randomly choose the icon from include_files/exclude_files to put on image
    if label_req:
        randnum = np.random.randint(0, len(include_files))
        index_req = iconImgNameList.index(include_files[randnum])
    else:
        randnum = np.random.randint(0, len(exclude_files))
        index_req = iconImgNameList.index(exclude_files[randnum])

    icon_img = iconImgList[index_req]
    label = iconImgNameList[index_req]
    scale = np.random.uniform(1, 1.5)
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
    
def scaleImage(image, scale):
    return cv2.resize(image, None, fx=scale, fy=scale)

def augment_data():
    for backgroundFilePath in backgroundFilePaths:
        for bkgImgPath in glob.glob(os.path.join(backgroundFilePath, "*.png")):
            filenameWithExt = os.path.split(bkgImgPath)[1]
            filename = os.path.splitext(filenameWithExt)[0]
            bkgImg = cv2.imread(bkgImgPath)
            bgHeight, bgWidth, _ = np.shape(bkgImg)

            # TODO flipping background for more synthetic data
            #variants_bkg = [bkgImg, cv2.flip(bkgImg, 0), cv2.flip(bkgImg ,1), cv2(bkgImg, -1)]
            # for avoiding the overlapp of icons
            occupancy_bits = np.zeros((bgHeight, bgWidth), dtype=np.int8)
            count = 0
            outputName = filename

            with open(os.path.join(outputfolder+"labels/", str(outputName + ".txt")), "w+") as f:
                # for True icons, which require labels
                while count < label_required_icon_count:
                    try:
                        choose_and_place_sample(bkgImg, occupancy_bits, filenameWithExt, f, True)
                        count += 1
                    except AssertionError as e:
                        print(e)
                    
            # for False icons, which dont require labels
            while count < icon_per_sample:
                try:
                    choose_and_place_sample(bkgImg, occupancy_bits, filenameWithExt, f, False)
                    count += 1
                except AssertionError as e:
                    print(e)

            cv2.imwrite(os.path.join(outputfolder+"images/", str(outputName + ".png")), bkgImg)
            
if __name__ == "__main__":
    augment_data()
