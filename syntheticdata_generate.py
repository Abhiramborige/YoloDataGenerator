import numpy as np
import cv2 as cv2
import os
import glob
import yaml

# https://github.com/srp-31/Data-Augmentation-for-Object-Detection-YOLO-
with open("./config.yaml") as fp:
    config_params = yaml.load(fp, Loader=yaml.FullLoader)

backgroundFilePaths = config_params.get("BACKGROUND_FILE_PATHS")
iconFilePath = config_params.get("SAMPLE_FILES_PATH")
outputfolder = config_params.get("OUTPUT_PATH")
icon_per_sample = config_params.get("OUTPUT_PER_SAMPLE")
if not (os.path.isdir(outputfolder+"images/")):
    os.makedirs(outputfolder+"images/")
if not (os.path.isdir(outputfolder+"labels/")):
    os.makedirs(outputfolder+"labels/")

iconImgList =[]
iconImgNameList = []
for name in glob.glob(os.path.join(iconFilePath, "*.png")):
    iconImgList.append(cv2.imread(name, cv2.IMREAD_UNCHANGED))
    iconImgNameList.append(os.path.basename(name))

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
            return False, bkgImg, 0
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
        return True, bkgImg, boundRectFin
    else:
        return False, bkgImg, 0
    
def scaleImage(image, scale):
    return cv2.resize(image, None, fx=scale, fy=scale)

def augment_data():
    for backgroundFilePath in backgroundFilePaths:
        for bkgImgPath in glob.glob(os.path.join(backgroundFilePath, "*.png")):
            filenameWithExt = os.path.split(bkgImgPath)[1]
            filename = os.path.splitext(filenameWithExt)[0]
            bkgImg = cv2.imread(bkgImgPath)
            bgHeight, bgWidth, _ = np.shape(bkgImg)
            # for avoiding the overlapp of icons
            occupancy_bits = np.zeros((bgHeight, bgWidth), dtype=np.int8)

            count = 0
            outputName = filename
            with open(os.path.join(outputfolder+"labels/", str(outputName + ".txt")), "w+") as f:
                while count < icon_per_sample:
                    # randomly choose the icon to put on image
                    randnum = np.random.randint(0, len(iconImgList))
                    icon_img = iconImgList[randnum]
                    label = iconImgNameList[randnum]
                    scale = np.random.uniform(1, 1.4)
                    icon_img = scaleImage(icon_img, scale)

                    flag, finalImg, finalBoundRect = place_distorted_sample(
                        bkgImg, icon_img, occupancy_bits
                    )
                    bkgImg = finalImg
                    if flag:
                        count = count + 1
                        print("Icon "+ label +" put on image: ", filenameWithExt)
                        details = f.read() + (
                            f"{label.split('.')[0]} "
                            + " ".join(
                                str(coord) for coord in np.reshape(finalBoundRect, 4)
                            )
                            + "\n"
                        )
                        f.write(details)
                    else:
                        print("Not able to put icon on image")
            cv2.imwrite(os.path.join(outputfolder+"images/", str(outputName + ".png")), bkgImg)

            
if __name__ == "__main__":
    augment_data()
