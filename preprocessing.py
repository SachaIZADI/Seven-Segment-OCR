# import the necessary packages
from imutils.perspective import four_point_transform
import imutils
import cv2

import numpy as np


import glob




def adjust_gamma(image, gamma=1.0):
   invGamma = 1.0 / gamma
   table = np.array([
      ((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)])
   return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))


def preprocessing(image):
    image = imutils.resize(image, height=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)


    shapeMask = cv2.threshold(blurred, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


    cnts = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None


    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)


        if len(approx) == 4:
            displayCnt = approx
            break

    warped = four_point_transform(gray, displayCnt.reshape(4, 2))
    #output = four_point_transform(image, displayCnt.reshape(4, 2))

    enlighted = adjust_gamma(warped, gamma=3)


    # faire du padding

    return(enlighted)



if __name__ == "__main__":

    fail = [0,0,0]

    for file in glob.glob('Datasets/HQ_digital/*jpg'):
        image = cv2.imread(file)
        try :
            preprocessed = preprocessing(image)
            cv2.imwrite('Datasets/HQ_digital_preprocessing/'+str(file).split('/')[-1], preprocessed)
        except:
            fail[0]+=1

    for file in glob.glob('Datasets/LQ_digital/*jpg'):
        image = cv2.imread(file)
        try:
            preprocessed = preprocessing(image)
            cv2.imwrite('Datasets/LQ_digital_preprocessing/' + str(file).split('/')[-1], preprocessed)
        except:
            fail[1] += 1

    for file in glob.glob('Datasets/MQ_digital/*jpg'):
        image = cv2.imread(file)
        try:
            preprocessed = preprocessing(image)
            cv2.imwrite('Datasets/MQ_digital_preprocessing/' + str(file).split('/')[-1], preprocessed)
        except:
            fail[2] += 1


    print(fail)