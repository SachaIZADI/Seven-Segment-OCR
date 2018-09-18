# import the necessary packages
from imutils.perspective import four_point_transform
import imutils
import cv2

import numpy as np


import glob


def distance_from_center(square, image):
    center_sq = 0.5*(square[0]+ square[3])
    center_image = 0.5*np.array(image.shape[:2])
    distance = np.linalg.norm(center_sq-center_image)
    return(distance)



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
    gamma = adjust_gamma(blurred, gamma=0.7)

    shapeMask = cv2.threshold(gamma, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


    cnts = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None


    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        #checker que approx est bien centré et de la bonne taille

        if len(approx) == 4:
            if displayCnt is None:
                displayCnt = approx

            old_dist = distance_from_center(displayCnt, shapeMask)
            newdist = distance_from_center(approx, shapeMask)

            if old_dist > newdist and cv2.contourArea(approx)>0.1*(shapeMask.shape[0]*shapeMask.shape[1]) :
                displayCnt = approx



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
        except Exception as e :
            #print(e)
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