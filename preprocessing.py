# import the necessary packages
from imutils.perspective import four_point_transform
import imutils
import cv2
import scipy.spatial as sp


import glob


def distance_from_center(square, image):
    center_sq = 0.5*(square[0]+ square[2])
    center_image = 0.5*np.array([image.shape[1],image.shape[0]])
    distance = np.linalg.norm(center_sq-center_image)
    return(distance)


def sortpts_clockwise(A):
    # Sort A based on Y(col-2) coordinates
    sortedAc2 = A[np.argsort(A[:,1]),:]

    # Get top two and bottom two points
    top2 = sortedAc2[0:2,:]
    bottom2 = sortedAc2[2:,:]

    # Sort top2 points to have the first row as the top-left one
    sortedtop2c1 = top2[np.argsort(top2[:,0]),:]
    top_left = sortedtop2c1[0,:]

    # Use top left point as pivot & calculate sq-euclidean dist against
    # bottom2 points & thus get bottom-right, bottom-left sequentially
    sqdists = sp.distance.cdist(top_left[None], bottom2, 'sqeuclidean')
    rest2 = bottom2[np.argsort(np.max(sqdists,0))[::-1],:]

    # Concatenate all these points for the final output
    return np.concatenate((sortedtop2c1,rest2),axis =0)



def dimension_rectangle(shape):

    rectangle = shape.copy()

    rectangle = rectangle.reshape(4,2)
    rectangle = sortpts_clockwise(rectangle)
    Ly = 0
    Lx = 0

    for i in range(4):
        d = rectangle[i]-rectangle[(i+1)%4]
        if abs(d[0]) > Lx :
            Lx = abs(d[0])
        if abs(d[1]) > Ly :
            Ly = abs(d[1])
    return((Lx,Ly))



def adjust_gamma(image, gamma=1.0):
   invGamma = 1.0 / gamma
   table = np.array([
      ((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)])
   return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))


import numpy as np


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

        if len(approx) == 4:
            if displayCnt is None:
                displayCnt = approx

            old_dist = distance_from_center(displayCnt, shapeMask)
            new_dist = distance_from_center(approx, shapeMask)

            Lx, Ly = dimension_rectangle(displayCnt)

            if old_dist > new_dist and Ly < Lx and cv2.contourArea(approx) > 0.03*(shapeMask.shape[0]*shapeMask.shape[1]) :
                displayCnt = approx

    displayCnt = displayCnt.reshape(4,2)
    displayCnt = sortpts_clockwise(displayCnt)

    src_pts = displayCnt.copy()
    src_pts = src_pts.astype(np.float32)

    dst_pts = np.array([[0,0],[400,0],[400,100],[0,100]], dtype=np.float32)
    dst_pts = dst_pts.astype(np.float32)

    persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(gray, persp, (400, 100))

    #warped = four_point_transform(gray, displayCnt.reshape(4, 2))
    output = four_point_transform(image, displayCnt.reshape(4, 2))

    enlighted = adjust_gamma(warped, gamma=3)


    # faire du padding

    return(warped)



if __name__ == "__main__":

    fail = [0,0,0,0,0]

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


    for file in glob.glob('Datasets/HQ_analog/*jpg'):
        image = cv2.imread(file)
        try:
            preprocessed = preprocessing(image)
            cv2.imwrite('Datasets/HQ_analog_preprocessing/' + str(file).split('/')[-1], preprocessed)
        except:
            fail[3] += 1


    for file in glob.glob('Datasets/LQ_analog/*jpg'):
        image = cv2.imread(file)
        try:
            preprocessed = preprocessing(image)
            cv2.imwrite('Datasets/LQ_analog_preprocessing/' + str(file).split('/')[-1], preprocessed)
        except:
            fail[4] += 1

    print(fail)