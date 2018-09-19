import numpy as np

import imutils
import cv2
import scipy.spatial as sp
from skimage.measure import label, regionprops


import glob



class frameExtractor:

    def __init__(self, src_file_name, dst_file_name=None, output_shape =(400,100)):
        """
        :param src_file_name:
        :param dst_file_name:
        :param output_shape:
        """
        self.image = cv2.imread(src_file_name)
        self.dst_file_name = dst_file_name
        self.output_shape = output_shape
        self.frame = None



    def distance_from_center(self, rectangle):
        """
        :param rectangle:
        :return:
        """
        center_rc = 0.5*(rectangle[0]+ rectangle[2])
        center_image = 0.5*np.array([self.image.shape[1],self.image.shape[0]])
        distance = np.linalg.norm(center_rc-center_image)
        return distance



    def sort_pts_clockwise(A):
        """
        :param A:
        :return:
        """
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


    def adjust_gamma(image, gamma=1.0):
        """
        :param gamma:
        :return:
        """
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
        return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))


    def preprocessing(self):
        """
        :return:
        """
        self.image = imutils.resize(self.image, height=500)

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        gamma = frameExtractor.adjust_gamma(blurred, gamma=0.7)
        shapeMask = cv2.threshold(gamma, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        label_image = label(shapeMask)

        Cnt = None
        position = [0, 0, 0, 0]

        for region in regionprops(label_image):

            minr, minc, maxr, maxc = region.bbox
            c = np.array([[minc, minr], [minc, maxr], [maxc, minr], [maxc, maxr]])

            if Cnt is None:
                Cnt = c
                position = [minr, minc, maxr, maxc]

            old_dist = self.distance_from_center(Cnt)
            new_dist = self.distance_from_center(c)

            Lx = maxc - minc
            Ly = maxr - minr


            c = frameExtractor.sort_pts_clockwise(c)

            if old_dist>new_dist and Ly<Lx and cv2.contourArea(c)>0.05*(shapeMask.shape[0]*shapeMask.shape[1]):
                displayCnt = c
                position = [minr, minc, maxr, maxc]

        Cnt = Cnt.reshape(4, 2)
        Cnt = frameExtractor.sort_pts_clockwise(Cnt)


        try:

            crop_img = self.image[max(0, position[0] - 30):min(position[2] + 30, self.image.shape[0]),\
                       max(0, position[1] - 30):min(self.image.shape[1], position[3] + 30)]

            crop_blurred = cv2.GaussianBlur(crop_img, (5, 5), 0)
            crop_gamma = frameExtractor.adjust_gamma(crop_blurred, gamma=0.4)
            crop_gray = cv2.cvtColor(crop_gamma, cv2.COLOR_BGR2GRAY)
            crop_thresh = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            cnts = cv2.findContours(crop_thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            Cnt_bis = None

            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                if len(approx) == 4:
                    Cnt_bis = approx
                    break

            if cv2.contourArea(Cnt_bis)<0.5*(crop_img.shape[0]*crop_img.shape[1]):
                raise ValueError("Couldn't find the box, so switching to ad hoc method.")

            Cnt_bis = Cnt_bis.reshape(4, 2)
            Cnt_bis = frameExtractor.sort_pts_clockwise(Cnt_bis)
            src_pts = Cnt_bis.copy()
            src_pts = src_pts.astype(np.float32)

            dst_pts = np.array([[0, 0], [400, 0], [400, 100], [0, 100]], dtype=np.float32)
            dst_pts = dst_pts.astype(np.float32)

            persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(crop_img, persp, (400, 100))


        except:

            src_pts = Cnt.copy()
            src_pts = src_pts.astype(np.float32)

            dst_pts = np.array([[0, 0], [400, 0], [400, 100], [0, 100]], dtype=np.float32)
            dst_pts = dst_pts.astype(np.float32)

            persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(gray, persp, (400, 100))


        self.frame = warped


    def extractAndSaveFrame(self):
        self.preprocessing()
        cv2.imwrite(self.dst_file_name, self.frame)





if __name__ == "__main__":

    fail = [0,0,0,0,0]

    for file in glob.glob('Datasets/HQ_digital/*jpg'):

        try:
            f = frameExtractor(src_file_name=file,
                               dst_file_name='Datasets/HQ_digital_preprocessing/'+str(file).split('/')[-1],
                               output_shape =(400,100))
            f.extractAndSaveFrame()
        except:
            fail[0] += 1


    for file in glob.glob('Datasets/LQ_digital/*jpg'):
        try:
            f = frameExtractor(src_file_name=file,
                               dst_file_name='Datasets/LQ_digital_preprocessing/' + str(file).split('/')[-1],
                               output_shape=(400, 100))
            f.extractAndSaveFrame()
        except:
            fail[1] += 1


    for file in glob.glob('Datasets/MQ_digital/*jpg'):
        try:
            f = frameExtractor(src_file_name=file,
                               dst_file_name='Datasets/MQ_digital_preprocessing/' + str(file).split('/')[-1],
                               output_shape=(400, 100))
            f.extractAndSaveFrame()
        except:
            fail[2] += 1


    for file in glob.glob('Datasets/HQ_analog/*jpg'):
        try:
            f = frameExtractor(src_file_name=file,
                               dst_file_name='Datasets/HQ_analog_preprocessing/' + str(file).split('/')[-1],
                               output_shape=(400, 100))
            f.extractAndSaveFrame()
        except:
            fail[3] += 1


    for file in glob.glob('Datasets/LQ_analog/*jpg'):
        try:
            f = frameExtractor(src_file_name=file,
                               dst_file_name='Datasets/LQ_analog_preprocessing/' + str(file).split('/')[-1],
                               output_shape=(400, 100))
            f.extractAndSaveFrame()
        except:
            fail[4] += 1

            
    print(fail)