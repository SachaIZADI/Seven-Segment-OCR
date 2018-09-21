import shutil
import os
import numpy as np
import imutils
import cv2
import scipy.spatial as sp
from skimage.measure import label, regionprops
import glob
from utils.homofilt import HomomorphicFilter

import skimage.filters as ft



class frameExtractor:

    def __init__(self, image=None, src_file_name=None, dst_file_name=None, return_image=False, output_shape =(400,100)):
        """
        Use this class to extract the frame/LCD screen from the image. This is our step 1 for image preprocessing.
        The final frame is extracted in grayscale.
        Note that it works for the "digital" case and can be used for the "analog" case, but it is more efficient on the "digital" case.
        :param image: RGB image (numpy array NxMx3) with a screen to extract. If image is None, the image will be extracted from src_filename
        :param src_file_name: filename to load the source image where the screen needs to be extracted (e.g. HQ_digital/0a07d2cff5beb0580bca191427e8cd6e1a0eb678.jpg)
        :param dst_file_name: filename to save the preprocessed image (e.g. HQ_digital_frame/0a07d2cff5beb0580bca191427e8cd6e1a0eb678.jpg
        :param return_image: a boolean, if True extractAndSave returns an image (np. array) / if False it just saves the image.
        :param output_shape: shape (in pxl) of the output image.
        """
        if image is None :
            self.image = cv2.imread(src_file_name)
        else :
            self.image = image
        self.dst_file_name = dst_file_name
        self.return_image = return_image
        self.output_shape = output_shape
        self.raw_frame = None
        self.frame = None
        self.sliced_frame = None


    def distance_from_center(self, rectangle):
        """
        Use this function to measure how far a rectangle is from the center of an image.
        Most of the time the frame is approx. in the middle of the picture.
        Note that the code works for shapes that are approx. rectangles.
        :param rectangle: a 4x2 array with the coordinates of each corner of the rectangle.
        :return: the distance (a float) between the center of the rectangle and the center of the picture.
        """
        center_rc = 0.5*(rectangle[0]+ rectangle[2])
        center_image = 0.5*np.array([self.image.shape[1],self.image.shape[0]])
        distance = np.linalg.norm(center_rc-center_image)
        return distance



    def sort_pts_clockwise(A):
        """
        Use this function to sort in clockwise order points in R^2.
        Credit: https://stackoverflow.com/questions/30088697/4-1-2-numpy-array-sort-clockwise
        :param A: a Nx2 array with the 2D coordinates of the points to sort.
        :return: a Nx2 array with the points sorted in clockwise order starting with the top-left point.
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
        Use this function to adjust illumination in an image.
        Credit: https://stackoverflow.com/questions/33322488/how-to-change-image-illumination-in-opencv-python
        :param image: A grayscale image (NxM int array in [0, 255]
        :param gamma: A positive float. If gamma<1 the image is darken / if gamma>1 the image is enlighten / if gamma=1 nothing happens.
        :return: the enlighten/darken version of image
        """
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
        return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))


    def frameDetection(self):
        """
        The core method of the class. Use it to extract the frame in the image.
        The extracted frame is in grayscale.
        The followed steps are :
            1. grayscale + smoothering + gamma to make the frame darker + binary threshold (rational = the frame is one of the darkest part in the picture).
            2. extract regions of "interest".
            3. heuristic to find a region of interest that is large enough, in the center of the picture and where length along x-axis > length along y-axis.
            4. make a perspective transform to crop the image and deal with perspective deformations.
        """
        self.image = imutils.resize(self.image, height=500)

        # Step 1: grayscale + smoothering + gamma to make the frame darker + binary threshold
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        gamma = frameExtractor.adjust_gamma(blurred, gamma=0.7)
        shapeMask = cv2.threshold(gamma, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Step 2: extract regions of "interest".
        label_image = label(shapeMask)

        Cnt = None
        position = [0, 0, 0, 0]

        for region in regionprops(label_image):
            # Step 3: heuristic to find a region large enough, in the center & with length along x-axis > length along y-axis.
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


        # Step 4: Make a perspective transform to crop the image and deal with perspective deformations.
        try:
            # Crop the image around the region of interest (but keep a bit of distance with a 30px padding).
            # Darken + Binary threshold + rectangle detection.
            # If this technique fails, raise an error and use basic methods (except part).

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
            # More basic techniques that give +/- acceptable results when the first technique fails.
            src_pts = Cnt.copy()
            src_pts = src_pts.astype(np.float32)

            dst_pts = np.array([[0, 0], [400, 0], [400, 100], [0, 100]], dtype=np.float32)
            dst_pts = dst_pts.astype(np.float32)

            persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(gray, persp, (400, 100))

        # Frame is extracted from the initial image in grayscale (not other processing done on the image).
        self.raw_frame = warped


    # TODO : check why they fail
    """
    http://www.amphident.de/en/blog/preprocessing-for-automatic-pattern-identification-in-wildlife-removing-glare.html
    http://people.csail.mit.edu/yichangshih/mywebsite/reflection.pdf
    http://news.mit.edu/2015/algorithm-removes-reflections-photos-0511
    """
    def preprocessFrame(self):
        """
        Final preprocessing that outputs a clean image 'cleaned_img' with more contrasts
        """
        try :
            gray = cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2GRAY)
        except :
            gray = self.raw_frame
        thresh = cv2.equalizeHist(gray)
        thresh = cv2.threshold(thresh, 45, 255, cv2.THRESH_BINARY_INV)[1]
        cleaned_img = cv2.dilate(thresh, None, iterations=1)
        self.frame = cleaned_img


    def sliceFrame(self):
        """
        Use this method to slice the frame and only keep the integer part (e.g. 123.45 becomes 123).
        Heuristic: comma is approx. at 8/13 of the image.
        :return:
        """
        stop_at = int(np.floor(self.output_shape[0]*8/13))
        self.sliced_frame = np.array(self.frame)[:,:stop_at]


    def extractAndSaveFrame(self):
        """
        Use this method to
                1. detect and select the frame/screen.
                2. preprocessing to only keep numbers (and remove noise).
                3. slice the frame to only keep integer part.
                4. save the sliced frame in dst_file_name.
        :return: the extracted frame (np.array) if it was specified when instantiating the class.
        """
        self.frameDetection()
        self.preprocessFrame()
        self.sliceFrame()
        cv2.imwrite(self.dst_file_name, self.sliced_frame)
        if self.return_image:
            return self.sliced_frame
        else:
            return



# --------------------- End of the class -----------------------------------



"""
A main function to preprocess all the images.
"""

if __name__ == "__main__":

    if os.path.exists('Datasets_frames/'):
        shutil.rmtree('Datasets_frames/')
        os.makedirs('Datasets_frames/')
    else:
        os.makedirs('Datasets_frames/')

    fail = [0, 0, 0]

    for file in glob.glob('Datasets/HQ_digital/*jpg'):

        try:
            f = frameExtractor(image=None,
                               src_file_name=file,
                               dst_file_name='Datasets_frames/' + str(file).split('/')[-1],
                               return_image=False,
                               output_shape=(400, 100))
            f.extractAndSaveFrame()
        except:
            fail[0] += 1

    for file in glob.glob('Datasets/LQ_digital/*jpg'):
        try:
            f = frameExtractor(image=None,
                               src_file_name=file,
                               dst_file_name='Datasets_frames/' + str(file).split('/')[-1],
                               return_image=False,
                               output_shape=(400, 100))
            f.extractAndSaveFrame()
        except:
            fail[1] += 1

    for file in glob.glob('Datasets/MQ_digital/*jpg'):
        try:
            f = frameExtractor(image=None,
                               src_file_name=file,
                               dst_file_name='Datasets_frames/' + str(file).split('/')[-1],
                               return_image=False,
                               output_shape=(400, 100))
            f.extractAndSaveFrame()
        except:
            fail[2] += 1

    print(fail)