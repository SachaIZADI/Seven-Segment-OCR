import numpy as np



class bounding_box:

    """
    The aim of this class is to extract digits from the frame-only preprocessed image.
    As labels only give round number, our objective is to extract digits localised before the comma.
    For that, we try to delimit digits by bounding boxs.
    We tried several approaches (see annexes), but we only keep here the most successful one.
    """

    def __init__(self, image=None, src_file_name=None, dst_file_name=None, return_image=False):
        self.image = image
        self.src_file_name = src_file_name
        self.dst_file_name = dst_file_name
        self.return_image = return_image
        self.box_positions = None


    def get_bd_dummy(self):
        # Dimension de chaque cut

        """1st approach : dummy approach
        Get the bounding box considering that the comma is at 8/13 of the image
        and dividing the area by 4 before the detected comma

        :param ppc_img : the preprocessed image (output of a preprocess fct) ie the exctracted screen + constrats
        :return dist : the distance between each cut (used after in the cut_and_affect_to_folder ) ie the size of the bounding boxes
        plots the image with the computed cuts
        """
        self.box_positions = self.image.shape[1]/4











##### ----- All the things bellow are annex


def detect_comma(image):
    """
    2nd approach :
        Detect the comma with  computer vision-related functions.
        Divide the area by 4 identical regions before the detected comma

        :param image : the preprocessed image
        :return res : the X coordinate of the comma
    """
    #we dilate the image to fill gaps that may intervene during preprocessing
    thresh = cv2.dilate(image, None, iterations=2)
    edged = cv2.Canny(thresh, 50, 200)
    label_image = label(edged)
    res = 500

    #regionprops() detects the rgions of interest in the image
    for region in regionprops(label_image):
        #minr,minc,maxr,maxc are the coordinates of the region
        minr, minc, maxr, maxc = region.bbox
        height = maxr - minr
        length = maxc - minc
        #conditions to detect the comma : must be a square, always located in the same area of the image
        if (abs(length - height) < 5 and minc >= 0.55 * image.shape[1] and maxc <= 0.8 * image.shape[
            1] and minr >= 0.6 * image.shape[0]
                and height > 0.10 * image.shape[0]):
            res = minc
    return (res)




def get_bd_comma(ppc_img):
    """
    :param ppc_img : the preprocessed image (output of a preprocess fct) ie the exctracted screen + constrats
    :return dist : the distance between each cut (used after in the cut_and_affect_to_folder ) ie the size of the bounding boxes
    plots the image with the computed cuts
    """
    coma_place = detect_comma(ppc_img)  # detect comma, function from Priscille
    dist = coma_place / 4
    cuts = np.array(range(5)) * dist
    # plt.subplot(223)
    plt.imshow(ppc_img)
    for i in cuts:
        plt.axvline(x=i, color='b', linestyle='--')
    return (dist)



def get_bd_peaks(ppc_img, coma_detection=False):
    """3rd approach :
     Get the bounding box with AGGREGATION of the binary pixels on the HORIZONTAL AXIS
    (is we sums the pixels over the columns to get a 2D array of size pcc_img.shape[1] )
    From this 2D curve we look for peaks (find_peaks function)
    The founded peaks delimit the bounding boxes

    :param ppc_img : the preprocessed image (output of a preprocess fct) ie the exctracted screen + constrats
    :return peakind3 : the x axis of the cuts + plots the image with the computed cuts

    """

    # Check
    if ppc_img is None:
        return (None)

    # Aggregation on the horizontal axis to get a 2D array
    y = -np.array([sum(j) for j in ppc_img.T])
    x = np.array(range(len(y)))

    # Find peaks on this 2D array
    peakind2 = scipy.signal.find_peaks(y, distance=50, prominence=1, width=15)[0]

    # Adds the first delimiter (first founding box), which does not exists in reality
    if len(peakind2) > 1 & len(y) > 1:
        temp = y[:peakind2[0] - 1]
    else:
        temp = y[0]
    temp = np.where(temp == 0)[0]
    if len(temp) > 1:
        peakind3 = np.append(peakind2, [-1])
    else:
        peakind3 = peakind2

    # Deletes boxes after the coma if coma_detection is set to True
    if coma_detection:
        coma_place = detect_comma(ppc_img)
        peakind3 = peakind3[peakind3 <= coma_place]

    # Plot results
    plt.figure()
    plt.subplot(221)
    plt.plot(x, y)
    plt.plot(x[peakind3], y[peakind3])
    plt.subplot(222)
    plt.imshow(ppc_img)
    for i in peakind3:
        plt.axvline(x=i, color='b', linestyle='--')

    return (peakind3)
