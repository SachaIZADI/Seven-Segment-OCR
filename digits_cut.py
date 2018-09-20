import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt




class cutDigits:

    def __init__(self, image=None, src_file_name=None, dst_folder_name='Datasets_digits', last_digit=4, labels=None):
        """
        The aim of this class is to extract digits from the frame-only preprocessed image.
        We to delimit digits by bounding boxes.
        We tried several approaches, but we present here the most successful one, a "dummy" yet efficient approach.
        :param image: RGB image (numpy array NxMx3) of a SLICED SCREEN. If image is None, the image will be extracted from src_filename
        :param src_file_name: filename of a SLICED SCREEN to load the source image (e.g. HQ_digital_preprocessing/0a07d2cff5beb0580bca191427e8cd6e1a0eb678.jpg)
        :param dst_folder_name: home FOLDERname where to save the extracted digits.
        :param last_digit: int, the number of digits you want to extract starting from the left (0 = no digits / 4 = all four digits).
        :param labels: list, list of labels corresponding to the image, e.g. if th image shows 123.45, the labels will be ['x',1,2,3].
        """
        if image is None :
            self.image = cv2.imread(src_file_name)
        else:
            self.image = image
        self.src_file_name = src_file_name
        self.dst_folder_name = dst_folder_name
        self.last_digit=last_digit
        self.labels = labels

        self.box_size = None
        self.boxes = []



    def get_bounding_box_dummy(self):
        """
        Use this method to get bounding boxes and extract numbers by dividing the area in 4 equal parts ("dummy" yet efficient approach).
        """

        self.boxes = []
        self.box_size = self.image.shape[1]/4

        for i in range(self.last_digit):
            inf = i * self.box_size
            sup = (i+1) * self.box_size
            self.boxes += [self.image[:, int(inf):int(sup)]]


    def save_to_folder(self) :
        """
        Use this method to save the extracted bounding boxes.
        """
        if self.dst_folder_name is None :
            return

        for i in range(len(self.boxes)):
            if self.labels :
                box = self.boxes[i]
                label = self.labels[i]
                src_file_name = self.src_file_name.split('/')[-1].split('.')[0]
                dst_file_name = 'Datasets_digits/%s/%s_%s.jpg' % (label, src_file_name, str(i))
                cv2.imwrite(dst_file_name, box)

            else :
                box = self.boxes[i]
                src_file_name = self.src_file_name.split('/')[-1].split('.')[0]
                dst_file_name = 'Datasets_digits/%s/%s_%s.jpg' % ('missing_label', src_file_name, str(i))
                cv2.imwrite(dst_file_name, box)


# --------------------- End of the class -----------------------------------





"""
A main function to cut the digits on all images.
"""

if __name__ == "main":

    A = 'Datasets_digits/%s/%s_%s'%('a','b','c')


    # ---- INITIALISATION ----

    # eleven folder 'Datasets'
    raw_dir = "Datasets_raw/"

    # may be HQ_digital, MQ_digital or LQ_digital
    cat_dir =  "LQ_digital"

    # path to Sacha's output, with the extracted screen
    preprocessed_dir = "Datasets_preprocessed/"+ cat_dir +"_preprocessing/"

    all_images = os.listdir(raw_dir + cat_dir)
    all_images_preprocessed = os.listdir(preprocessed_dir)

    # output path to save individual digits in
    # of the form "Datasets_digits/" and contains the '0', '1', ... 'X' folders
    digits_path = "Datasets_digits/"

    # Csv file with the image name, 'cadran_1', 'cadran_2', 'cadran_3', 'cadran_4' columns containing the digits' labels before the comma
    labels_path = "Datasets_labels/"+cat_dir+".csv"

    # convert file into dataframe
    labels_df   = csv_labels_to_df(labels_path)

    # ---- LOOP ----

    for ind in all_images_preprocessed:
        if ind != ".DS_Store":
            print(ind)
            image = cv2.imread(preprocessed_dir + ind)                  # get the extracted screen from img
            #warped = extract_screen(image)
            preprocessed_img = preprocess2(image)                       # preprocess the img
            dist = get_bd_dummy(preprocessed)                           # get bounding boxes' size

            # get the labels of the digits before the comma
            labels = labels_df[labels_df['image'] == ind][['cadran_1', 'cadran_2', 'cadran_3', 'cadran_4']].values

            # get bounding boxes and save truncated images in the folder corresponding to its label
            cut_and_affect_to_folder(preprocessed_img, dist, labels[0], \
                                     digits_path, ind, last_digit= 2)
            plt.close()



# DECOMMENT IF IMAGES NOT PREPROCESSED ALREADY
'''for ind in all_images:
    print(ind)
    image = cv2.imread(input_dir + ind)
    warped = extract_screen(image)
    preprocessed = preprocess_short(warped)
    dist = get_bd_short_wo_comma(preprocessed)
    preprocessed_img = warped
    labels = labels_df[labels_df['image'] == ind][['cadran_1', 'cadran_2', 'cadran_3', 'cadran_4']].values
    affect_to_folder(preprocessed_img, dist, labels[0], digits_path, ind)

    #plt.savefig(output_dir + "bd_plot_" + ind)
    plt.close()'''