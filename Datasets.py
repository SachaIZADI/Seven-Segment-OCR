import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image

class Dataset:
    
    def __init__(self):
            self.data = self.full_data()
                        
    def full_data(self):
        suffix = ".csv"
        csv_directory = 'Datasets/'
        csv_files = [i for i in os.listdir(csv_directory) if i.endswith( suffix )]
        full_data = []
        for i in range(len(csv_files)):
            data = pd.read_csv(csv_directory +csv_files[i], sep=';', index_col = 0)
            full_data.append(data)
            
        full_data = pd.concat(full_data, axis=0)
        full_data = full_data.replace("X", 10)
        return full_data
        
class Dataset_Multi(Dataset):
    
    def __init__(self):
        Dataset.__init__(self)
        self.frame_directory = 'Datasets_frames/'
        self.frame_data = self.data[self.data["image"].isin(os.listdir(self.frame_directory))]
        
    def convert_to_arrays(self,samples):
        X = []
        for sample in samples:
            ID =  'Datasets_frames/' + "%s" % (sample)
            img = Image.open(ID)
            img = np.array(img)
            img = img.reshape((img.shape[0],img.shape[1],1))
            X.append(img)
        X = np.asarray(X)
        return X
        
        
class Dataset_Single(Dataset):
    
    def __init__(self):
        Dataset.__init__(self)
        self.digits_directory = 'Datasets_digits/'
        self.digits_data = self.digits_data()

    
    def digits_data(self):
        ids = []
        labels = []
        for i in range(11):
            directory = self.digits_directory + '%i/' %i
            for j in os.listdir(directory):
                ids.append(directory+j)
                labels.append(i)
        digits_data = pd.DataFrame(list(zip(ids,labels)))
        
        return digits_data 
                
    def convert_to_arrays(self,samples):
        X = []
        for sample in samples:
            img = Image.open(sample)
            img = np.array(img)
            img.resize((100,256))
            img = img.reshape((img.shape[0],img.shape[1],1))
            X.append(img)
        X = np.asarray(X)
        return X