import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplotas plt 
from PIL import Image

class Dataset:
    
    def __init__(self, config):
            self.config= config
            self.csv_directory = self.config.csv_directory
            self.data = self.full_data()
                        
    def full_data(self):
        suffix = ".csv"
        csv_directory = self.csv_directory
        csv_files = [i for i in os.listdir(csv_directory) if i.endswith( suffix )]
        full_data = []
        for i in range(len(csv_files)):
            data = pd.read_csv(csv_directory+'/'+csv_files[i], sep=';', index_col = 0)
            full_data.append(data)
            
        full_data = pd.concat(full_data, axis=0)
        full_data = full_data.replace("X", 10)
        return full_data
        
class Dataset_Multi(Dataset):
    
    def __init__(self, config):
        Dataset.__init__(self, config)
        self.frame_directory = self.config.frame_directory
        self.frame_data = self.data[self.data["image"].isin(os.listdir(self.frame_directory))]
        
    def convert_to_arrays(self,samples):
        X = []
        for sample in samples:
            ID =  'Dataset_frames' + "%s" % (sample)
            img = Image.open(ID)
            img = np.array(img)
            img = img.reshape((img.shape[0],img.shape[1],1))
            X.append(img)
        X = np.asarray(X)
        return X
        
        
class Dataset_Single(Dataset):
    
    def __init__(self, config):
        Dataset.__init__(self, config)
        self.digits_directory = self.config.digits_directory
        self.digits_data = self.digits_data()
    
    def data(self):
        ids = []
        labels = []
        for i in range(self.config.n_classes):
            n_directory = self.digits_directory + '%i' %i
            for j in os.listdir(n_directory):
                ids.append(directory+j)
                labels.append(i)
        digits_data = pd.DataFrame(list(zip(ids,labels)))
        
        return digits_data 
                
    def convert_to_arrays(self,samples):
        X = []
        for sample in samples:
            img = Image.open(sample)
            img = np.array(img)
            img.resize((self.config.digit_height,self.config.digit_height))
            img = img.reshape((img.shape[0],img.shape[1],1))
            X.append(img)
        X = np.asarray(X)
        return X