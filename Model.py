import numpy as np
import pandas as pd
import os
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers.core import Dropout, Activation
from keras.layers import BatchNormalization
import tensorflow as tf
from keras import regularizers
import keras.backend
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import TensorBoard,EarlyStopping
from Datasets import Dataset_Multi, Dataset_Single
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class Model(object):
    
    def __init__(self):
        
        self.data_init()
        self.model_init()
    
    def data_init(self):
        pass
    
    def model_init(self):
        pass
    
    def train_predict(self):
        pass

class Model_Multi(Model):
    
    def __init__(self):
        Model.__init__(self)
 
    def data_init(self):
        self.dataset = Dataset_Multi()
        self.data = self.dataset.frame_data
        self.X =  self.data.iloc[:,1]
        self.y = self.data.iloc[:,2:]
        
        self.ids_train, self.ids_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.25, random_state=1)        
        self.y_train_vect = [self.y_train["cadran_1"], self.y_train["cadran_2"], self.y_train["cadran_3"], self.y_train["cadran_4"]]
        self.y_val_vect =  [self.y_val["cadran_1"], self.y_val["cadran_2"], self.y_val["cadran_3"], self.y_val["cadran_4"]]
        
        self.X_train = self.dataset.convert_to_arrays(self.ids_train)
        self.X_val = self.dataset.convert_to_arrays(self.ids_val)
              
    def model_init(self):

        model_input = Input((100,246,1))

        x = Conv2D(32, (3, 3), padding='same', name='conv2d_hidden_1', kernel_regularizer=regularizers.l2(0.01))(model_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(3, 3),name='maxpool_2d_hidden_1')(x)
        x = Dropout(0.30)(x)

        x = Conv2D(64, (3, 3), padding='same', name='conv2d_hidden_2', kernel_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(3, 3),name='maxpool_2d_hidden_2')(x)
        x = Dropout(0.30)(x)

        x = Conv2D(128, (3, 3), padding='same', name='conv2d_hidden_3', kernel_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(3, 3),name='maxpool_2d_hidden_3')(x)
        x = Dropout(0.30)(x)

        x = Flatten()(x)

        x = Dense(256, activation ='relu', kernel_regularizer=regularizers.l2(0.01))(x)

        digit1 = (Dense(output_dim =11,activation = 'softmax', name='digit_1'))(x)
        digit2 = (Dense(output_dim =11,activation = 'softmax', name='digit_2'))(x)
        digit3 = (Dense(output_dim =11,activation = 'softmax', name='digit_3'))(x)
        digit4 = (Dense(output_dim =11,activation = 'softmax', name='digit_4'))(x)

        outputs = [digit1, digit2, digit3, digit4]

        self.model = keras.models.Model(input = model_input , output = outputs)
        self.model._make_predict_function()
        
    def train(self, lr = 1e-3, epochs=50):
        optimizer = Adam(lr=lr, decay=lr/10)
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer= optimizer, metrics = ['accuracy'])
        keras.backend.get_session().run(tf.initialize_all_variables())
        self.history = self.model.fit(self.X_train, self.y_train_vect, batch_size= 50, nb_epoch=epochs, verbose=1, validation_data=(self.X_val, self.y_val_vect))
        
        
    def plot_loss(self):
        
        for i in range(1,5):
            plt.figure(figsize=[8,6])
            plt.plot(self.history.history['digit_%i_loss' %i],'r',linewidth=0.5)
            plt.plot(self.history.history['val_digit_%i_loss' %i],'b',linewidth=0.5)
            plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
            plt.xlabel('Epochs ',fontsize=16)
            plt.ylabel('Loss',fontsize=16)
            plt.title('Loss Curves Digit %i' %i,fontsize=16)
            plt.show()
        
        
      

    def plot_acc(self):
        
        for i in range(1,5):
            plt.figure(figsize=[8,6])
            plt.plot(self.history.history['digit_%i_acc' %i],'r',linewidth=0.5)
            plt.plot(self.history.history['val_digit_%i_acc' %i],'b',linewidth=0.5)
            plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
            plt.xlabel('Epochs ',fontsize=16)
            plt.ylabel('Accuracy',fontsize=16)
            plt.title('Accuracy Curves Digit %i' %i,fontsize=16)
            plt.show()
        

    def predict(self):
        self.y_pred = self.model.predict(self.X_val)
        correct_preds = 0
        
        for i in range(self.X_val.shape[0]):
            pred_list_i = [np.argmax(pred[i]) for pred in self.y_pred]
            val_list_i  = self.y_val.values[i].astype('int')
            if np.array_equal(val_list_i, pred_list_i):
                correct_preds = correct_preds + 1
            print('exact accuracy', correct_preds / self.X_val.shape[0])
            
        mse = 0 
        diff = []
        for i in range(self.X_val.shape[0]):
                pred_list_i = [np.argmax(pred[i]) for pred in self.y_pred]
                pred_number = 1000* pred_list_i[0] + 100* pred_list_i[1] + 10 * pred_list_i[2] + 1* pred_list_i[3]
                val_list_i  = self.y_val.values[i].astype('int')
                val_number = 1000* val_list_i[0] + 100*  val_list_i[1] + 10 *  val_list_i[2] + 1*  val_list_i[3]
                diff.append(val_number - pred_number)
        print('difference label vs. prediction', diff)

    
    def train_predict(self):
        
        self.train()
        self.plot_loss()
        self.plot_acc()
        self.predict()
        
class Model_Single(Model):
    
    
    def __init__(self):
            Model.__init__(self)

    def data_init(self):
        self.dataset = Dataset_Single()

        self.data = self.dataset.digits_data         
        self.X =  self.data.iloc[:,0]
        self.y = self.data.iloc[:,1]

        self.ids_train, self.ids_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.25, random_state=1)
        self.X_train = self.dataset.convert_to_arrays(self.ids_train)
        self.X_val = self.dataset.convert_to_arrays(self.ids_val)

    def model_init(self):


        model_input = Input((100, 256, 1))
        x = Conv2D(32, (3, 3), padding='same', name='conv2d_hidden_1', kernel_regularizer=regularizers.l2(0.01))(model_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(3, 3),name='maxpool_2d_hidden_1')(x)
        x = Dropout(0.30)(x)

        x = Conv2D(63, (3, 3), padding='same', name='conv2d_hidden_2', kernel_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(3, 3),name='maxpool_2d_hidden_2')(x)
        x = Dropout(0.30)(x)

        x = Conv2D(128, (3, 3), padding='same', name='conv2d_hidden_3', kernel_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(3, 3),name='maxpool_2d_hidden_3')(x)
        x = Dropout(0.30)(x)

        x = Flatten()(x)

        x = Dense(1024, activation ='relu', kernel_regularizer=regularizers.l2(0.01))(x)

        output = Dense(output_dim =11,activation = 'softmax', name='output')(x)

        self.model = keras.models.Model(input = model_input , output = output)
        self.model._make_predict_function() 

    def train(self, lr = 1e-3, epochs=5):
        optimizer = Adam(lr=lr, decay=lr/10)
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer= optimizer, metrics = ['accuracy'])
        keras.backend.get_session().run(tf.initialize_all_variables())
        self.history = self.model.fit(self.X_train, self.y_train, batch_size= 32, nb_epoch=30, verbose=1, validation_data=(self.X_val, self.y_val))


    def plot_acc(self):
        plt.figure(figsize=[8,6])
        plt.plot(self.history.history['acc'],'r',linewidth=0.5)
        plt.plot(self.history.history['val_acc'],'b',linewidth=0.5)
        plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Accuracy',fontsize=16)
        plt.title('Accuracy Curves Digit',fontsize=16)
        plt.show()

    def plot_loss(self):        
        plt.figure(figsize=[8,6])
        plt.plot(self.history.history['loss'],'r',linewidth=0.5)
        plt.plot(self.history.history['val_loss'],'b',linewidth=0.5)
        plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.title('Loss Curves Digit',fontsize=16)
        plt.show()

    def predict(self):
        
        self.y_pred = self.model.predict(self.X_val)
        
        ids = []
        pred_list = []
        val_list = []

        for i in range(self.X_val.shape[0]):
            self.val_id = self.ids_val.values[i]
            ids.append(str(self.val_id.split('/')[2].split('-')[0][:-1]))
            pred_list_i = np.argmax(self.y_pred[i]).astype('int')
            pred_list.append(pred_list_i)
            val_list_i  = self.y_val.values[i].astype('int')
            val_list.append(val_list_i) 

        q = []

        for i in np.unique(ids):
            q.append([i, np.where(np.isin(ids,i))[0]])

        correct_count = 0 
        for i in range(len(q)):
            v = []
            p = []
            for j in range(len((q[i][1]))):
                idx = (q[i][1][j])
                val_list_i = val_list[idx]
                pred_list_i = pred_list[idx]
                v.append(val_list_i)
                p.append(pred_list_i)
            if np.array_equal(p, v):
                correct_count = correct_count + 1
        print('real_acc', correct_count /self.X_val.shape[0])


    def train_predict(self):

        self.train()
        self.plot_loss()
        self.plot_acc()
        self.predict()


