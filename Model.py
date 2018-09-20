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

from Datasets import Dataset

class Model(object):
    
    def __init__(self, config):
        
        self.config = config
        self.data_init()
        self.model_init()
    
    def data_init(self):
        pass
    
    def model_init(self):
        pass
    
    def train_predict(self):
        pass

class Model_Multi(Model):
    
    def __init__(self, config):
        Model.__init__(self, config)
 
    def data_init(self):
        self.dataset = Dataset_Multi(self.config)
        self.data = self.dataset.frame_data
        self.X =  self.data.iloc[:,1]
        self.y = self.data.iloc[:,2:]
        
        self.ids_train, self.ids_val, self.y_train, selfy_val = train_test_split(X, y, test_size=0.25, random_state=1)        
        self.y_train_vect = [self.y_train["cadran_1"], self.y_train["cadran_2"], self.y_train["cadran_3"], self.y_train["cadran_4"]]
        self.y_val_vect =  [self.y_val["cadran_1"], self.y_val["cadran_2"], self.y_val["cadran_3"], self.y_val["cadran_4"]]
        
        self.X_train = self.dataset.convert_to_arrays(ids_train)
        self.X_val = self.dataset.convert_to_arrays(ids_val)
              
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

        self.model = Model(input = model_input , output = outputs)
        self.model._make_predict_function()
        
    def train(self, lr = 1e-3, epochs=100):
        optimizer = Adam(lr=lr, decay=lr/10)
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer= optimizer, metrics = ['accuracy'])
        keras.backend.get_session().run(tf.initialize_all_variables())
        history = self.model.fit(self.X_train, self.y_train_vect, batch_size= 50, nb_epoch=100, verbose=1, validation_data=(self.X_val, self.y_val_vect))
        
        
    def plot_loss(self):
        
        for i in range(1,5):
            plt.figure(figsize=[8,6])
            plt.plot(history.history['digit_%i_acc' %i],'r',linewidth=0.5)
            plt.plot(history.history['val_digit_%i_acc' %i],'b',linewidth=0.5)
            plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
            plt.xlabel('Epochs ',fontsize=16)
            plt.ylabel('Accuracy',fontsize=16)
            plt.title('Accuracy Curves Digit %i' %i,fontsize=16)
            plt.show()

    def plot_acc(self):
        
        for i in range(1,5):
            plt.figure(figsize=[8,6])
            plt.plot(history.history['digit_%i_loss' %i],'r',linewidth=0.5)
            plt.plot(history.history['val_digit_%i_loss' %i],'b',linewidth=0.5)
            plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
            plt.xlabel('Epochs ',fontsize=16)
            plt.ylabel('Loss',fontsize=16)
            plt.title('Loss Curves Digit %i' %i,fontsize=16)
            plt.show()
        
    def predict(self):
        
        preds = self.model.predict(self.X_val)
        correct_preds = 0
        
        for i in range(X_val.shape[0]):
            pred_list_i = [np.argmax(pred[i]) for pred in y_pred]
            val_list_i  = y_val.values[i].astype('int')
            if np.array_equal(val_list_i, pred_list_i):
                correct_preds = correct_preds + 1
            print('exact accuracy', correct_preds / X_val.shape[0])
            
            mse = 0 
            diff = []
            for i in range(X_val.shape[0]):
                    pred_list_i = [np.argmax(pred[i]) for pred in y_pred]
                    pred_number = 1000* pred_list_i[0] + 100* pred_list_i[1] + 10 * pred_list_i[2] + 1* pred_list_i[3]
                    val_list_i  = y_val.values[i].astype('int')
                    val_number = 1000* val_list_i[0] + 100*  val_list_i[1] + 10 *  val_list_i[2] + 1*  val_list_i[3]
                    diff.append(val_number - pred_number)
            print('difference label vs. prediction', diff)

    
    def train_predict(self):
        
        self.train(self.config.lr, self.config.epochs)
        self.plot_loss()
        self.plot_acc()
        self.predict()
        
        
    class Model_Digit(Model):
    
    def __init__(self, config):
        Model.__init__(self, config)
 
    def data_init(self):
        self.dataset = Dataset_Digit(self.config)
        
        self.data = self.digits_data         
        self.X =  self.data.iloc[:,0]
        self.y = self.data.iloc[:,1]
        
        self.ids_train, self.ids_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.25, random_state=1)
        self.X_train = self.dataset.convert_to_arrays(ids_train)
        self.X_val = self.dataset.convert_to_arrays(ids_val)
              
    def model_init(self):
        
        
        model_input = Input((30,50,1))
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

        self.model = Model(input = model_input , output = output)
        self.model._make_predict_function() 
        
    def train(self, lr = 1e-3, epochs=100):
        optimizer = Adam(lr=lr, decay=lr/10)
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer= optimizer, metrics = ['accuracy'])
        keras.backend.get_session().run(tf.initialize_all_variables())
        history = self.model.fit(self.X_train, self.y_train, batch_size= 32, nb_epoch=30, verbose=1, validation_data=(self.X_val, self.y_val))
        
        
    def plot_acc(self):
        plt.figure(figsize=[8,6])
        plt.plot(history.history['acc'],'r',linewidth=0.5)
        plt.plot(history.history['val_acc'],'b',linewidth=0.5)
        plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Accuracy',fontsize=16)
        plt.title('Accuracy Curves Digit',fontsize=16)
        plt.show()

    def plot_loss(self):        
        plt.figure(figsize=[8,6])
        plt.plot(history.history['loss'],'r',linewidth=0.5)
        plt.plot(history.history['val_loss'],'b',linewidth=0.5)
        plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.title('Loss Curves Digit',fontsize=16)
        plt.show()
        
    def predict(self):
        
        preds = self.model.predict(self.X_val)
        correct_preds = 0
        
    
    def train_predict(self):
        
        self.train(self.config.lr, self.config.epochs)
        self.plot_loss()
        self.plot_acc()
        self.predict()
    
    
    
    
        
        
        
        