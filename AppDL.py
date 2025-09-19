# common dependencies
import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



# package dependencies
from modules import (interface, DLLib, nstLib, FaceNet, facemesh, ResNet50, AgePredictor, rnn)


# 3rd party dependencies
import pandas as pd
import tensorflow as tf
import winsound
#import h5py
import Augmentor


from ultralytics import YOLO
#import dlib 

import warnings
warnings.filterwarnings('ignore')

from termcolor import colored

#from tensorflow import keras
from keras import layers
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import regularizers
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity



#from keras_vggface.vggface import VGGFace

from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')

import imutils
import re

from tensorflow.keras.models import model_from_json

import webbrowser

import threading
import tkinter as tk
from tkinter import *
from tkinter import ttk
import tkinter.scrolledtext as tkscrolled

import cv2  
import numpy as np # linear algebra
from PIL import Image, ImageTk
from PIL.ExifTags import TAGS

from matplotlib import pyplot as plt
from matplotlib import cm

from matplotlib.pyplot import imshow
from matplotlib.colors import ListedColormap

from skimage.color import rgb2gray
from skimage.io import imread
from sklearn import metrics
from sklearn.metrics import precision_score , recall_score, accuracy_score
from sklearn.metrics import f1_score
from sklearn.datasets import make_blobs
import sklearn.datasets

from sklearn.model_selection import train_test_split # to split the dataset for training and testing 

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_tree
#from sklearn.metrics import plot_confusion_matrix
from micromlgen import port
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC
#from sklearn.inspection import DecisionBoundaryDisplay

import time
import math

if getattr(sys, 'frozen', False):
    import pyi_splash

TrainingLoad = False
first_time = 0

binaryclas = False
binaryclasT = False
binaryclasApp = False
MulticlassApp = False
signalFit1 = False
signalFit2 = False
ImageRecognitionH = False
schVar = False
FileLayers = False
XGBoostFlag = False

def set_data(file_pathT, file_pathD):
    global train_x, test_x, train_y, test_y
    global X_train, X_test, Y_train, Y_test
    
    train_x_orig, train_y, test_x_orig, test_y, classes = DLLib.load_dataset(file_pathD,file_pathT)       
    if ImageRecognitionSigns == True:
        X_train = train_x_orig/255.
        X_test = test_x_orig/255.
        Y_train = convert_to_one_hot(train_y, 6).T
        Y_test = convert_to_one_hot(test_y, 6).T
        print(colored(("number of training examples = " + str(X_train.shape[0])),"blue"))
        print(colored(("number of test examples = " + str(X_test.shape[0])),"blue"))
        print("X_train shape: " + str(X_train.shape))
        print("Y_train shape: " + str(Y_train.shape))
        print(colored(("X_test shape: " + str(X_test.shape)),"green"))
        print(colored(("Y_test shape: " + str(Y_test.shape)),"green"))
        
        images_iter = iter(X_train)
        labels_iter = iter(Y_train)
        plt.figure(figsize=(10, 10))
        plt.title("Sample of 25 pictures")

        filas = 5
        columnas = 5
        num_imagenes = filas*columnas                
        for i in range(num_imagenes):
            ax = plt.subplot(filas, columnas, i + 1)
            plt.imshow(X_train[i])
            plt.title(np.argmax(Y_train[i]))
            plt.axis("off")
            #im=Image.fromarray(train_x_orig[i])
            #im.save("signs"+ str(i) + ".jpg")
                
        plt.savefig('output/Fig.png')
        schVar=False                
        display_image('output/Fig.png',schVar,image_label) 
        
        
        
    else:
        # Explore your dataset
        m_train = train_x_orig.shape[0]
        num_px = train_x_orig.shape[1]
        m_test = test_x_orig.shape[0]
        print(colored(("Number of training examples: " + str(m_train)),"blue"))
        print(colored(("Number of testing examples: " + str(m_test)),"blue"))
        print(colored(("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)"),"red", "on_green"))
        print ("train_x_orig shape: " + str(train_x_orig.shape))
        print ("train_y shape: " + str(train_y.shape))
        print(colored(("test_x_orig shape: " + str(test_x_orig.shape)),"green"))
        print(colored(("test_y shape: " + str(test_y.shape)),"green"))        
        OutputTextExif.insert(tk.END, f"\nN training examples: {(m_train):}") 
        OutputTextExif.insert(tk.END, f"\nN test examples: {(m_test):}") 
        # Reshape the training and test examples
        train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
        test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
        # Standardize data to have feature values between 0 and 1.
        train_x = train_x_flatten/255.
        test_x = test_x_flatten/255.        
        #print ("train_x's shape: " + str(train_x.shape))
        #print ("test_x's shape: " + str(test_x.shape))  
        #index = 1
        #plt.imshow(train_x_orig[index])
        #plt.show() 
        if 1==1:
            images_iter = iter(train_x_orig)
            labels_iter = iter(train_y)
            plt.figure(figsize=(10, 10))              
            plt.title("Sample of 25 pictures")
            for i in range(25):
                ax = plt.subplot(5, 5, i + 1)
                plt.imshow(train_x_orig[i])
                #im=Image.fromarray(train_x_orig[i])
                #im.save("cat"+ str(i) + ".jpg")
                #plt.imshow(next(images_iter).numpy().astype("uint8"))            
                #plt.title(train_y[i])            
                plt.axis("off")           
            plt.show()
        
        

def open_image():
    global file_path
    file_path = filedialog.askopenfilename(title="Open Image File", filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")])        
    #path = Path(file_path)
    #suffix=path.suffix
    if file_path:
        display_image(file_path,schVar,image_label)    
        
        
def display_image(file_path,schVar,image_label):
    global grayscaled_result
    global imagec
    maxwidth = 520
    maxheight = 700
    image = Image.open(file_path)
  
    width, height = image.size                # Code to scale up or down as necessary to a given max height or width but keeping aspect ratio
    if width >= height:
        scalingfactor = maxwidth/width
        width = maxwidth
        height = int(height*scalingfactor)
    else:
        scalingfactor = maxheight/height
        height = maxheight
        width = int(width*scalingfactor)     
    
    image = image.resize((width,height), Image.Resampling.LANCZOS)    
    #photoIM = ImageTk.PhotoImage(original_image)
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)        
    # iterating over all EXIF data fields              
    image_label.photo = photo    
    #OutputText.delete("1.0", "end")
    #OutputText.insert(tk.END, '')    
    if schVar == False:
        if ImageRecognitionH == False:
            num_px=64
        else:
            num_px=128        
        imagec = np.array(Image.open(file_path).resize((num_px, num_px)))    
        #Image.open(file_path).resize((num_px, num_px))).save("geeks.png")
        imagec=imagec[:,:,:3]   #image without alpha channel for PNG images format
        imagec = imagec / 255.        
        if ImageRecognitionSigns == True or ConvModelVar.get() == True:
            imagec = imagec.reshape((1, num_px, num_px, 3))
        else:
            imagec = imagec.reshape((1, num_px * num_px * 3)).T
        
    else:
        schVar = False
    #print(imagec)
    #my_image_prediction = predict(image, parameters)          
        
       
def schedule_check(t):
    """
    Programar la ejecución de la función `check_if_done()` dentro de 
    un segundo.
    """
    root.after(100, check_if_done, t)


def showEND():
    frequency = 1250
    duration = 100
    winsound.Beep(frequency, duration)

    if NeuralStyleTransferFlag == True:
        OutputTextExif.insert(tk.END, f"\nLRate: {(Lrate):.6f}")
        schVar = True        
        display_image('output/image_0.jpg',schVar,image_label)
        #img.show()  
    
    else:
        if RNNFlag == True:
             OutputTextExif.insert(tk.END, f"\nDino: {(last_name):}")  
        else:
            if ConvModelVar.get() == 1:
                OutputTextExif.insert(tk.END, f"\nLRate: {(Lrate):.6f}")  
            else:
                OutputTextExif.insert(tk.END, f"\nLRate Decay: {(LDecay.get()):}")  
                OutputTextExif.insert(tk.END, f"\nLRate: {(Lrate):.6f}")  
                OutputTextExif.insert(tk.END, f"\nBeta: {(Beta):.6f}") 
                OutputTextExif.insert(tk.END, f"\nLayerDim: {(layers_dims):}")
                OutputTextExif.insert(tk.END, f"\nMiniBatchEn: {(MiniBatch.get()):}")
                OutputTextExif.insert(tk.END, f"\nMiniBatchSize: {(MiniBatchSize):}") 
            if binaryclas == False and binaryclasT == False and signalFit1 == False and signalFit2 == False and SignalFitCSV == False and ConvModelVar.get() == 0:
                pred_test, AccTest = DLLib.predict(test_x, test_y, parameters, activation = activationMVar, activationL = activationLVar) 
                OutputTextExif.insert(tk.END, f"\nAccuracy Test: {(AccTest):.6f}")     
                pred_train, AccTrain = DLLib.predict(train_x, train_y, parameters, activation = activationMVar, activationL = activationLVar)      
                OutputTextExif.insert(tk.END, f"\nAccu Training: {(AccTrain):.6f}")                      
                confusion_matrix = metrics.confusion_matrix(np.squeeze(test_y), np.squeeze(pred_test))
                cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])                       
                print(colored(("Acc Training = ",AccTrain),"green"))
                sklearn_precision = precision_score(np.squeeze(pred_test),np.squeeze(test_y))
                print(colored(("Precision Test = ",sklearn_precision),"red", "on_green"))
                sklearn_recall = recall_score(np.squeeze(pred_test),np.squeeze(test_y))
                print(colored(("Recall Test = ",sklearn_recall),"cyan"))
                sklearn_f1_score = f1_score(np.squeeze(pred_test),np.squeeze(test_y))
                print(colored(("F1_Score Test = ",sklearn_f1_score),"red"))
                acc = accuracy_score(np.squeeze(pred_test),np.squeeze(test_y))
                print(colored(("Acc Test = ",acc),"red", "on_green"))
                OutputTextExif.insert(tk.END, f"\nPrecision Test: {(sklearn_precision):.6f}")
                OutputTextExif.insert(tk.END, f"\nRecall Test: {(sklearn_recall):.6f}")
                OutputTextExif.insert(tk.END, f"\nF1_Score Test: {(sklearn_f1_score):.6f}")
                OutputTextExif.insert(tk.END, f"\nAccuracy Test: {(acc):.6f}")               
                cm_display.plot()
                plt.savefig('output/Fig.png')
                schVar=False                
                display_image('output/Fig.png',schVar,image_label)                
                plt.show()            
            if MulticlassApp == True:                
                model_predict = lambda Xl: np.argmax(model.predict(Xl),axis=1)
                fig,ax = plt.subplots(1,1)
                fig.canvas.toolbar_visible = False
                fig.canvas.header_visible = False
                fig.canvas.footer_visible = False
             
                #add the original data to the decison boundary
                interface.plt_mc_data(ax, train_X,train_Y, classes, legend=True)
                #plot the decison boundary. 
                interface.plot_cat_decision_boundary_mc(ax, train_X, model_predict, vector=True)
                ax.set_title("model decision boundary")
                plt.xlabel(r'$x_0$');
                plt.ylabel(r"$x_1$");                 
                fig.savefig('output/Fig.png')
                schVar=False            
                display_image('output/Fig.png',schVar,image_label)   
            elif ConvModelVar.get() == 1 and ImageRecognitionSigns == False:         
                prediction = model.predict(test_x)               
                #model.predict(test_x, verbose=0)[:100]
                #np.set_printoptions(threshold=sys.maxsize)      #print all data                
                prediction = np.round(prediction,1).astype(int)  
                #prediction =  np.array(prediction).argmax(axis=1)   #find the max value of array and return its position               
                confusion_matrix = metrics.confusion_matrix(np.squeeze(test_x.labels), np.squeeze(prediction))                
                cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])                       
                cm_display.plot()
                plt.savefig('output/Fig.png')
                schVar=False                
                display_image('output/Fig.png',schVar,image_label) 
                #plt.show()
            elif ConvModelVar.get() == 1 and ImageRecognitionSigns == True:         
                prediction = model.predict(X_test)
                #np.set_printoptions(threshold=sys.maxsize)      #print all data
                #print(prediction)
                prediction =  np.array(prediction).argmax(axis=1)   #find the max value of array and return its position
                #prediction=np.argmax(prediction, axis=1)
                Y_test2=np.argmax(Y_test, axis=1)
                #print(prediction)
                #print(Y_test2)                
                confusion_matrix = metrics.confusion_matrix(np.squeeze(Y_test2), np.squeeze(prediction))
                cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1, 2, 3, 4, 5])                       
                cm_display.plot()
                plt.savefig('output/Fig.png')
                schVar=False                
                display_image('output/Fig.png',schVar,image_label) 
                plt.show()          
            elif binaryclas == True:
                pred_train, AccTrain = DLLib.predict(train_x, train_y, parameters, activation = activationMVar, activationL = activationLVar)      
                OutputTextExif.insert(tk.END, f"\nAccu Training: {(AccTrain):.6f}")    
                plt.clf()   # Clear figure
                plt.figure(2)
                interface.plot_decision_boundary(lambda x: DLLib.predict_dec(parameters,x.T, activation = activationMVar, activationL = activationLVar), train_x, train_y, Show=False)
                plt.title("Decision Boundary for NN size " +  str(layers_dims))               
                plt.figure(2).savefig('output/Fig.png')
                schVar=False                
                display_image('output/Fig.png',schVar,image_label)                
            elif binaryclasT == True:
                pred_train, AccTrain = DLLib.predict(train_x, train_y, parameters, activation = activationMVar, activationL = activationLVar)      
                OutputTextExif.insert(tk.END, f"\nAccu Training: {(AccTrain):.6f}")    
                pred_test, AccTest = DLLib.predict(test_X, test_Y, parameters, activation = activationMVar, activationL = activationLVar)      
                OutputTextExif.insert(tk.END, f"\nAccu Test: {(AccTest):.6f}")
                plt.clf()   # Clear figure
                plt.figure(2)
                #plt.title("Model without regularization")
                axes = plt.gca()
                axes.set_xlim([-0.75,0.40])
                axes.set_ylim([-0.75,0.65])
                interface.plot_decision_boundary(lambda x: DLLib.predict_dec(parameters,x.T, activation = activationMVar, activationL = activationLVar), train_x, train_y, Show=False)
                plt.xlabel("x1")
                plt.ylabel('x2')  
                plt.title("Decision Boundary for NN size " +  str(layers_dims))               
                plt.figure(2).savefig('output/Fig.png')
                schVar=False                
                display_image('output/Fig.png',schVar,image_label)                

            elif BinaryClassComparison == True:                  
                plt.clf()   # Clear figure
                names = [
                    "Nearest Neighbors",                   
                    "RBF SVM",
                    "Gaussian Process",
                    "Decision Tree",
                    "Random Forest",
                    "Neural Net",                    
                ]
                cm_bright = ListedColormap(["#FF0000", "#0000FF"])
                i = 1
                for model in models:
                    ax = plt.subplot(2,3, i)                   
                        
                    interface.plot_cat_decision_boundary_mc2(ax, X, model, vector=True)
                    ax.scatter(train_x.T[:, 0], train_x.T[:, 1], c=train_y.T, cmap=cm_bright, edgecolors="k")
                    # Plot the testing points
                    ax.scatter(test_X[:, 0], test_X[:, 1], c=test_Y, cmap=cm_bright, edgecolors="k", alpha=0.6,)
                    score = model.score(test_X, test_Y)
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)                    
                    ax.set_title(names[i-1])
                    ax.text(
                        x_max - 0.3,
                        y_min + 0.3,
                        ("%.3f" % score).lstrip("0"),
                        size=15,
                        horizontalalignment="right",
                    )
                    i = i + 1             
                          
                plt.show()             
            elif binaryclasApp == True:
                pred_train, AccTrain = DLLib.predict(train_x, train_y, parameters, activation = activationMVar, activationL = activationLVar)      
                OutputTextExif.insert(tk.END, f"\nAccu Training: {(AccTrain):.6f}")    
                plt.clf()   # Clear figure
                plt.figure(2)
                interface.plot_decision_boundary(lambda x: DLLib.predict_dec(parameters,x.T, activation = activationMVar, activationL = activationLVar), train_x, train_y, Show=False)
                plt.title("Decision Boundary for NN size " +  str(layers_dims))                
                plt.xlabel("MicroChip Test 1")
                plt.ylabel('MicroChip Test 2')           
                plt.figure(2).savefig('output/Fig.png')
                schVar=False                
                display_image('output/Fig.png',schVar,image_label)          
            elif signalFit1 == True:      
                x = np.linspace(0, 2 * np.pi, 1000)
                y = np.sin(x)
                plt.clf()   # Clear figure
                plt.figure(2)            
                plt.plot(x, y)             
                x=train_x
                prediction = DLLib.predict3(x,  parameters, activation = activationMVar, activationL = activationLVar) 
                plt.plot(np.squeeze(x), np.squeeze(prediction))           
                #plt.plot(np.squeeze(x), np.squeeze(prediction)-train_Y)
                plt.grid(True)
                plt.figure(2).savefig('output/Fig.png')
                schVar=False                
                display_image('output/Fig.png',schVar,image_label)
                plt.show() 
            elif signalFit2 == True:          
                x = np.linspace(0, 2 * np.pi, 1000)
                y = np.matrix([np.sin(x),np.sin(x-2*np.pi/3),np.sin(x+2*np.pi/3)])
                plt.clf()   # Clear figure
                plt.figure(2)            
                plt.plot(x, y.T)             
                x=train_x
                prediction = DLLib.predict3(x,  parameters, activation = activationMVar, activationL = activationLVar)                          
                plt.plot(np.squeeze(x), np.squeeze(prediction).T)           
                #plt.plot(np.squeeze(x), np.squeeze(prediction)-train_Y)
                plt.grid(True)
                plt.figure(2).savefig('output/Fig.png')
                schVar=False                
                display_image('output/Fig.png',schVar,image_label)                
                plt.show()
            elif SignalFitCSV == True:  
                x=train_x                
                prediction = DLLib.predict3(x,  parameters, activation = activationMVar, activationL = activationLVar) 
                plt.clf()   # Clear figure
                plt.figure(2)
                plt.plot(np.squeeze(prediction))
                plt.plot(np.squeeze(train_Y.T))            
                plt.grid(True)
                plt.figure(2).savefig('output/Fig.png') 
                schVar=False
                display_image('output/Fig.png',schVar,image_label)                   
                plt.show()
            elif XGBoostFlag == True:
                plt.figure(1)  
                plot_confusion_matrix(model, train_X, train_Y, cmap='Blues')
                plt.figure(2)  
                plot_confusion_matrix(model, test_X, test_Y, cmap='Blues')
                plt.show()        
                #print(port(model))
                #with open('XGBClassifier.h', 'w') as file:
                     #file.write(port(model))                  


def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """

    input_img = tf.keras.Input(shape=input_shape)
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    Z1 = tf.keras.layers.Conv2D(filters = 8 , kernel_size= (4,4), strides = (1,1), padding='same')(input_img)
    ## RELU
    A1 = tf.keras.layers.ReLU()(Z1)
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.keras.layers.MaxPool2D(pool_size=(8,8), strides=(8, 8), padding='same')(A1)
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    Z2 = tf.keras.layers.Conv2D(filters = 16 , kernel_size= (2,2), strides = (1,1), padding='same')(P1)
    ## RELU
    A2 = tf.keras.layers.ReLU()(Z2)
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.keras.layers.MaxPool2D(pool_size=(4,4), strides=(4, 4), padding='same')(A2)
    ## FLATTEN
    F = tf.keras.layers.Flatten()(P2)
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
    outputs = tf.keras.layers.Dense(units=6, activation='softmax')(F)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model        
 
 
                      
def graficar_valor_arreglo(i, arr_predicciones, etiqueta_real):
    arr_predicciones, etiqueta_real = arr_predicciones[i], etiqueta_real[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    grafica = plt.bar(range(6), arr_predicciones, color="#777777")
    plt.ylim([0, 1]) 
    etiqueta_prediccion = np.argmax(arr_predicciones)

    grafica[etiqueta_prediccion].set_color('red')
    print(etiqueta_real)
    #grafica[etiqueta_real].set_color('blue')



def TrainExe():
    global start
    global parameters
    global costs, Lrate, Beta
    global activationMVar, activationLVar, MiniBatchSize
    global test_x, text_y, test_it, history, test_y, train_x, train_y
    global model, models
    global generated_image, vgg_model_outputs,  a_C, a_S, optimizer
    global data, char_to_ix, ix_to_char
    global last_name
    
    start = time.time()
    #print(LayerDimension)
    Niternations = num_iterations.get()
    Lrate = LearningRate.get()
    if IniWeights.get() == 1:
        IniSTR = "zeros"
    else:
        if IniWeights.get() == 2:
            IniSTR = "random"
        else:
            IniSTR = "he"
    if optimizer.get() == 1:
        OptSTR = "gd"
    else:
        if optimizer.get() == 2:
            OptSTR = "momentum"
        else:
            OptSTR = "adam"
    if FileLayers == False:
        if ActivationM.get() == 1:
            activationMVar ="relu"
        else:
            activationMVar ="tanh"
        if ActivationL.get() == 1:
            activationLVar ="sigmoid"
        else:
            activationLVar ="linear"
        
    if CostFunction.get() == 1:
        cost_functionVar ="binaryCross"
    else:
        cost_functionVar ="mse"        
            
    lambDA = lambDAVar.get()
    MiniBatchSize = MiniBatchVar.get()
    Beta = BetaVar.get()
    
    if NeuralStyleTransferFlag == True:
        generated_image = tf.Variable(generated_image)
        #train_step_test(train_step, generated_image)        
        epochs = int(Niternations)
        optim = tf.keras.optimizers.Adam(learning_rate=float(LearningRate.get()))
        for i in range(epochs):
            cost_avg=nstLib.train_step(optim,vgg_model_outputs,generated_image,a_S,a_C)            
            if epochs > 1000:
                # Print the cost every 100 iterations
                if i % 250 == 0 or i == epochs - 1:                
                    print("Cost after epoch {}: {:.6f}".format(i, np.squeeze(cost_avg)))                   
                    #image = tensor_to_image(generated_image)
                    #imshow(image)
                    #image.save(f"output/image_{i}.jpg")
                    #plt.show()              
            elif epochs > 500:
                # Print the cost every 10 iterations
                if i % 10 == 0 or i == epochs - 1:
                    print("Cost after epoch {}: {:.6f}".format(i, np.squeeze(cost_avg)))                        
            else:
                # Print the cost every 1 iterations
                if i % 1 == 0 or i == epochs - 1:
                    print("Cost after epoch {}: {:.6f}".format(i, np.squeeze(cost_avg)))         
        image = nstLib.tensor_to_image(generated_image)        
        image.save(f"output/image_0.jpg")         
    else:
        if RNNFlag == True:
            parameters, last_name = rnn.modelRNN(data.split("\n"), ix_to_char, char_to_ix, int(Niternations), learning_rate=float(Lrate), verbose = True)           
        else:        
            if ConvModelVar.get() == 1: #if convolutional model   
                if ImageRecognitionSigns == True:                
                    if Resnet.get() == 1:
                        #tf.keras.backend.set_learning_phase(True)
                        model = ResNet50.ResNet50(input_shape = (64, 64, 3), classes = 6)
                        print(model.summary())
                        np.random.seed(1)
                        tf.random.set_seed(2)
                        opt = tf.keras.optimizers.Adam(learning_rate=float(Lrate))
                        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
                        
                        #train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
                        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
                        history=model.fit(X_train, Y_train, epochs = int(Niternations), batch_size = 32, validation_data=test_dataset)
                                                                                                
                        preds = model.evaluate(X_test, Y_test, batch_size = 32)
                        print(colored(("Loss = " + str(preds[0])),"green"))
                        print(colored(("Test Acc = " + str(preds[1])), "blue"))
                        
                        if 0==1:
                            for imagenes_prueba, etiquetas_prueba in test_dataset.take(1):
                              imagenes_prueba = imagenes_prueba.numpy()
                              etiquetas_prueba = etiquetas_prueba.numpy()
                              predicciones = model.predict(imagenes_prueba)
                            
                            filas = 5
                            columnas = 5
                            num_imagenes = filas*columnas
                            fig1, axes1 = plt.subplots(filas, columnas, figsize=(1.5*columnas,2*filas))
                            for i in range(num_imagenes):
                              plt.subplot(filas, 2*columnas, 2*i+1)
                              ax = axes1[i//columnas, i%columnas]
                              ax.imshow(X_train[i].reshape(64,64,3))
                              ax.set_title('Label: {}'.format(np.argmax(Y_train[i])))
                              plt.subplot(filas, 2*columnas, 2*i+2)
                              graficar_valor_arreglo(i, predicciones, etiquetas_prueba)                
                        
                        if 0==1:                    
                            filas = 4
                            columnas = 8
                            num = filas*columnas                    
                            fig1, axes1 = plt.subplots(filas, columnas, figsize=(1.5*columnas,2*filas))
                            for i in range(num):
                                 ax = axes1[i//columnas, i%columnas]
                                 ax.imshow(X_train[i].reshape(64,64,3))
                                 ax.set_title('Label: {}'.format(np.argmax(Y_train[i])))
                            plt.tight_layout()
                            plt.show()
                               
                        #pre_trained_model = load_model('resnet50.h5')                
                        #preds = pre_trained_model.evaluate(X_test, Y_test)
                        #print ("Loss = " + str(preds[0]))
                        #print ("Test Accuracy = " + str(preds[1]))
                        
                    else:#RecognitionSigns convolutional model
                        model = convolutional_model((64, 64, 3))
                        opt = tf.keras.optimizers.Adam(learning_rate=float(Lrate))
                        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
                        model.summary()                        
                        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
                        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
                        
                        #AUTOTUNE = tf.data.experimental.AUTOTUNE
                        #train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
                        
                        if DataAugmented.get():
                            rango_rotacion = 30
                            mov_ancho = 0.25
                            mov_alto = 0.25
                            #rango_inclinacion=15 #No uso este de momento pero si quieres puedes probar usandolo!
                            rango_acercamiento=[0.5,1.5]

                            datagen = ImageDataGenerator(
                                rotation_range = rango_rotacion,
                                width_shift_range = mov_ancho,
                                height_shift_range = mov_alto,
                                zoom_range=rango_acercamiento,
                                #shear_range=rango_inclinacion #No uso este de momento pero si quieres puedes probar usandolo!
                            )
                            datagen.fit(X_train)                   
                            if 0==1:
                                print(X_train.shape)
                                filas = 4
                                columnas = 8
                                num = filas*columnas
                                print('ANTES:\n')
                                fig1, axes1 = plt.subplots(filas, columnas, figsize=(1.5*columnas,2*filas))
                                for i in range(num):
                                     ax = axes1[i//columnas, i%columnas]
                                     ax.imshow(X_train[i].reshape(64,64,3))
                                     ax.set_title('Label: {}'.format(np.argmax(Y_train[i])))
                                plt.tight_layout()
                                plt.show()
                                print('DESPUES:\n')
                                fig2, axes2 = plt.subplots(filas, columnas, figsize=(1.5*columnas,2*filas))
                                for X, Y in datagen.flow(X_train,Y_train, batch_size=num,shuffle=False):
                                     for i in range(0, num):
                                          ax = axes2[i//columnas, i%columnas]
                                          ax.imshow(X[i].reshape(64,64,3))
                                          ax.set_title('Label: {}'.format(int(np.argmax(Y[i]))))
                                     break
                                plt.tight_layout()
                                plt.show()
                            #Los datos para entrenar saldran del datagen, de manera que sean generados con las transformaciones que indicamos
                            data_gen_entrenamiento = datagen.flow(X_train, Y_train, batch_size=32)
                            history = model.fit(data_gen_entrenamiento, epochs=int(Niternations), validation_data=test_dataset)
                            model.save('numeros_sign_conv_ad_do.keras')
                            #mkdir carpeta_salida

                            #!tensorflowjs_converter --input_format keras numeros_sign_conv_ad_do.h5 carpeta_salida
                        else:
                            history = model.fit(train_dataset, epochs=int(Niternations), validation_data=test_dataset)            
                            model.save('output/numeros_sign_conv.keras')
                  
                else:
                    if Resnet.get() == 1:
                        datagen = ImageDataGenerator(rescale=1.0/1.0)
                        train_x = datagen.flow_from_directory('support/dataset_dogs_vs_cats/train/', class_mode='binary', batch_size=128, target_size=(128, 128))
                        test_x = datagen.flow_from_directory('support/dataset_dogs_vs_cats/test/', class_mode='binary', batch_size=128, target_size=(128, 128))              
                        model = ResNet50.ResNet50(input_shape = (128, 128, 3), classes = 2)
                        print(model.summary())
                        np.random.seed(1)
                        tf.random.set_seed(2)
                        opt = tf.keras.optimizers.Adam(learning_rate=float(Lrate))
                        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
                        
                        #train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
                        test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_x.labels)).batch(64)
                        history=model.fit(train_x, epochs = int(Niternations), batch_size = 32, validation_data=test_x)
                                                                                                
                        preds = model.evaluate(X_test, Y_test, batch_size = 32)
                        print(colored(("Loss = " + str(preds[0])),"green"))
                        print(colored(("Test Acc = " + str(preds[1])), "blue"))
                    else:
                        datagen = ImageDataGenerator(rescale=1.0/1.0)
                        print("Convolutional Model:")
                        # prepare iterators
                        train_x = datagen.flow_from_directory('support/dataset_dogs_vs_cats/train/', class_mode='binary', batch_size=int(MiniBatchVar.get()), target_size=(128, 128))
                        test_x = datagen.flow_from_directory('support/dataset_dogs_vs_cats/test/', class_mode='binary', batch_size=int(MiniBatchVar.get()), target_size=(128, 128))                        
                        print("Minibatch", int(MiniBatchVar.get()))
                        model = tf.keras.models.Sequential([
                            tf.keras.layers.Rescaling(1./255, input_shape=(128, 128, 3)),
                            #tf.keras.layers.Flatten(input_shape=(128, 128, 3)),
                            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
                            layers.MaxPooling2D(2, 2),
                            layers.Conv2D(64, (3, 3), activation='relu'),
                            layers.MaxPooling2D(2, 2),
                            layers.Conv2D(64, (3, 3), activation='relu'),
                            layers.MaxPooling2D(2, 2),
                            layers.Conv2D(64, (3, 3), activation='relu'),
                            layers.MaxPooling2D(2, 2),

                            layers.Flatten(),
                            layers.Dense(512, activation='relu'),
                            layers.BatchNormalization(),
                            layers.Dense(512, activation='relu'),
                            layers.Dropout(0.1),
                            layers.BatchNormalization(),
                            layers.Dense(512, activation='relu'),
                            layers.Dropout(0.2),
                            layers.BatchNormalization(),
                            layers.Dense(1, activation='sigmoid')
                        ])
                        opt = tf.keras.optimizers.Adam(learning_rate=float(Lrate))
                        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
                        model.summary()                   
                        #history = model.fit(x=train_x_orig, y=train_y.T, batch_size= 128, epochs=int(Niternations), validation_data=(test_x_orig,test_y.T))
                        history = model.fit(train_x, batch_size = int(MiniBatchVar.get()), epochs=int(Niternations), validation_data=test_x)
            else:       
                if 0==1:
                    datagen = ImageDataGenerator(rescale=1.0/1.0)
                    # prepare iterators
                    train_x = datagen.flow_from_directory('dataset_dogs_vs_cats/train/', class_mode='binary', batch_size=int(TrainSizeVar.get()), target_size=(128, 128))
                    test_x = datagen.flow_from_directory('dataset_dogs_vs_cats/test/', class_mode='binary', batch_size=int(TestSizeVar.get()), target_size=(128, 128))                               
                    images, labels=next(test_x)           
                    # Reshape the training and test examples
                    test_x_flatten = images.reshape(images.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
                    test_y = labels.reshape(labels.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions                    
                    images, labels=next(train_x)           
                    # Reshape the training and test examples
                    train_x_flatten = images.reshape(images.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
                    train_y = labels.reshape(labels.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions                                
                    print("X",test_x_flatten.shape)
                    print("Y",test_y.shape)                    
                    # Standardize data to have feature values between 0 and 1.            
                    train_x = train_x_flatten/255.0
                    test_x = test_x_flatten/255.0                
                
                if XGBoostFlag == True:
                    # fit model no training data                    
                    RANDOM_STATE = 55 ## We will pass it to every sklearn call so we ensure reproducibility
                    model = XGBClassifier(n_estimators = 500, learning_rate = float(Lrate),verbosity = 1, random_state = RANDOM_STATE)
                    model.fit(train_X, train_Y,  eval_set = [(test_X,test_Y)], early_stopping_rounds = 10)
                    print(test_X[0:1])
                    print(test_Y[0:1])
                    y_pred = model.predict(test_X)
                    print("Prediction",y_pred)               
                    print("Best interation",model.best_iteration)
                    print(f"Metrics train:\n\tAccuracy score: {accuracy_score(model.predict(train_X),train_Y):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(model.predict(test_X),test_Y):.4f}")
                    print(metrics.classification_report(test_Y, y_pred, digits = 3))
                    #plot_tree(model, num_trees=1, rankdir='TD')  #The 'rankdir' argument controls the direction of the tree (LR: left to right / TD: top to down)
                    #plt.figure(figsize=(12, 8))
                                
                
                
                elif MulticlassApp == True:
                
                    if 1==0:
                        model = Sequential([ 
                                Dense(25, activation = 'relu'),
                                Dense(15, activation = 'relu'),
                                Dense(4, activation = 'softmax')    # < softmax activation here
                            ]
                        )
                        model.summary()
                        model.compile(
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                            optimizer=tf.keras.optimizers.Adam(float(Lrate)),
                        )
                        model.fit(train_X,train_Y, batch_size = int(MiniBatchVar.get()), epochs=int(Niternations)) 
                        
                    else:
                                                
                        model = Sequential([ 
                                Dense(120, activation = 'relu', name = "L1", kernel_regularizer=tf.keras.regularizers.l2(float(lambDA))),
                                Dense(40, activation = 'relu', name = "L2", kernel_regularizer=tf.keras.regularizers.l2(float(lambDA))),
                                Dense(classes, activation = 'linear', name = "L3")   #<-- Note
                            ]
                        )
                        #model.summary()
                        model.compile(
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  #<-- Note
                            optimizer=tf.keras.optimizers.Adam(float(Lrate)),
                        )                        
                        history = model.fit(train_X,train_Y, batch_size = int(MiniBatchVar.get()), epochs=int(Niternations))                
                else:
                    if BinaryClassComparison == True:                       
                        models=[KNeighborsClassifier(n_neighbors=3).fit(train_x.T,train_y.T),
                        SVC(gamma=2, C=1, random_state=42).fit(train_x.T,train_y.T),
                        GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42).fit(train_x.T,train_y.T),
                        DecisionTreeClassifier(min_samples_split = 100, max_depth=5, random_state = 42).fit(train_x.T,train_y.T),
                        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42).fit(train_x.T,train_y.T),
                        MLPClassifier(hidden_layer_sizes=(10,),activation='relu', verbose=10, alpha=1.2, max_iter=1000, random_state=42).fit(train_x.T,train_y.T),
                        ]
                    else:
                        if MiniBatch.get() == 0:
                            parameters, costs = DLLib.L_layer_model(train_x, train_y, layers_dims, keep_prob_list, beta = float(Beta), beta1 = 0.9, beta2 = 0.999,
                                                epsilon = 1e-8, optimizer = OptSTR, learning_rate = float(Lrate),num_epochs = int(Niternations),
                                                print_cost = True, initialization = IniSTR, lambd = float(lambDA), decay = LDecay.get(), time_interval = TimeInterVar.get(),
                                                activation = activationMVar, activationL = activationLVar, cost_function = cost_functionVar)
                        else:    
                            parameters, costs = DLLib.L_layer_modelMiniBatch(train_x, train_y, layers_dims, keep_prob_list, beta = float(Beta), beta1 = 0.9, beta2 = 0.999,  
                                                                            epsilon = 1e-8, optimizer = OptSTR, learning_rate = float(Lrate), num_epochs = int(Niternations), 
                                                                            mini_batch_size = int(MiniBatchSize), print_cost = True, initialization = IniSTR, lambd = float(lambDA), 
                                                                            decay = LDecay.get(), time_interval = TimeInterVar.get(), activation = activationMVar, 
                                                                            activationL = activationLVar, cost_function = cost_functionVar)

                    
                    
                   
                                                                        
                     
        
def plot():
    # plotting the graph 
    if ConvModelVar.get() == 1:
        if ImageRecognitionSigns == True:
            # The history.history["loss"] entry is a dictionary with as many values as epochs that the
            # model was trained on. 
            df_loss_acc = pd.DataFrame(history.history)
            df_loss= df_loss_acc[['loss','val_loss']]
            df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
            df_acc= df_loss_acc[['accuracy','val_accuracy']]
            df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
            df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
            df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
            plt.show()
        else:
            history_df = pd.DataFrame(history.history)
            history_df.loc[:, ['loss', 'val_loss']].plot()
            history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
            plt.show()
            # prediction
            result = model.predict(test_x,batch_size = 32,verbose = 0)

            # Evaluvate
            loss,acc = model.evaluate(test_x, batch_size = 32, verbose = 0)

            print('The accuracy of the model for testing data is:',acc*100)
            print('The Loss of the model for testing data is:',loss)   
    if MulticlassApp == True:
        fig,ax = plt.subplots(1,1)        
        ax.plot(history.history['loss'], label='loss')
        ax.set_ylim([0, 2])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('loss (cost)')
        ax.legend()
        ax.grid(True)
        plt.show()
    
    
    else:
        plt.figure(1)
        plt.plot(np.squeeze(costs))     
        plt.title("Cost NN size " +  str(layers_dims))
        plt.xlabel('iterations (per ten)')
        #2,1, xlabel='iterations (per ten)', ylabel ='Cost')    #add_subplot(nrows, ncolumns, position)
        #colorbar(pc, shrink=0.6, ax=axsRight)    
        #plt.show()
        if binaryclas == True or binaryclasApp == True or binaryclasT == True:
            # Plot the decision boundary 
            plt.figure(2)
            interface.plot_decision_boundary(lambda x: DLLib.predict_dec(parameters, x.T, activation = activationMVar, activationL = activationLVar), train_x, train_y)
            plt.title("Decision Boundary for NN size " +  str(layers_dims))                     
        #plt.grid(True)
        plt.show()

       
def check_if_done(t):
    # Si el hilo ha finalizado, restaruar el botón y mostrar un mensaje.
    if not t.is_alive():
        end = time.time()
        progressbar.step(99.9)
        progressbar.stop()
        OutputText.delete("1.0", "end")
        OutputText.insert(tk.END, f"Processed\nTook: {(end - start):.3f}s")     
        showEND()     
        
        """plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per ten)')
        #plt.title("Learning rate, Beta =" + str(Lrate)+ str(Beta))
        plt.show()  """
        
    else:
        # Si no, volver a chequear en unos momentos.
        schedule_check(t) 
    
    
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y    

    
def train():
    global train_X,train_Y   
    global progressbar
    global layers_dims
    global train_x, train_y   
    global test_x, test_y   

    if binaryclas == True or binaryclasApp == True or  binaryclasT == True or BinaryClassComparison == True:    
        train_x=train_X
        train_y=train_Y
        print(train_x.shape)
        print(train_y.shape)
        LayerDimension=[]
        LayerDimension=LayerDim.get()          
        input_array = interface.Convert(LayerDimension)
        layers_dims = interface.convert_strings_to_ints(input_array)
        print(layers_dims)
    else:
        if SignalFitCSV == True:
            print('X1', x1_feat[:,0])
            print('X2', x2_feat[:,None])
            x=np.concatenate((x1_feat,x2_feat), axis=1)
            y=y_feat      #T.shape should b (x,m)

            print(x)
            print(y)
            train_X = x
            train_Y = y
            train_X = train_X.T
            train_Y = train_Y.T 
            print(train_X.shape)
            print(train_Y.shape)            
            train_x=train_X
            train_y=train_Y           
            LayerDimension=[]
            LayerDimension=LayerDim.get()          
            input_array = interface.Convert(LayerDimension)
            layers_dims = interface.convert_strings_to_ints(input_array)      
        elif signalFit1 == True:
            x = np.linspace(0, 2 * np.pi, 1000)
            #x = (x - x.mean())/x.std()
            y = np.sin(x)
            x=x[:,None]
            y=y[:,None]      #T.shape should b (x,m)
            train_X = x
            train_Y = y
            train_X = train_X.T
            train_Y = train_Y.T            
            train_x=train_X
            train_y=train_Y           
            LayerDimension=[]
            LayerDimension=LayerDim.get()          
            input_array = interface.Convert(LayerDimension)
            layers_dims = interface.convert_strings_to_ints(input_array)           
        else:
            if signalFit2 == True:
                x = np.linspace(0, 2 * np.pi, 1000)
                #x = (x - x.mean())/x.std()
                y = np.matrix([np.sin(x),np.sin(x-2*np.pi/3),np.sin(x+2*np.pi/3)])
                x=x[:,None]
                #y=y.T#[:,None]      #T.shape should b (x,m)
                #print(y.shape) 
                #print(x)
                train_X = x
                train_Y = y
                train_X = train_X.T
                #train_Y = train_Y.T
                train_Y = train_Y
                train_x=train_X
                train_y=train_Y        
                LayerDimension=[]
                LayerDimension=LayerDim.get()
                #LayerDimension=getvar(LayerDim_entry.cget('textvariable'))
                input_array = interface.Convert(LayerDimension)
                layers_dims = interface.convert_strings_to_ints(input_array)
            else:              
                LayerDimension=[]
                LayerDimension=LayerDim.get()                           
                input_array = interface.Convert(LayerDimension)
                layers_dims = interface.convert_strings_to_ints(input_array)                
    
    progressbar = ttk.Progressbar(mode="indeterminate")
    progressbar.place(x=10, y=770, width=200)
    # Iniciar el movimiento de la barra indeterminada.
    progressbar.start(10)
    OutputText.delete("1.0", "end")
    OutputText.insert(tk.END, 'Processing...')
    
    # Iniciar la descarga en un nuevo hilo.
    t = threading.Thread(target=TrainExe)
    t.start()
    # Comenzar a chequear periódicamente si el hilo ha finalizado.
    schedule_check(t)
    
    
def check_button(first_time):
    global test_X, test_Y, classes, x_min, x_max, y_min, y_max, X, data_y
    
    if binaryclas == True:
        if first_time == 0:
            first_time = 1
            noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = DLLib.load_extra_datasets()

            datasets = {"noisy_circles": noisy_circles,
                        "noisy_moons": noisy_moons,
                        "blobs": blobs,
                        "gaussian_quantiles": gaussian_quantiles}

            dataset = "gaussian_quantiles"
            plt.figure(1)
            train_X, train_Y = datasets[dataset]            
            plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
            plt.title("Input data Analysis ")  
            plt.figure(1).savefig('output/Fig.png')
            schVar=False            
            display_image('output/Fig.png',schVar,image_label)
            plt.show()
            train_X = train_X.T
            train_Y = train_Y.reshape((1, train_Y.shape[0]))  
            if dataset == "noisy_moons":
                train_Y = train_Y%2
    elif BinaryClassComparison == True:
        if first_time == 0:
            first_time = 1
            noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = DLLib.load_extra_datasets()

            datasets = {"noisy_circles": noisy_circles,
                        "noisy_moons": noisy_moons,
                        "blobs": blobs,
                        "gaussian_quantiles": gaussian_quantiles}

            dataset = "gaussian_quantiles"
            plt.figure(1)
            X, data_y = datasets[dataset]           
            train_X, test_X, train_Y, test_Y = train_test_split(X, data_y, test_size=0.4, random_state=42)
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
            # Plot the testing points
            plt.scatter(test_X[:, 0], test_X[:, 1], c=test_Y, s=40, cmap=plt.cm.Spectral);
            plt.title("Input data Analysis ")  
            plt.figure(1).savefig('output/Fig.png')
            schVar=False            ,
            display_image('output/Fig.png',schVar,image_label)
            plt.show()
            train_X = train_X.T
            train_Y = train_Y.reshape((1, train_Y.shape[0]))  
            if dataset == "noisy_moons":
                train_Y = train_Y%2
    
    elif binaryclasT == True:
        if first_time == 0:
            first_time = 1
            train_X, train_Y, test_X, test_Y = interface.load_2D_dataset()
            print ('The shape of X_train is: ' + str(train_X.shape))
            print ('The shape of y_train is: ' + str(train_Y.shape))
            print ('The shape of X_test: ' + str(test_X.shape))
            print ('The shape of y_test is: ' + str(test_Y.shape))
            print ('We have m = %d training examples' % (len(train_Y)))
            plt.figure(1)
            # Plot examples
            #plt.scatter(train_X.T[:, 0], train_X.T[:, 1], c=train_Y.T, s=40, cmap=plt.cm.Spectral);
            #interface.plot_data(train_X, train_Y[:], pos_label="Accepted", neg_label="Rejected")

            plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral);


            # Set the y-axis label
            #plt.ylabel('Microchip Test 2') 
            # Set the x-axis label
            #plt.xlabel('Microchip Test 1') 
            #plt.legend(loc="upper right")
            plt.figure(1).savefig('output/Fig.png')
            schVar=False            
            display_image('output/Fig.png',schVar,image_label)            
            plt.show()       
            train_X = train_X
            train_Y = train_Y           
    elif binaryclasApp == True:
        if first_time == 0:
            first_time = 1
            train_X, train_Y = interface.load_data("support/ex2data2.txt")
            # print X_train
            print("X_train:", train_X[:5])
            print("Type of X_train:",type(train_X))
            # print y_train
            print("y_train:", train_Y[:5])
            print("Type of y_train:",type(train_Y))
            print ('The shape of X_train is: ' + str(train_X.shape))
            print ('The shape of y_train is: ' + str(train_Y.shape))
            print ('We have m = %d training examples' % (len(train_Y)))
            plt.figure(1)
            # Plot examples
            interface.plot_data(train_X, train_Y[:], pos_label="Accepted", neg_label="Rejected")

            # Set the y-axis label
            plt.ylabel('Microchip Test 2') 
            # Set the x-axis label
            plt.xlabel('Microchip Test 1') 
            plt.legend(loc="upper right")
            plt.figure(1).savefig('output/Fig.png')
            schVar=False            
            display_image('output/Fig.png',schVar,image_label)            
            plt.show()
            #print("Original shape of data:", train_X.shape)
            mapped_X =  interface.map_feature(train_X[:, 0], train_X[:, 1])
            #print("Shape after feature mapping:", mapped_X.shape)
            #print("X_train[0]:", train_X[0])
            #print("mapped X_train[0]:", mapped_X[0])            
            train_X = train_X.T
            train_Y = train_Y.reshape((1, train_Y.shape[0]))
    elif MulticlassApp == True:
        if first_time == 0:
            first_time = 1
            if 1==0:
                classes = 4
                m = 1000
                centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
                std = 1.0
                train_X, train_Y = make_blobs(n_samples=m, centers=centers, cluster_std=std,random_state=30)                        
                css = np.unique(train_Y)
                fig,ax = plt.subplots(1,1)

                interface.plt_mc_data(ax, train_X,train_Y,classes, legend=True, size=50, equal_xy = False)
                ax.set_title("Multiclass Data")
                ax.set_xlabel("x0")
                ax.set_ylabel("x1")
                #plt.show()    
                print(f"shape of train_X: {train_X.shape}, shape of train_Y: {train_Y.shape}")
            else:
                classes = 6
                m = 800
                std = 0.4
                centers = np.array([[-1, 0], [1, 0], [0, 1], [0, -1],  [-2,1],[-2,-1]])
                train_X, train_Y = make_blobs(n_samples=m, centers=centers, cluster_std=std, random_state=2, n_features=2)
                css = np.unique(train_Y)
                fig,ax = plt.subplots(1,1)

                interface.plt_mc_data(ax, train_X,train_Y,classes, legend=True, size=50, equal_xy = False)
                ax.set_title("Multiclass Data")
                ax.set_xlabel("x0")
                ax.set_ylabel("x1")
                #plt.show()    
                print(f"shape of train_X: {train_X.shape}, shape of train_Y: {train_Y.shape}")                
                
            fig.savefig('output/Fig.png')
            schVar=False            
            display_image('output/Fig.png',schVar,image_label)           

    else:        
        if first_time == 0:
            first_time = 1
            if SignalFitCSV == True: 
                plt.figure(1)            
                plt.plot(x1_feat)
                plt.figure(2)            
                plt.plot(x2_feat)
                plt.figure(3)            
                plt.plot(y_feat)
                train_Y=y_feat
                plt.grid(True)
                train_X = 0
                schVar=False
                plt.figure(3).savefig('output/Fig.png') 
                display_image('output/Fig.png',schVar,image_label)  
                plt.show()
            elif XGBoostFlag ==True:
                # split data into X and y
                X = df.iloc[:,:-1].values
                y = df.iloc[:,-1].values
                train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2, random_state=5)
                print(train_X.shape)
                print(train_Y.shape)
                print(test_X.shape)
                print(test_Y.shape)                
            
            elif signalFit1 == True:
                x = np.linspace(0, 2 * np.pi, 1000)  
                y = np.sin(x)
                train_X = x
                train_Y = y
                plt.figure(1)            
                plt.plot(x, y)
                plt.grid(True)
                plt.figure(1).savefig('output/Fig.png')
                schVar=False                
                display_image('output/Fig.png',schVar,image_label)                
                plt.show()                     
            elif signalFit2 == True:
                x = np.linspace(0, 2 * np.pi, 1000)
                #x = (x - x.mean())/x.std()
                #x = np.arange(0.0, 1, 0.001)
                y = np.matrix([np.sin(x),np.sin(x-2*np.pi/3),np.sin(x+2*np.pi/3)])
                train_X = x
                train_Y = y.T
                plt.figure(1)            
                plt.plot(x, y.T)
                plt.grid(True)
                plt.figure(1).savefig('output/Fig.png')
                schVar=False                
                display_image('output/Fig.png',schVar,image_label)                
                plt.show() 
            else:
                first_time = 0
        
    return train_X,train_Y
    #root.after(100, check_button)
        


def predictT():
    if FaceNetFlag == True:            

        min_dist = 100
        identity = ""
        detected  = False
        for face in range(len(faces)):
            person = faces[face]
            dist, detected = FaceNet.verify2(file_path, person, database[person], FRmodel, webcam=False)
            if detected == True and dist<min_dist:
                min_dist = dist
                identity = person               
        
                
        #dist, door_open  = verify(file_path, "kian", database, FRmodel)
        #print(min_dist)
        #print(identity)
        
        OutputTextExif.insert(tk.END, f"\nDistance: {(min_dist):.6f}")
        OutputTextExif.insert(tk.END, f"\nIdentity: {(identity)}") 
        
        """min_dist, identity  = FaceNet.who_is_it(file_path, database, FRmodel)
        
        #OutputTextExif.insert(tk.END, f"\nDistance: {(dist):.6f}")
        #OutputTextExif.insert(tk.END, f"\nDoor open:{(door_open)}")
        
        if min_dist > 0.7:
            OutputTextExif.insert(tk.END, f"\nNot in the database")
            OutputTextExif.insert(tk.END, f"\nDistance: {(min_dist):.6f}")
        else:            
            OutputTextExif.insert(tk.END, f"\nDistance: {(min_dist):.6f}")
            OutputTextExif.insert(tk.END, f"\nIdentity: {(identity)}") """       
        
    
    elif YoloPrediction == True:
        img = cv2.imread(file_path, 1)
        results = model(img, stream=True)
        colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype="uint8")                      
        # coordinates
        for r in results:
            boxes = r.boxes
            for box in boxes:
                print(box.conf[0])
                if box.conf[0] > 0.20:
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                    # confidence
                    confidence = math.ceil((box.conf[0]*100))/100
                    #print("Confidence --->",confidence)

                    # class name
                    cls = int(box.cls[0])
                    #print("Class name -->", classNames[cls])
                    
                    label = '{} {:.2f}'.format(classNames[cls], confidence)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    (text_width, text_height) = cv2.getTextSize(label, font, fontScale=fontScale, thickness=1)[0]
                    text_offset_x = x1
                    text_offset_y = y1 - 5
                    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                    color = colors[cls].tolist()

                    overlay = img.copy()
                    cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                    # add opacity (transparency to the box)
                    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

                    # object details
                    org = [x1, y1 - 5]
                    #color = (255, 0, 0)
                    thickness = 2
                    # put box in cam
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(img, label, org, font, fontScale=fontScale, color=(0, 0, 0), thickness=thickness)
            cv2.imwrite("output/PhotoYolo.jpg", img)
            schVar=False
            display_image('output/PhotoYolo.jpg',schVar,image_label) 
        
    elif AgePrediction == True:
        frame = cv2.imread(file_path, 1)              
        detector = dlib.get_frontal_face_detector()            
        #frame = imutils.resize(frameOrg, width = 800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
        faceRects = detector(gray, 1) # result
        print(faceRects)
        
        margin = 40
        FONT_SCALE = 0.8e-3  # Adjust for larger font size in all images
        THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images
        height, width, _ = frame.shape
        for face in faceRects:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())        
            left = x - margin // 2
            right = x + w + margin // 2
            bottom = y - margin // 2
            top = y + h + margin // 2            
            roi = frame[bottom:top,left:right]            
            #roi = frame[x:x+w,y:y+h]  
            #cv2.rectangle(frame, (left -1, bottom - 1), (right + 1 ,top + 1), (0, 0, 255), 2)
            #roi = frame[y:y+y1,x:x+x1]

            img = np.around(np.array(roi) / 255.0, decimals=12)
            img = cv2.resize(img, (224, 224))
            img2 = cv2.resize(roi, (224, 224))
            #cv2.imwrite("Photo1.jpg", img2)
            x_train = np.expand_dims(img, axis=0)        
            Age = AgePredictor.predictAge(faceage_vgg_model, x_train)    
            #label = 'Age: {:.2f}'.format(Age)
            label = 'Age:'
            label2 = '{:.2f}'.format(Age)
            print(colored(("Age = ", round(Age,2)),"green"))
            OutputTextExif.insert(tk.END, f"\nAge: {(Age):.2f}")        
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = min(width, height) * FONT_SCALE
            org = [x, y - 5]
            org2 = [x, y + h + 15]
            thickness = math.ceil(min(width, height) * THICKNESS_SCALE)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, org, font, fontScale=fontScale, color=(0, 0, 255), thickness=thickness, lineType=cv2.LINE_AA)
            cv2.putText(frame, label2, org2, font, fontScale=fontScale, color=(0, 0, 255), thickness=thickness, lineType=cv2.LINE_AA)
            cv2.imwrite("output/PhotoAge.jpg", frame)
            schVar=False
            display_image('output/PhotoAge.jpg',schVar,image_label) 
        
    else:    
        if ImageRecognitionSigns == True:        
            prediction = model.predict(imagec)
            a = (np.argmax(prediction))
            OutputTextExif.insert(tk.END, f"\nPrediction: {(a):}")
            print(a)    
        
        elif binaryclas == False and signalFit1 == False and signalFit2 == False and SignalFitCSV == False:            
            if ConvModelVar.get() == 1:            
                prediction = model.predict(imagec)
                a = (np.argmax(prediction))
                OutputTextExif.insert(tk.END, f"\nPrediction: {(a):}")
                print(a)
            else:                       
                prediction = DLLib.predict2(imagec,  parameters, activation = activationMVar, activationL = activationLVar)  
                print ("Prediction: " + str(np.squeeze(prediction)))   
                #OutputTextExif.insert(tk.END, f"Cost after iteration {}: {}".format(np.squeeze(costs))) 
                if ImageRecognitionH == False:
                    a = "Cat" if np.squeeze(prediction) > 0 else "No Cat"
                else:
                    a = "Dog" if np.squeeze(prediction) > 0 else "Cat"
                OutputTextExif.insert(tk.END, f"\nPrediction: {(a):}")
        else:
            if SignalFitCSV == True:
                #plt.figure(4)            
                #plt.plot(x, y)             
                x=train_x
                print(x.shape)
                print(train_Y.T)
                prediction = DLLib.predict3(x,  parameters, activation = activationMVar, activationL = activationLVar) 
                plt.figure(1) 
                plt.plot(np.squeeze(prediction))  
                print(prediction)            
                plt.figure(2) 
                plt.plot(np.squeeze(prediction)-np.squeeze(train_Y.T))
                plt.figure(3)
                plt.plot(np.squeeze(prediction))
                plt.plot(np.squeeze(train_Y.T))            
                plt.grid(True)
                plt.show()     
            elif signalFit1 == True:
                x = np.linspace(0, 2 * np.pi, 1000)
                y = np.sin(x)
                plt.figure(4)            
                plt.plot(x, y)             
                x=train_x
                prediction = DLLib.predict3(x,  parameters, activation = activationMVar, activationL = activationLVar) 
                plt.plot(np.squeeze(x), np.squeeze(prediction))           
                plt.plot(np.squeeze(x), np.squeeze(prediction)-train_Y)
                plt.grid(True)
                plt.show() 
            else:
                if signalFit2 == True:
                    x = np.linspace(0, 2 * np.pi, 1000)
                    y = np.matrix([np.sin(x),np.sin(x-2*np.pi/3),np.sin(x+2*np.pi/3)])
                    plt.figure(4)            
                    plt.plot(x, y.T)             
                    x=train_x
                    prediction = DLLib.predict3(x,  parameters, activation = activationMVar, activationL = activationLVar)                          
                    plt.plot(np.squeeze(x), np.squeeze(prediction).T)           
                    #plt.plot(np.squeeze(x), np.squeeze(prediction)-train_Y)
                    plt.grid(True)
                    plt.show()
                else:                
                    x = np.matrix([[-1.07, 1.07], [-0.044, -0.044]])
                    #print(x)
                    #print(x.shape[1])
                    prediction = DLLib.predict2(x,  parameters, activation = activationMVar, activationL = activationLVar)
                    OutputTextExif.insert(tk.END, f"\nPrediction: {(np.squeeze(prediction)):}")



def minibatch_train(train_batch_size,seed,hdf5_path):
    
    #hdf5_path = 'support/cats_dogs/cats_dogs_128.hdf5'
    print(hdf5_path)    
    dataset = h5py.File(hdf5_path, "r")    
    np.random.seed(seed)    
    # shuffle indexes,int numbers range from 0 to 20000
    #permutation = list(np.random.permutation(2000))
    permutation = list(np.random.permutation(20000))   
    # get the "train_batch_size" indexes  
    #np.set_printoptions(threshold=sys.maxsize)    
    train_batch_index=permutation[0:train_batch_size]
    #print(train_batch_index)
    
    # the shape of "train_labels" now is (20000,1)
    train_labels=np.array(dataset["train_labels"]).reshape(20000,-1)
    #print(train_labels.T)
    # get the corresponding labels according "train_batch_index"
    train_batch_labels=train_labels[train_batch_index]   
   # train_batch_labels= np.eye(2)[train_batch_labels.reshape(-1)] #convert to one_hot code    
    train_batch_imgs=[]
    for i in range(train_batch_size):
        img=(dataset['train_img'])[train_batch_index[i]]
        img=img/255.
        train_batch_imgs.append(img)    
    train_batch_imgs=np.array(train_batch_imgs)    
    dataset.close()
    
    return(train_batch_imgs,train_batch_labels)

def minibatch_test(test_batch_size,seed,hdf5_path): 
     
    #hdf5_path = 'support/cats_dogs/cats_dogs_128.hdf5'
    dataset = h5py.File(hdf5_path, "r")    
    np.random.seed(seed)
    #np.set_printoptions(threshold=sys.maxsize)
    permutation = list(np.random.permutation(5000))        
    test_batch_index= permutation[0:test_batch_size]  
    #print("test_batch_index", test_batch_index)
    test_labels= np.array(dataset["test_labels"]).reshape(5000,-1)
    #print("Test_labels1:" ,test_labels.T)
    test_batch_labels= test_labels[test_batch_index]
    #print("Test_labels2:" ,test_batch_labels.T)
#test_batch_labels= np.eye(2)[test_batch_labels.reshape(-1)]    
    test_batch_imgs=[]
    for i in range(test_batch_size):
        img=(dataset['test_img'])[test_batch_index[i]]
        img=img/255.
        test_batch_imgs.append(img)    
    test_batch_imgs=np.array(test_batch_imgs)    
    dataset.close()  
    
    return(test_batch_imgs,test_batch_labels)


def DataAugmentedFunction():

    # Passing the path of the image directory
    p = Augmentor.Pipeline("CatDogs")
    print(p)
     
    # Defining augmentation parameters and generating 5 samples
    p.flip_left_right(0.5)
    p.black_and_white(0.1)
    p.rotate(0.3, 10, 10)
    p.skew(0.4, 0.5)
    p.zoom(probability = 0.2, min_factor = 1.1, max_factor = 1.5)
    p.sample(5)


def C1():
    filename = 'file:///'+os.getcwd()+'/' + 'doc/C1.html'
    webbrowser.open_new_tab(filename)

def C2():
    filename = 'file:///'+os.getcwd()+'/' + 'doc/C2.html'
    webbrowser.open_new_tab(filename)
    
def C3():
    filename = 'file:///'+os.getcwd()+'/' + 'doc/C3.html'
    webbrowser.open_new_tab(filename)
  
def C4():
    filename = 'file:///'+os.getcwd()+'/' + 'doc/C4.html'
    webbrowser.open_new_tab(filename)

def C5():
    filename = 'file:///'+os.getcwd()+'/' + 'doc/C5.html'
    webbrowser.open_new_tab(filename)

def BinaryClass(): 
    global train_X,train_Y
    global signalFit1, signalFit2, SignalFitCSV,binaryclas
    global ImageRecognitionH, ImageRecognitionSigns, NeuralStyleTransferFlag, RNNFlag
    global FaceNetFlag, AgePrediction, YoloPrediction, binaryclasApp, binaryclasT, MulticlassApp
    global keep_prob_list, XGBoostFlag
    
    lambDAVar_entry.config(state="normal")
    Beta_entry.config(state="normal")
    LayerDim_entry.config(state="normal")
    TrainSize_entry.config(state="disabled")
    Testsize_entry.config(state="disabled")
    TimeInter_entry.config(state="normal")
    CheckLDecay.config(state="normal")
    CheckDataAugmented.config(state="disabled")
    ConvModel.config(state="disabled")
    CheckB.config(state="normal") 
    ConvModelResnet.config(state="disabled") 
    radioIniZeros.config(state="normal") 
    radioIniRandom.config(state="normal") 
    radioIniHe.config(state="normal") 
    radioOptGD.config(state="normal") 
    radioOptMom.config(state="normal") 
    radioOptAdam.config(state="normal") 
    radioRelu.config(state="normal") 
    radioTanh.config(state="normal") 
    radioSigmoid.config(state="normal") 
    radioLinear.config(state="normal") 
    radioBinaryCross.config(state="normal") 
    radioMSE.config(state="normal") 
    Lrate_entry.config(state="normal")     
    Minibatch_entry.config(state="normal")
    niterations_entry.config(state="normal")     
    num_iterations.set(1000)
    LearningRate.set(1.2)   
    plt.close()
    first_time = 0
    binaryclas = True
    binaryclasApp = False 
    MulticlassApp = False    
    binaryclasT = False       
    ImageRecognitionH = False
    ImageRecognitionSigns = False
    signalFit1 = False
    signalFit2 = False
    NeuralStyleTransferFlag = False
    SignalFitCSV = False 
    RNNFlag = False
    FaceNetFlag = False
    AgePrediction = False
    YoloPrediction = False
    XGBoostFlag = False
    FileLayers = False
    ActivationM.set(2)
    ActivationL.set(1)
    CostFunction.set(1)
    ConvModelVar.set(2)
    LayerDim.set("2 4 1")
    keep_prob_list = [1.0]
    train_X,train_Y = check_button(first_time)
    
def BinaryClassComparison(): 
    global train_X,train_Y
    global signalFit1, signalFit2, SignalFitCSV,binaryclas
    global ImageRecognitionH, ImageRecognitionSigns, NeuralStyleTransferFlag, RNNFlag
    global FaceNetFlag, AgePrediction, YoloPrediction, binaryclasApp, binaryclasT, MulticlassApp
    global keep_prob_list, XGBoostFlag, BinaryClassComparison
    
    lambDAVar_entry.config(state="normal")
    Beta_entry.config(state="normal")
    LayerDim_entry.config(state="normal")
    TrainSize_entry.config(state="disabled")
    Testsize_entry.config(state="disabled")
    TimeInter_entry.config(state="normal")
    CheckLDecay.config(state="normal")
    CheckDataAugmented.config(state="disabled")
    ConvModel.config(state="disabled")
    CheckB.config(state="normal") 
    ConvModelResnet.config(state="disabled") 
    radioIniZeros.config(state="normal") 
    radioIniRandom.config(state="normal") 
    radioIniHe.config(state="normal") 
    radioOptGD.config(state="normal") 
    radioOptMom.config(state="normal") 
    radioOptAdam.config(state="normal") 
    radioRelu.config(state="normal") 
    radioTanh.config(state="normal") 
    radioSigmoid.config(state="normal") 
    radioLinear.config(state="normal") 
    radioBinaryCross.config(state="normal") 
    radioMSE.config(state="normal") 
    Lrate_entry.config(state="normal")     
    Minibatch_entry.config(state="normal")
    niterations_entry.config(state="normal")     
    num_iterations.set(1000)
    LearningRate.set(1.2)   
    plt.close()
    first_time = 0
    binaryclas = False
    BinaryClassComparison = True
    binaryclasApp = False 
    MulticlassApp = False    
    binaryclasT = False       
    ImageRecognitionH = False
    ImageRecognitionSigns = False
    signalFit1 = False
    signalFit2 = False
    NeuralStyleTransferFlag = False
    SignalFitCSV = False 
    RNNFlag = False
    FaceNetFlag = False
    AgePrediction = False
    YoloPrediction = False
    XGBoostFlag = False
    FileLayers = False
    ActivationM.set(2)
    ActivationL.set(1)
    CostFunction.set(1)
    ConvModelVar.set(2)
    LayerDim.set("2 4 1")
    keep_prob_list = [1.0]
    train_X,train_Y = check_button(first_time)    
    
    
    
def BinaryClassT(): 
    global train_X,train_Y, test_X, test_Y
    global signalFit1, signalFit2, SignalFitCSV,binaryclas
    global ImageRecognitionH, ImageRecognitionSigns, NeuralStyleTransferFlag, RNNFlag
    global FaceNetFlag, AgePrediction, YoloPrediction, binaryclasApp, binaryclasT
    global keep_prob_list, XGBoostFlag
    
    lambDAVar_entry.config(state="normal")
    Beta_entry.config(state="normal")
    LayerDim_entry.config(state="normal")
    TrainSize_entry.config(state="disabled")
    Testsize_entry.config(state="disabled")
    TimeInter_entry.config(state="normal")
    CheckLDecay.config(state="normal")
    CheckDataAugmented.config(state="disabled")
    ConvModel.config(state="disabled")
    CheckB.config(state="normal") 
    ConvModelResnet.config(state="disabled") 
    radioIniZeros.config(state="normal") 
    radioIniRandom.config(state="normal") 
    radioIniHe.config(state="normal") 
    radioOptGD.config(state="normal") 
    radioOptMom.config(state="normal") 
    radioOptAdam.config(state="normal") 
    radioRelu.config(state="normal") 
    radioTanh.config(state="normal") 
    radioSigmoid.config(state="normal") 
    radioLinear.config(state="normal") 
    radioBinaryCross.config(state="normal") 
    radioMSE.config(state="normal") 
    Lrate_entry.config(state="normal")     
    Minibatch_entry.config(state="normal")
    niterations_entry.config(state="normal")     
    num_iterations.set(30000)
    LearningRate.set(0.3)   
    plt.close()
    first_time = 0
    binaryclas = False
    binaryclasT = True
    binaryclasApp = False 
    MulticlassApp = False
    ImageRecognitionH = False
    ImageRecognitionSigns = False
    signalFit1 = False
    signalFit2 = False
    NeuralStyleTransferFlag = False
    SignalFitCSV = False 
    XGBoostFlag = False
    RNNFlag = False
    FaceNetFlag = False
    AgePrediction = False
    YoloPrediction = False
    FileLayers = False
    ActivationM.set(1)
    ActivationL.set(1)
    CostFunction.set(1)
    ConvModelVar.set(2)
    IniWeights.set(2)
    LayerDim.set("2 20 3 1")
    keep_prob_list = [1.0, 1.0]    
    train_X,train_Y = check_button(first_time)    
    
def BinaryClassApp(): 
    global train_X,train_Y
    global signalFit1, signalFit2, SignalFitCSV,binaryclas
    global ImageRecognitionH, ImageRecognitionSigns, NeuralStyleTransferFlag, RNNFlag
    global FaceNetFlag, AgePrediction, YoloPrediction, binaryclasApp, binaryclasT, MulticlassApp
    global keep_prob_list, XGBoostFlag
    
    lambDAVar_entry.config(state="normal")
    Beta_entry.config(state="normal")
    LayerDim_entry.config(state="normal")
    TrainSize_entry.config(state="disabled")
    Testsize_entry.config(state="disabled")
    TimeInter_entry.config(state="normal")
    CheckLDecay.config(state="normal")
    CheckDataAugmented.config(state="disabled")
    ConvModel.config(state="disabled")
    CheckB.config(state="normal") 
    ConvModelResnet.config(state="disabled") 
    radioIniZeros.config(state="normal") 
    radioIniRandom.config(state="normal") 
    radioIniHe.config(state="normal") 
    radioOptGD.config(state="normal") 
    radioOptMom.config(state="normal") 
    radioOptAdam.config(state="normal") 
    radioRelu.config(state="normal") 
    radioTanh.config(state="normal") 
    radioSigmoid.config(state="normal") 
    radioLinear.config(state="normal") 
    radioBinaryCross.config(state="normal") 
    radioMSE.config(state="normal") 
    Lrate_entry.config(state="normal")     
    Minibatch_entry.config(state="normal")
    niterations_entry.config(state="normal")     
    num_iterations.set(10000)
    LearningRate.set(0.1)
    lambDAVar.set(0.05)
    plt.close()
    first_time = 0
    binaryclas = False  
    binaryclasApp = True
    binaryclasT = False       
    ImageRecognitionH = False
    ImageRecognitionSigns = False
    signalFit1 = False
    signalFit2 = False
    XGBoostFlag = False
    MulticlassApp = False    
    NeuralStyleTransferFlag = False
    SignalFitCSV = False 
    RNNFlag = False
    FaceNetFlag = False
    AgePrediction = False
    YoloPrediction = False
    FileLayers = False
    ActivationM.set(2)
    ActivationL.set(1)
    CostFunction.set(1)
    ConvModelVar.set(2)
    LayerDim.set("2 4 1")
    keep_prob_list = [1.0]     
    train_X,train_Y = check_button(first_time)

    #t_X, parameters = DLLib.forward_propagation_with_dropout_test_case()
    #keep_prob_temp = 0.7
    #A3, cache, D = DLLib.L_model_forward(t_X, keep_prob_temp, parameters,  activation = "relu", activationL = "sigmoid")
    #print ("A3 = " + str(A3))
    #print (D)
    
    #t_X, t_Y, cache = DLLib.backward_propagation_with_dropout_test_case()
    #keep_prob = 0.8
    #cost_function = "mse"
    #lambd = 0.0
    #gradients = DLLib.L_model_backward(t_X, t_Y, cache, Dx, cost_function, lambd, keep_prob, activation = "relu", activationL = "sigmoid")

    #print ("dA1 = \n" + str(gradients["dA1"]))
    #print ("dA2 = \n" + str(gradients["dA2"]))    
    
    
def MultClassApp(): 
    global train_X,train_Y
    global signalFit1, signalFit2, SignalFitCSV,binaryclas
    global ImageRecognitionH, ImageRecognitionSigns, NeuralStyleTransferFlag, RNNFlag
    global FaceNetFlag, AgePrediction, YoloPrediction, binaryclasApp, binaryclasT
    global keep_prob_list, MulticlassApp, XGBoostFlag
    
    lambDAVar_entry.config(state="normal")
    Beta_entry.config(state="normal")
    LayerDim_entry.config(state="normal")
    TrainSize_entry.config(state="disabled")
    Testsize_entry.config(state="disabled")
    TimeInter_entry.config(state="normal")
    CheckLDecay.config(state="normal")
    CheckDataAugmented.config(state="disabled")
    ConvModel.config(state="disabled")
    CheckB.config(state="normal") 
    ConvModelResnet.config(state="disabled") 
    radioIniZeros.config(state="normal") 
    radioIniRandom.config(state="normal") 
    radioIniHe.config(state="normal") 
    radioOptGD.config(state="normal") 
    radioOptMom.config(state="normal") 
    radioOptAdam.config(state="normal") 
    radioRelu.config(state="normal") 
    radioTanh.config(state="normal") 
    radioSigmoid.config(state="normal") 
    radioLinear.config(state="normal") 
    radioBinaryCross.config(state="normal") 
    radioMSE.config(state="normal") 
    Lrate_entry.config(state="normal")     
    Minibatch_entry.config(state="normal")
    niterations_entry.config(state="normal")     
    num_iterations.set(1000)
    LearningRate.set(0.01)
    lambDAVar.set(0.1)
    plt.close()
    first_time = 0
    binaryclas = False  
    binaryclasApp = False   
    MulticlassApp = True
    binaryclasT = False       
    ImageRecognitionH = False
    ImageRecognitionSigns = False
    signalFit1 = False
    signalFit2 = False
    NeuralStyleTransferFlag = False
    SignalFitCSV = False 
    XGBoostFlag = False
    RNNFlag = False
    FaceNetFlag = False
    AgePrediction = False
    YoloPrediction = False
    FileLayers = False
    ActivationM.set(1)
    ActivationL.set(2)
    CostFunction.set(1)
    ConvModelVar.set(2)
    LayerDim.set("2 4 4")
    keep_prob_list = [1.0]     
    train_X,train_Y = check_button(first_time)
    

def SignalFit1():
    global signalFit1, signalFit2, SignalFitCSV,binaryclas
    global ImageRecognitionH, ImageRecognitionSigns, NeuralStyleTransferFlag, RNNFlag
    global FaceNetFlag, AgePrediction, YoloPrediction, binaryclasApp, binaryclasT
    global keep_prob_list, MulticlassApp, XGBoostFlag
    
    lambDAVar_entry.config(state="normal")
    Beta_entry.config(state="normal")
    LayerDim_entry.config(state="normal")
    TrainSize_entry.config(state="disabled")
    Testsize_entry.config(state="disabled")
    TimeInter_entry.config(state="normal")
    CheckLDecay.config(state="normal")
    CheckDataAugmented.config(state="disabled")
    ConvModel.config(state="disabled")
    CheckB.config(state="normal") 
    ConvModelResnet.config(state="disabled") 
    radioIniZeros.config(state="normal") 
    radioIniRandom.config(state="normal") 
    radioIniHe.config(state="normal") 
    radioOptGD.config(state="normal") 
    radioOptMom.config(state="normal") 
    radioOptAdam.config(state="normal") 
    radioRelu.config(state="normal") 
    radioTanh.config(state="normal") 
    radioSigmoid.config(state="normal") 
    radioLinear.config(state="normal") 
    radioBinaryCross.config(state="normal") 
    radioMSE.config(state="normal") 
    Minibatch_entry.config(state="normal") 
    Lrate_entry.config(state="normal")
    niterations_entry.config(state="normal") 
    num_iterations.set(5000)
    LearningRate.set(0.0075)     
    plt.close()
    first_time = 0
    ImageRecognitionH = False
    ImageRecognitionSigns = False
    binaryclas = False
    MulticlassApp = False    
    binaryclasApp = False 
    XGBoostFlag = False
    binaryclasT = False       
    signalFit1 = True
    signalFit2 = False
    SignalFitCSV = False  
    NeuralStyleTransferFlag = False   
    RNNFlag = False  
    FaceNetFlag = False
    AgePrediction = False   
    YoloPrediction = False 
    FileLayers = False    
    ActivationM.set(2)
    ActivationL.set(2)
    CostFunction.set(2)
    ConvModelVar.set(2)    
    LayerDim.set("1 5 1")
    keep_prob_list = [1.0]     
    check_button(first_time)
    #LayerDim_entry['textvariable'] = LayerDim    

def SignalFit2():     
    global signalFit1, signalFit2, SignalFitCSV,binaryclas
    global ImageRecognitionH, ImageRecognitionSigns, NeuralStyleTransferFlag, RNNFlag
    global FaceNetFlag, AgePrediction, YoloPrediction, binaryclasApp, binaryclasT
    global keep_prob_list, MulticlassApp, XGBoostFlag
    
    lambDAVar_entry.config(state="normal")
    Beta_entry.config(state="normal")
    LayerDim_entry.config(state="normal")
    TrainSize_entry.config(state="disabled")
    Testsize_entry.config(state="disabled")
    TimeInter_entry.config(state="normal")
    CheckLDecay.config(state="normal")
    CheckDataAugmented.config(state="disabled")
    ConvModel.config(state="disabled")
    CheckB.config(state="normal") 
    ConvModelResnet.config(state="disabled") 
    radioIniZeros.config(state="normal") 
    radioIniRandom.config(state="normal") 
    radioIniHe.config(state="normal") 
    radioOptGD.config(state="normal") 
    radioOptMom.config(state="normal") 
    radioOptAdam.config(state="normal") 
    radioRelu.config(state="normal") 
    radioTanh.config(state="normal") 
    radioSigmoid.config(state="normal") 
    radioLinear.config(state="normal") 
    radioBinaryCross.config(state="normal") 
    radioMSE.config(state="normal") 
    Minibatch_entry.config(state="normal") 
    Lrate_entry.config(state="normal") 
    niterations_entry.config(state="normal")     
    num_iterations.set(5000)
    LearningRate.set(0.0075)     
    plt.close()
    first_time = 0
    ImageRecognitionH = False
    ImageRecognitionSigns = False
    binaryclas = False
    binaryclasApp = False   
    MulticlassApp = False    
    binaryclasT = False   
    signalFit1 = False
    XGBoostFlag = False
    SignalFitCSV = False
    NeuralStyleTransferFlag = False
    RNNFlag = False
    signalFit2 = True
    FaceNetFlag = False    
    AgePrediction = False 
    YoloPrediction = False  
    FileLayers = False          
    ActivationM.set(2)
    ActivationL.set(2)
    CostFunction.set(2)
    ConvModelVar.set(2)    
    LayerDim.set("1 5 3") 
    keep_prob_list = [1.0]
    #LayerDim_entry['textvariable'] = LayerDim
    print(LayerDim)
    check_button(first_time)
    
def SignalFitCSV():     
    global signalFit1, signalFit2, SignalFitCSV,binaryclas
    global ImageRecognitionH, ImageRecognitionSigns, NeuralStyleTransferFlag, RNNFlag
    global FaceNetFlag, AgePrediction, YoloPrediction, binaryclasApp, binaryclasT
    global x1_feat, x2_feat, y_feat
    global keep_prob_list, MulticlassApp, XGBoostFlag
    
    lambDAVar_entry.config(state="normal")
    Beta_entry.config(state="normal")
    LayerDim_entry.config(state="normal")
    TrainSize_entry.config(state="disabled")
    Testsize_entry.config(state="disabled")
    TimeInter_entry.config(state="normal")
    CheckLDecay.config(state="normal")
    CheckDataAugmented.config(state="disabled")
    ConvModel.config(state="disabled")
    CheckB.config(state="normal") 
    ConvModelResnet.config(state="disabled") 
    radioIniZeros.config(state="normal") 
    radioIniRandom.config(state="normal") 
    radioIniHe.config(state="normal") 
    radioOptGD.config(state="normal") 
    radioOptMom.config(state="normal") 
    radioOptAdam.config(state="normal") 
    radioRelu.config(state="normal") 
    radioTanh.config(state="normal") 
    radioSigmoid.config(state="normal") 
    radioLinear.config(state="normal") 
    radioBinaryCross.config(state="normal") 
    radioMSE.config(state="normal") 
    Minibatch_entry.config(state="normal")
    Lrate_entry.config(state="normal")
    niterations_entry.config(state="normal")     
    num_iterations.set(50)    
    LearningRate.set(0.0005)         
    plt.close()
    first_time = 0
    ImageRecognitionH = False
    ImageRecognitionSigns = False
    binaryclas = False
    binaryclasApp = False 
    binaryclasT = False    
    MulticlassApp = False
    signalFit1 = False
    signalFit2 = False
    SignalFitCSV = True
    XGBoostFlag = False
    NeuralStyleTransferFlag = False
    RNNFlag = False
    FaceNetFlag = False
    AgePrediction = False
    YoloPrediction = False
    FileLayers = False    
    ActivationM.set(2)
    ActivationL.set(2)
    CostFunction.set(2)
    ConvModelVar.set(2)    
    LayerDim.set("2 10 10 1")  
    keep_prob_list = [1.0, 1.0] 
    #LayerDim_entry['textvariable'] = LayerDim
    print(LayerDim)
    x1_feat, x2_feat, y_feat, file_pathCSV = interface.open_datasetTrainingCSV()
    OutputText.delete("1.0", "end")
    OutputTextExif.insert(tk.END, f"Training CSV File: \n{(file_pathCSV):}")  
    check_button(first_time)    
    
def XGBoost():
    global train_X,train_Y     
    global signalFit1, signalFit2, SignalFitCSV,binaryclas
    global ImageRecognitionH, ImageRecognitionSigns, NeuralStyleTransferFlag, RNNFlag
    global FaceNetFlag, AgePrediction, YoloPrediction, binaryclasApp, binaryclasT, XGBoostFlag
    global df
    global keep_prob_list, MulticlassApp
    
    lambDAVar_entry.config(state="normal")
    Beta_entry.config(state="normal")
    LayerDim_entry.config(state="normal")
    TrainSize_entry.config(state="disabled")
    Testsize_entry.config(state="disabled")
    TimeInter_entry.config(state="normal")
    CheckLDecay.config(state="normal")
    CheckDataAugmented.config(state="disabled")
    ConvModel.config(state="disabled")
    CheckB.config(state="normal") 
    ConvModelResnet.config(state="disabled") 
    radioIniZeros.config(state="normal") 
    radioIniRandom.config(state="normal") 
    radioIniHe.config(state="normal") 
    radioOptGD.config(state="normal") 
    radioOptMom.config(state="normal") 
    radioOptAdam.config(state="normal") 
    radioRelu.config(state="normal") 
    radioTanh.config(state="normal") 
    radioSigmoid.config(state="normal") 
    radioLinear.config(state="normal") 
    radioBinaryCross.config(state="normal") 
    radioMSE.config(state="normal") 
    Minibatch_entry.config(state="normal")
    Lrate_entry.config(state="normal")
    niterations_entry.config(state="normal")     
    num_iterations.set(50)    
    LearningRate.set(0.1)         
    plt.close()
    first_time = 0
    ImageRecognitionH = False
    ImageRecognitionSigns = False
    binaryclas = False
    binaryclasApp = False 
    binaryclasT = False    
    MulticlassApp = False
    signalFit1 = False
    signalFit2 = False
    SignalFitCSV = False
    XGBoostFlag = True
    NeuralStyleTransferFlag = False
    RNNFlag = False
    FaceNetFlag = False
    AgePrediction = False
    YoloPrediction = False
    FileLayers = False    
    ActivationM.set(2)
    ActivationL.set(2)
    CostFunction.set(2)
    ConvModelVar.set(2)    
    LayerDim.set("2 10 10 1")  
    keep_prob_list = [1.0, 1.0] 
    schVar=False    
    display_image('support/ML/nn2.png',schVar,image_label)  
    print(LayerDim)
    df = pd.read_csv('support/datasets_228_482_diabetes.csv')
    print(df.head())
    train_X,train_Y = check_button(first_time)        

def ImageRecognition_64px(): 
    global signalFit1, signalFit2, SignalFitCSV,binaryclas
    global ImageRecognitionH, ImageRecognitionSigns, NeuralStyleTransferFlag, RNNFlag
    global FaceNetFlag, AgePrediction, YoloPrediction, binaryclasApp, binaryclasT
    global keep_prob_list, MulticlassApp, XGBoostFlag
    
    lambDAVar_entry.config(state="normal")
    Beta_entry.config(state="normal")
    LayerDim_entry.config(state="normal")
    TrainSize_entry.config(state="normal")
    Testsize_entry.config(state="normal")
    TimeInter_entry.config(state="normal")
    CheckLDecay.config(state="normal")
    CheckDataAugmented.config(state="normal")
    ConvModel.config(state="disabled")
    CheckB.config(state="normal") 
    ConvModelResnet.config(state="disabled") 
    radioIniZeros.config(state="normal") 
    radioIniRandom.config(state="normal") 
    radioIniHe.config(state="normal") 
    radioOptGD.config(state="normal") 
    radioOptMom.config(state="normal") 
    radioOptAdam.config(state="normal") 
    radioRelu.config(state="normal") 
    radioTanh.config(state="normal") 
    radioSigmoid.config(state="normal") 
    radioLinear.config(state="normal") 
    radioBinaryCross.config(state="normal") 
    radioMSE.config(state="normal") 
    Lrate_entry.config(state="normal")     
    Minibatch_entry.config(state="normal")
    niterations_entry.config(state="normal")     
    num_iterations.set(1000)
    LearningRate.set(0.0075)  
    plt.close()
    first_time = 0
    ImageRecognitionH = False
    ImageRecognitionSigns = False
    binaryclas = False
    binaryclasApp = False 
    binaryclasT = False       
    signalFit1 = False
    MulticlassApp = False    
    signalFit2 = False
    NeuralStyleTransferFlag = False
    RNNFlag = False
    SignalFitCSV = False 
    XGBoostFlag = False
    FaceNetFlag = False  
    AgePrediction = False
    YoloPrediction = False
    FileLayers = False    
    LayerDim.set("12288 20 7 5 1")
    keep_prob_list = [1.0, 1.0, 1.0, 1.0]  
    #LayerDim_entry['textvariable'] = LayerDim
    ActivationM.set(1)
    ActivationL.set(1)
    CostFunction.set(1)
    ConvModelVar.set(2)
    file_pathT = interface.open_datasetTraining()
    file_pathD = interface.open_datasetTest()
    OutputText.delete("1.0", "end")
    OutputTextExif.insert(tk.END, f"Training File: \n{(file_pathT):}")
    OutputTextExif.insert(tk.END, f"\nTest File: \n{(file_pathD):}")      
    set_data(file_pathT,file_pathD)
    #train_X,train_Y = check_button(first_time)

def ImageRecognition_128px(): 
    global signalFit1, signalFit2, SignalFitCSV,binaryclas
    global ImageRecognitionH, ImageRecognitionSigns, NeuralStyleTransferFlag, RNNFlag
    global FaceNetFlag, AgePrediction, YoloPrediction, binaryclasApp, binaryclasT
    global train_x, test_x, train_y, test_y, train_x_orig, test_x_orig
    global keep_prob_list, MulticlassApp, XGBoostFlag
    
    lambDAVar_entry.config(state="normal")
    Beta_entry.config(state="normal")
    LayerDim_entry.config(state="normal")
    TrainSize_entry.config(state="normal")
    Testsize_entry.config(state="normal")
    TimeInter_entry.config(state="normal")
    CheckLDecay.config(state="normal")
    CheckDataAugmented.config(state="normal")
    ConvModel.config(state="normal")
    CheckB.config(state="normal") 
    ConvModelResnet.config(state="normal") 
    radioIniZeros.config(state="normal") 
    radioIniRandom.config(state="normal") 
    radioIniHe.config(state="normal") 
    radioOptGD.config(state="normal") 
    radioOptMom.config(state="normal") 
    radioOptAdam.config(state="normal") 
    radioRelu.config(state="normal") 
    radioTanh.config(state="normal") 
    radioSigmoid.config(state="normal") 
    radioLinear.config(state="normal") 
    radioBinaryCross.config(state="normal") 
    radioMSE.config(state="normal") 
    Lrate_entry.config(state="normal")     
    Minibatch_entry.config(state="normal")  
    niterations_entry.config(state="normal") 
    num_iterations.set(100)
    LearningRate.set(0.0075)     
    plt.close()
    first_time = 0
    ImageRecognitionH = True
    ImageRecognitionSigns = False
    binaryclas = False
    binaryclasApp = False 
    binaryclasT = False       
    signalFit1 = False
    signalFit2 = False
    XGBoostFlag = False
    MulticlassApp = False    
    NeuralStyleTransferFlag = False
    RNNFlag = False 
    SignalFitCSV = False  
    FaceNetFlag = False
    AgePrediction = False
    YoloPrediction = False  
    FileLayers = False
    LayerDim.set("49152 40 10 5 1")
    keep_prob_list = [1.0, 1.0, 1.0, 1.0]      
    #LayerDim_entry['textvariable'] = LayerDim
    ActivationM.set(1)
    ActivationL.set(1)
    CostFunction.set(1)
    ConvModelVar.set(2)
    
    if DataAugmented.get():
        DataAugmentedFunction()
    
    
    # Processes input images
    folder = "data"

    #train_batch_size=128   # minibatch for training
    #test_batch_size=500    # minibatch for test
    
    
    train_batch_size=int(TrainSizeVar.get())
    test_batch_size=int(TestSizeVar.get())
    #print(train_batch_size)  
    #print(test_batch_size)
    
    seed=1                 # seed for shuffle training images,see cats_dogs_batch.py
    # get the training images and labels 
    hdf5_path = interface.open_dataset()  
    OutputText.delete("1.0", "end")
    OutputTextExif.insert(tk.END, f"Training/Test File: \n{(hdf5_path):}")
    (train_x_orig,train_y)=minibatch_train(train_batch_size,seed,hdf5_path)  
    (test_x_orig,test_y)=minibatch_test(test_batch_size,seed,hdf5_path)
    
    train_y = train_y.T
    test_y = test_y.T
    
    #print(train_y)
    #Explore your dataset
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]

    
    print(colored(("Number of training examples: " + str(m_train)),"red"))
    print(colored(("Number of testing examples: " + str(m_test)),"red"))
    print(colored(("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)"),"red"))
    print("train_x_orig shape: " + str(train_x_orig.shape))
    print("train_y shape: " + str(train_y.shape))
    print(colored(("test_x_orig shape: " + str(test_x_orig.shape)),"green"))
    print(colored(("test_y shape: " + str(test_y.shape)), "green"))
    
    OutputTextExif.insert(tk.END, f"\nN training examples: {(m_train):}") 
    OutputTextExif.insert(tk.END, f"\nN test examples: {(m_test):}") 

    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/1.
    test_x = test_x_flatten/1.
    #print(train_x)
    
    print ("train_x's shape: " + str(train_x.shape))
    print ("train_y's shape: " + str(train_y.shape))
    print ("test_x's shape: " + str(test_x.shape))
    #index = 1
    #plt.imshow(train_x_orig[index])
    #plt.show() 
    if 1==1:
        images_iter = iter(train_x_orig)
        labels_iter = iter(train_y)
        plt.figure(figsize=(10, 10))              
        plt.title("Sample of 25 pictures")
        for i in range(25):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(train_x_orig[i])
            #im=Image.fromarray(train_x_orig[i])
            #im.save("cat"+ str(i) + ".jpg")
            #plt.imshow(next(images_iter).numpy().astype("uint8"))            
            #plt.title(train_y[i])            
            plt.axis("off")

        plt.savefig('output/Fig.png')
        schVar=False                
        display_image('output/Fig.png',schVar,image_label)              
    
def ImageRecognitionSigns():
    global signalFit1, signalFit2, SignalFitCSV,binaryclas
    global ImageRecognitionH, ImageRecognitionSigns, NeuralStyleTransferFlag, RNNFlag
    global FaceNetFlag, AgePrediction, YoloPrediction, binaryclasApp, binaryclasT, MulticlassApp, XGBoostFlag
    
    lambDAVar_entry.config(state="disabled")
    Beta_entry.config(state="disabled")
    LayerDim_entry.config(state="disabled")
    TrainSize_entry.config(state="disabled")
    Testsize_entry.config(state="disabled")
    TimeInter_entry.config(state="disabled")
    CheckLDecay.config(state="disabled")
    CheckDataAugmented.config(state="normal")
    ConvModel.config(state="normal")
    CheckB.config(state="disabled") 
    ConvModelResnet.config(state="normal") 
    radioIniZeros.config(state="disabled") 
    radioIniRandom.config(state="disabled") 
    radioIniHe.config(state="disabled") 
    radioOptGD.config(state="disabled") 
    radioOptMom.config(state="disabled") 
    radioOptAdam.config(state="disabled") 
    radioRelu.config(state="disabled") 
    radioTanh.config(state="disabled") 
    radioSigmoid.config(state="disabled") 
    radioLinear.config(state="disabled") 
    radioBinaryCross.config(state="disabled") 
    radioMSE.config(state="disabled") 
    Minibatch_entry.config(state="disabled")     
    Lrate_entry.config(state="normal") 
    niterations_entry.config(state="normal") 
    num_iterations.set(100)
    LearningRate.set(0.0015)     
    plt.close()
    first_time = 0
    ImageRecognitionH = False
    ImageRecognitionSigns = True
    binaryclas = False
    binaryclasApp = False 
    binaryclasT = False       
    signalFit1 = False
    signalFit2 = False
    XGBoostFlag = False
    SignalFitCSV = False    
    MulticlassApp = False    
    NeuralStyleTransferFlag = False
    RNNFlag = False    
    FaceNetFlag = False  
    AgePrediction = False
    YoloPrediction = False
    FileLayers = False
    LayerDim.set("49152 40 10 5 1")   
    ActivationM.set(1)
    ActivationL.set(1)
    CostFunction.set(1)
    ConvModelVar.set(1)
    file_pathT = interface.open_datasetTraining()
    file_pathD = interface.open_datasetTest()
    OutputText.delete("1.0", "end")
    OutputTextExif.insert(tk.END, f"Training File: \n{(file_pathT):}")
    OutputTextExif.insert(tk.END, f"\nTest File: \n{(file_pathD):}")      
    set_data(file_pathT,file_pathD)
    
    
def RNN():
    global signalFit1, signalFit2, SignalFitCSV,binaryclas
    global ImageRecognitionH, ImageRecognitionSigns, NeuralStyleTransferFlag, RNNFlag
    global data, char_to_ix, ix_to_char
    global FaceNetFlag, AgePrediction, YoloPrediction, binaryclasApp, binaryclasT, MulticlassApp, XGBoostFlag
        
    ImageRecognitionH = False
    ImageRecognitionSigns = False
    binaryclas = False
    binaryclasApp = False 
    binaryclasT = False   
    signalFit1 = False
    signalFit2 = False
    NeuralStyleTransferFlag = False
    MulticlassApp = False    
    RNNFlag = True
    SignalFitCSV = False
    FaceNetFlag = False
    AgePrediction = False
    YoloPrediction = False   
    FileLayers = False  
    XGBoostFlag = False    
    
    lambDAVar_entry.config(state="disabled")
    Beta_entry.config(state="disabled")
    LayerDim_entry.config(state="disabled")
    TrainSize_entry.config(state="disabled")
    Testsize_entry.config(state="disabled")
    TimeInter_entry.config(state="disabled")
    CheckLDecay.config(state="disabled")
    CheckDataAugmented.config(state="disabled")
    ConvModel.config(state="disabled")
    CheckB.config(state="disabled") 
    ConvModelResnet.config(state="disabled") 
    radioIniZeros.config(state="disabled") 
    radioIniRandom.config(state="disabled") 
    radioIniHe.config(state="disabled") 
    radioOptGD.config(state="disabled") 
    radioOptMom.config(state="disabled") 
    radioOptAdam.config(state="disabled") 
    radioRelu.config(state="disabled") 
    radioTanh.config(state="disabled") 
    radioSigmoid.config(state="disabled") 
    radioLinear.config(state="disabled") 
    radioBinaryCross.config(state="disabled") 
    radioMSE.config(state="disabled") 
    Minibatch_entry.config(state="disabled") 
    Lrate_entry.config(state="normal")
    niterations_entry.config(state="normal")     
    
    LearningRate.set(0.01)
    num_iterations.set(20001)
    
    data = open('support/dinos.txt', 'r').read()
    data= data.lower()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))    
    chars = sorted(chars)
    print(chars)
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(ix_to_char)    

def FaceMesh():
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    mp_drawing_styles = mp.solutions.drawing_styles
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        refine_landmarks=True
        ) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:               
                continue
            image.flags.writeable = False
            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            results = face_mesh.process(image)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if results.multi_face_landmarks is not None:
                for face_landmarks in results.multi_face_landmarks:
                    #Coordenadas de la cara (arriba y abajo)
                    top = (face_landmarks.landmark[10].x, face_landmarks.landmark[10].y)
                    bottom = (face_landmarks.landmark[152].x, face_landmarks.landmark[152].y)
                    
                    mouth_top = (face_landmarks.landmark[13].x , face_landmarks.landmark[13].y)
                    mouth_bottom = (face_landmarks.landmark[14].x, face_landmarks.landmark[14].y)

                    distanceBetweenMouthPoints = math.sqrt( 
                        pow(mouth_top[0] - mouth_bottom[0], 2) +
                        pow(mouth_top[1] - mouth_bottom[1], 2)
                    )
                    
                    face_height = (bottom[1] - top[1])

                    #Obtener la distancia real de la boca dividida entre la cara
                    #para que sea mas 'relativamente constante'
                    real_distance = distanceBetweenMouthPoints * height
                    relative_distance = real_distance / face_height

                    #Obtener coordenadas del 'cuadrado' de la cara para poder mostrarlo en la pantalla despues
                    face_left_x = face_landmarks.landmark[234].x
                    face_right_x = face_landmarks.landmark[454].x
                    face_top_y = face_landmarks.landmark[10].y
                    face_bottom_y = face_landmarks.landmark[152].y
                    
                    
                    mouth_left_x = face_landmarks.landmark[57].x
                    mouth_right_x = face_landmarks.landmark[287].x
                    mouth_top_y = face_landmarks.landmark[0].y
                    mouth_bottom_y = face_landmarks.landmark[17].y


                    #Dejar algo de espacio alrededor
                    face_left_x = face_left_x - .1
                    face_right_x = face_right_x + .1
                    face_top_y = face_top_y - .1
                    face_bottom_y = face_bottom_y + .1

                    """cv2.line(
                        image, 
                        (int(top[0] * width), int(top[1] * height)),
                        (int(bottom[0] * width), int(bottom[1] * height)),
                        (0, 255, 0), 3
                    )"""
                    
                    cv2.line(
                        image, 
                        (int(mouth_top[0] * width), int(mouth_top[1] * height)),
                        (int(mouth_bottom[0] * width), int(mouth_bottom[1] * height)),
                        (0, 255, 240), 3
                    )

                    cv2.circle(image, (int(top[0] * width), int(top[1] * height)), 8, (0,0,255), -1)
                    cv2.circle(image, (int(bottom[0] * width), int(bottom[1] * height)), 8, (0,0,255), -1)
                    
                    cv2.circle(image, (int(mouth_top[0] * width), int(mouth_top[1] * height)), 4, (0,25,255), -1)
                    cv2.circle(image, (int(mouth_bottom[0] * width), int(mouth_bottom[1] * height)), 4, (0,25,255), -1)
                
                                        
                    
                    """mp_drawing.draw_landmarks(
                        image,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(150, 150, 150), thickness=1, circle_radius=1))"""
                        
                    """top = (int(face_landmarks.landmark[10].x * width), int(face_landmarks.landmark[10].y * height))
                    bottom = (int(face_landmarks.landmark[152].x * width), int(face_landmarks.landmark[152].y * height))
                    
                    cv2.line(image, top, bottom, (0, 255, 0), 3)
                    
                    cv2.circle(image,top, 8, (0,0,255), -1)
                    cv2.circle(image,bottom, 8, (0,0,255), -1)"""
                    degrees, movement = facemesh.detect_head_movement(top, bottom)
                    cv2.putText(image, 'Angle: ' + str(round(degrees,1)), (100, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, 'Movement: ' + str(round(movement,4)), (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, 'Dist: ' + str(round(relative_distance,4)), (100, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
            cv2.imshow("Frame", image)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break


    cap.release()
    cv2.destroyAllWindows()    


def YOLOimg():
    global signalFit1, signalFit2, SignalFitCSV,binaryclas
    global ImageRecognitionH, ImageRecognitionSigns, NeuralStyleTransferFlag, RNNFlag
    global FaceNetFlag, AgePrediction, YoloPrediction, binaryclasApp, binaryclasT, MulticlassApp, XGBoostFlag
    global model, classNames
    
    lambDAVar_entry.config(state="disabled")
    Beta_entry.config(state="disabled")
    LayerDim_entry.config(state="disabled")
    TrainSize_entry.config(state="disabled")
    Testsize_entry.config(state="disabled")
    TimeInter_entry.config(state="disabled")
    CheckLDecay.config(state="disabled")
    CheckDataAugmented.config(state="disabled")
    ConvModel.config(state="disabled")
    CheckB.config(state="disabled") 
    ConvModelResnet.config(state="disabled") 
    radioIniZeros.config(state="disabled") 
    radioIniRandom.config(state="disabled") 
    radioIniHe.config(state="disabled") 
    radioOptGD.config(state="disabled") 
    radioOptMom.config(state="disabled") 
    radioOptAdam.config(state="disabled") 
    radioRelu.config(state="disabled") 
    radioTanh.config(state="disabled") 
    radioSigmoid.config(state="disabled") 
    radioLinear.config(state="disabled") 
    radioBinaryCross.config(state="disabled") 
    radioMSE.config(state="disabled") 
    Minibatch_entry.config(state="disabled")  
    Lrate_entry.config(state="disabled")
    niterations_entry.config(state="disabled")
    YoloPrediction = True
    ImageRecognitionH = False
    ImageRecognitionSigns = False
    binaryclas = False
    binaryclasApp = False    
    binaryclasT = False    
    signalFit1 = False
    MulticlassApp = False    
    signalFit2 = False
    SignalFitCSV = False    
    NeuralStyleTransferFlag = False
    RNNFlag = False    
    FaceNetFlag = False  
    AgePrediction = False    
    FileLayers = False
    XGBoostFlag = False
    
    from ultralytics import YOLO
    # model
    model = YOLO("support/yolo/yolov8n.pt")
    
    # object classes
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]
    
def YOLOvideo():      
    lambDAVar_entry.config(state="disabled")
    Beta_entry.config(state="disabled")
    LayerDim_entry.config(state="disabled")
    TrainSize_entry.config(state="disabled")
    Testsize_entry.config(state="disabled")
    TimeInter_entry.config(state="disabled")
    CheckLDecay.config(state="disabled")
    CheckDataAugmented.config(state="disabled")
    ConvModel.config(state="disabled")
    CheckB.config(state="disabled") 
    ConvModelResnet.config(state="disabled") 
    radioIniZeros.config(state="disabled") 
    radioIniRandom.config(state="disabled") 
    radioIniHe.config(state="disabled") 
    radioOptGD.config(state="disabled") 
    radioOptMom.config(state="disabled") 
    radioOptAdam.config(state="disabled") 
    radioRelu.config(state="disabled") 
    radioTanh.config(state="disabled") 
    radioSigmoid.config(state="disabled") 
    radioLinear.config(state="disabled") 
    radioBinaryCross.config(state="disabled") 
    radioMSE.config(state="disabled") 
    Minibatch_entry.config(state="disabled")  
    Lrate_entry.config(state="disabled")
    niterations_entry.config(state="disabled")     
    
    # start video
    cap = cv2.VideoCapture("support/images/testing/video_sample2.mp4")

    from ultralytics import YOLO
    # model
    model = YOLO("support/yolo/yolov8n.pt")
    
    # object classes
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]
    colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype="uint8")              
                  
    while(cap.isOpened):
        success, img = cap.read()
        results = model(img, stream=True)
        
        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                #print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                #print("Class name -->", classNames[cls])
                
                label = '{} {:.2f}'.format(classNames[cls], confidence)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                (text_width, text_height) = cv2.getTextSize(label, font, fontScale=fontScale, thickness=1)[0]
                text_offset_x = x1
                text_offset_y = y1 - 5
                box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                color = colors[cls].tolist()

                overlay = img.copy()
                cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                # add opacity (transparency to the box)
                img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

                # object details
                org = [x1, y1 - 5]
                #color = (255, 0, 0)
                thickness = 2
                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                cv2.putText(img, label, org, font, fontScale=fontScale, color=(0, 0, 0), thickness=thickness)

        cv2.imshow('Video', img)
        if cv2.waitKey(1) == ord('q'):
            break 
    cap.release()
    cv2.destroyAllWindows()   
    
def YOLO():      
    lambDAVar_entry.config(state="disabled")
    Beta_entry.config(state="disabled")
    LayerDim_entry.config(state="disabled")
    TrainSize_entry.config(state="disabled")
    Testsize_entry.config(state="disabled")
    TimeInter_entry.config(state="disabled")
    CheckLDecay.config(state="disabled")
    CheckDataAugmented.config(state="disabled")
    ConvModel.config(state="disabled")
    CheckB.config(state="disabled") 
    ConvModelResnet.config(state="disabled") 
    radioIniZeros.config(state="disabled") 
    radioIniRandom.config(state="disabled") 
    radioIniHe.config(state="disabled") 
    radioOptGD.config(state="disabled") 
    radioOptMom.config(state="disabled") 
    radioOptAdam.config(state="disabled") 
    radioRelu.config(state="disabled") 
    radioTanh.config(state="disabled") 
    radioSigmoid.config(state="disabled") 
    radioLinear.config(state="disabled") 
    radioBinaryCross.config(state="disabled") 
    radioMSE.config(state="disabled") 
    Minibatch_entry.config(state="disabled")  
    Lrate_entry.config(state="disabled")
    niterations_entry.config(state="disabled")     
    
    # start webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    from ultralytics import YOLO
    # model
    model = YOLO("support/yolo/yolov8n.pt")
    
    # object classes
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]
    colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype="uint8")              
                  
    while True:
        success, img = cap.read()
        img = cv2.flip(img,1)
        results = model(img, stream=True)
        
        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                #print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                #print("Class name -->", classNames[cls])
                
                label = '{} {:.2f}'.format(classNames[cls], confidence)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                (text_width, text_height) = cv2.getTextSize(label, font, fontScale=fontScale, thickness=1)[0]
                text_offset_x = x1
                text_offset_y = y1 - 5
                box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                color = colors[cls].tolist()

                overlay = img.copy()
                cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                # add opacity (transparency to the box)
                img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

                # object details
                org = [x1, y1 - 5]
                #color = (255, 0, 0)
                thickness = 2
                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                cv2.putText(img, label, org, font, fontScale=fontScale, color=(0, 0, 0), thickness=thickness)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break 
    cap.release()
    cv2.destroyAllWindows()     

def Info():
    if AgePrediction == True:
        filename = 'file:///'+os.getcwd()+'/' + 'support/FaceNet/FaceNet.html'
        webbrowser.open_new_tab(filename)
        schVar=False        
        display_image('FaceNet/vgg-face-model6.jpg',schVar,image_label)         
        #image = Image.open("FaceNet/vgg-face-model2.jpg")
        #image.show()
        #imshow(image)
        #plt.show()
    elif Resnet.get() == 1:
        filename = 'file:///'+os.getcwd()+'/' + 'support/ResNet50/ResNet50.html'
        webbrowser.open_new_tab(filename)    
   

def AgePrediction():
    global generated_image, vgg_model_outputs, a_C, a_S
    global signalFit1, signalFit2, SignalFitCSV, binaryclas, ImageRecognitionH, ImageRecognitionSigns, NeuralStyleTransferFlag
    global FaceNetFlag, AgePrediction, YoloPrediction, binaryclasApp, binaryclasT, MulticlassApp, XGBoostFlag
    global FRmodel, database, faces
    global faceage_vgg_model
    
    lambDAVar_entry.config(state="disabled")
    Beta_entry.config(state="disabled")
    LayerDim_entry.config(state="disabled")
    TrainSize_entry.config(state="disabled")
    Testsize_entry.config(state="disabled")
    TimeInter_entry.config(state="disabled")
    CheckLDecay.config(state="disabled")
    CheckDataAugmented.config(state="disabled")
    ConvModel.config(state="disabled")
    CheckB.config(state="disabled") 
    ConvModelResnet.config(state="disabled") 
    Lrate_entry.config(state="disabled") 
    radioIniZeros.config(state="disabled") 
    radioIniRandom.config(state="disabled") 
    radioIniHe.config(state="disabled") 
    radioOptGD.config(state="disabled") 
    radioOptMom.config(state="disabled") 
    radioOptAdam.config(state="disabled") 
    radioRelu.config(state="disabled") 
    radioTanh.config(state="disabled") 
    radioSigmoid.config(state="disabled") 
    radioLinear.config(state="disabled") 
    radioBinaryCross.config(state="disabled") 
    radioMSE.config(state="disabled") 
    Minibatch_entry.config(state="disabled")
    niterations_entry.config(state="disabled")     
    NeuralStyleTransferFlag = False
    binaryclas = False
    binaryclasApp = False 
    MulticlassApp = False    
    binaryclasT = False   
    ImageRecognitionH = False
    ImageRecognitionSigns = False
    signalFit1 = False
    signalFit2 = False
    RNNFlag = False
    SignalFitCSV = False 
    FaceNetFlag = False
    AgePrediction = True
    YoloPrediction = False
    schVar=False
    FileLayers = False
    XGBoostFlag = False
    display_image('support/FaceNet/vgg-face-model6.jpg',schVar,image_label)  
        
    faceage_vgg_model = AgePredictor.ageModel()    
    if 0 == 1:
        for i in range(16):
            img1 = cv2.imread("support/FaceNet/images/cropped/ruben/" + str(i) + ".jpg", 1)
            #img = img1[...,::-1]
            #img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
            img = np.around(np.array(img1) / 255.0, decimals=12)
            img = cv2.resize(img, (224, 224))
            x_train = np.expand_dims(img, axis=0)        
            Age = AgePredictor.predictAge(faceage_vgg_model, x_train)
            print(colored(("Age = ", round(Age,2)),"green"))
            org = [70, 30]
            label = 'Age: {:.2f}'.format(Age)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1   
            #Path = "C:/RMolina/01_Learning/Coursera/AppDL/output"
            cv2.putText(img1, label, org, font, fontScale=fontScale, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)       
            cv2.imwrite("output/Photo" + str(i) + ".jpg", img1)
    #frame = cv2.imread("FaceNet/images/img3.jpeg", 1)




def FaceNetWebCam():
    global generated_image, vgg_model_outputs, a_C, a_S
    global signalFit1, signalFit2, SignalFitCSV, binaryclas, ImageRecognitionH, ImageRecognitionSigns, NeuralStyleTransferFlag
    global FaceNetFlag, AgePrediction, YoloPrediction, binaryclasApp, binaryclasT, MulticlassApp, XGBoostFlag
    global FRmodel, database, faces
    
    lambDAVar_entry.config(state="disabled")
    Beta_entry.config(state="disabled")
    LayerDim_entry.config(state="disabled")
    TrainSize_entry.config(state="disabled")
    Testsize_entry.config(state="disabled")
    TimeInter_entry.config(state="disabled")
    CheckLDecay.config(state="disabled")
    CheckDataAugmented.config(state="disabled")
    ConvModel.config(state="disabled")
    CheckB.config(state="disabled") 
    ConvModelResnet.config(state="disabled") 
    Lrate_entry.config(state="disabled") 
    radioIniZeros.config(state="disabled") 
    radioIniRandom.config(state="disabled") 
    radioIniHe.config(state="disabled") 
    radioOptGD.config(state="disabled") 
    radioOptMom.config(state="disabled") 
    radioOptAdam.config(state="disabled") 
    radioRelu.config(state="disabled") 
    radioTanh.config(state="disabled") 
    radioSigmoid.config(state="disabled") 
    radioLinear.config(state="disabled") 
    radioBinaryCross.config(state="disabled") 
    radioMSE.config(state="disabled") 
    Minibatch_entry.config(state="disabled")
    niterations_entry.config(state="disabled")     
    NeuralStyleTransferFlag = False
    binaryclas = False
    binaryclasApp = False
    binaryclasT = False       
    ImageRecognitionH = False
    ImageRecognitionSigns = False
    MulticlassApp = False    
    signalFit1 = False
    signalFit2 = False
    RNNFlag = False
    SignalFitCSV = False 
    FaceNetFlag = False
    YoloPrediction = False  
    FileLayers = False   
    XGBoostFlag = False    
    schVar=False
    display_image('support/FaceNet/vgg-face-model5.jpg',schVar,image_label) 
    
    json_file = open('support/FaceNet/keras-facenet-h5/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('support/FaceNet/keras-facenet-h5/model.h5')

    model.summary()    
    FRmodel = model
    
    # Plotting the model architecture
    #plot_model(model, to_file="InceptionFaceNet.png", show_shapes=True)
    
    faces =  ['Henry', 'Ben', 'Ruben']
    paths =  {'Henry': './support/FaceNet/images/cropped/henry', 'Ben': './support/FaceNet/images/cropped/ben', 'Ruben': './support/FaceNet/images/cropped/ruben'}
    if(len(faces) == 0):
        print("No images found in database!!")
        print("Please add images to database")
        sys.exit()
        

    faceage_vgg_model = AgePredictor.ageModel()     
    print(colored(("Faces = ",faces),"red"))
        
    database = {}
    for face in faces:
        database[face] = []

    for face in faces:
        for img in os.listdir(paths[face]):
            database[face].append(FaceNet.img_to_encoding(os.path.join(paths[face],img), FRmodel))

    if 1==0:
        fd = faceDetector('fd_model/haarcascade_frontalface_default.xml')
    else:   
        detector = dlib.get_frontal_face_detector()        
    margin = 40    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)        
        frame = imutils.resize(frame, width = 800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if 1==0:
            faceRects = fd.detect(gray)
        else:
            faceRects = detector(gray, 1) # result

        #for (x, y, w, h) in faceRects:
        AvgAge = 0
        cntAge = 0
        for face in faceRects:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())        
            left = x - margin // 2
            right = x + w + margin // 2
            bottom = y - margin // 2
            top = y + h + margin // 2            
            roi = frame[bottom:top,left:right]            
            #roi = frame[x:x+w,y:y+h]  
            #cv2.rectangle(frame, (left -1, bottom - 1), (right + 1 ,top + 1), (0, 0, 255), 2)
            #roi = frame[y:y+y1,x:x+x1]

            img = np.around(np.array(roi) / 255.0, decimals=12)
            img = cv2.resize(img, (224, 224))
            x_train = np.expand_dims(img, axis=0)        
            Age = AgePredictor.predictAge(faceage_vgg_model, x_train)
            AvgAge = AvgAge + Age            
            if cntAge >= 10:
                AvgAge = AvgAge / 10
                cntAge = 0
                print(colored(("Age = ", round(AvgAge,2)),"green"))
            else:
                cntAge = cntAge + 1
                
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            #cv2.imwrite("foto2.jpg", roi)
            roi = cv2.resize(roi,(160, 160)) 
            min_dist = 100
            identity = ""
            detected  = False
            #cv2.imwrite("foto1.jpg", roi)
            
            for face in range(len(faces)):
                person = faces[face]
                #print(roi.shape)
                dist, detected = FaceNet.verify2(roi, person, database[person], FRmodel, webcam=True)
                if detected == True and dist<min_dist:
                    min_dist = dist
                    identity = person
           
            if detected == True:
                #label = 'Identity: {} Dist: {:.2f} Age: {:.2f}'.format(identity, min_dist, Age)
                label1 = 'Identity: {}'.format(identity)
                label2 = 'Dist: {:.2f}'.format(min_dist)
                label3 = 'Age: {:.2f}'.format(AvgAge)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                """(text_width, text_height) = cv2.getTextSize(label, font, fontScale=fontScale, thickness=1)[0]
                text_offset_x = x
                text_offset_y = y - 5
                box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                color = (0, 0, 255)
                overlay = frame.copy()
                cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                # add opacity (transparency to the box)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)"""
                # object details
                org1 = [x, y - 85]
                org2 = [x, y - 45]
                org3 = [x, y - 5]
                thickness = 2              
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label1, org1, font, fontScale=fontScale, color=(0, 0, 255), thickness=thickness, lineType=cv2.LINE_AA)
                cv2.putText(frame, label2, org2, font, fontScale=fontScale, color=(0, 100, 200), thickness=thickness, lineType=cv2.LINE_AA)
                cv2.putText(frame, label3, org3, font, fontScale=fontScale, color=(0, 220, 0), thickness=thickness, lineType=cv2.LINE_AA)
            else:
                label1 = 'Identity: unknown'                
                label2 = 'Age: {:.2f}'.format(AvgAge)                
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1   
                org1 = [x, y - 45]
                org2 = [x, y - 5]
                thickness = 2              
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label1, org1, font, fontScale=fontScale, color=(0, 0, 255), thickness=thickness, lineType=cv2.LINE_AA)
                cv2.putText(frame, label2, org2, font, fontScale=fontScale, color=(0, 220, 0), thickness=thickness, lineType=cv2.LINE_AA)
                
             
        cv2.imshow('frame', frame)
        cv2.imwrite("foto1.jpg", frame)        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()   
    cv2.destroyAllWindows()            
            
           

def FaceNetFunc():
    global generated_image, vgg_model_outputs, a_C, a_S
    global signalFit1, signalFit2, SignalFitCSV, binaryclas, ImageRecognitionH, ImageRecognitionSigns, NeuralStyleTransferFlag
    global FaceNetFlag, AgePrediction, YoloPrediction, binaryclasApp, binaryclasT, MulticlassApp, XGBoostFlag
    global FRmodel, database, faces
    
    lambDAVar_entry.config(state="disabled")
    Beta_entry.config(state="disabled")
    LayerDim_entry.config(state="disabled")
    TrainSize_entry.config(state="disabled")
    Testsize_entry.config(state="disabled")
    TimeInter_entry.config(state="disabled")
    CheckLDecay.config(state="disabled")
    CheckDataAugmented.config(state="disabled")
    ConvModel.config(state="disabled")
    CheckB.config(state="disabled") 
    ConvModelResnet.config(state="disabled") 
    Lrate_entry.config(state="disabled") 
    radioIniZeros.config(state="disabled") 
    radioIniRandom.config(state="disabled") 
    radioIniHe.config(state="disabled") 
    radioOptGD.config(state="disabled") 
    radioOptMom.config(state="disabled") 
    radioOptAdam.config(state="disabled") 
    radioRelu.config(state="disabled") 
    radioTanh.config(state="disabled") 
    radioSigmoid.config(state="disabled") 
    radioLinear.config(state="disabled") 
    radioBinaryCross.config(state="disabled") 
    radioMSE.config(state="disabled") 
    Minibatch_entry.config(state="disabled") 
    niterations_entry.config(state="disabled")         
    NeuralStyleTransferFlag = False
    binaryclas = False
    binaryclasT = False    
    MulticlassApp = False    
    binaryclasApp = False    
    ImageRecognitionH = False
    ImageRecognitionSigns = False
    signalFit1 = False
    signalFit2 = False
    RNNFlag = False
    SignalFitCSV = False 
    FaceNetFlag = True
    YoloPrediction = False
    FileLayers = False
    XGBoostFlag = False
    
    json_file = open('support/FaceNet/keras-facenet-h5/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('support/FaceNet/keras-facenet-h5/model.h5')
    
    model.summary()    
    FRmodel = model

    """https://github.com/sainimohit23/FaceNet-Real-Time-face-recognition/blob/master/webcamFaceRecoMulti.py"""
    
    faces =  ['Henry', 'Ben', 'Ruben']
    paths =  {'Henry': './support/FaceNet/images/cropped/henry', 'Ben': './support/FaceNet/images/cropped/ben', 'Ruben': './support/FaceNet/images/cropped/ruben'}
    if(len(faces) == 0):
        print("No images found in database!!")
        print("Please add images to database")
        sys.exit()
        
    print(colored(("Faces = ",faces),"red"))
    
    database = {}
    for face in faces:
        database[face] = []

    for face in faces:
        for img in os.listdir(paths[face]):
            database[face].append(FaceNet.img_to_encoding(os.path.join(paths[face],img), FRmodel))


def get_layer_outputs(vgg, layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model



def NeuralStyleTransfer():
    global generated_image, vgg_model_outputs, a_C, a_S
    global signalFit1, signalFit2, SignalFitCSV, binaryclas, ImageRecognitionH, ImageRecognitionSigns, NeuralStyleTransferFlag
    global FaceNetFlag, AgePrediction, YoloPrediction, binaryclasApp, binaryclasT, MulticlassApp, XGBoostFlag
      
    lambDAVar_entry.config(state="disabled")
    Beta_entry.config(state="disabled")
    LayerDim_entry.config(state="disabled")
    TrainSize_entry.config(state="disabled")
    Testsize_entry.config(state="disabled")
    TimeInter_entry.config(state="disabled")
    CheckLDecay.config(state="disabled")
    CheckDataAugmented.config(state="disabled")
    ConvModel.config(state="disabled")
    CheckB.config(state="disabled") 
    ConvModelResnet.config(state="disabled") 
    Lrate_entry.config(state="normal") 
    radioIniZeros.config(state="disabled") 
    radioIniRandom.config(state="disabled") 
    radioIniHe.config(state="disabled") 
    radioOptGD.config(state="disabled") 
    radioOptMom.config(state="disabled") 
    radioOptAdam.config(state="disabled") 
    radioRelu.config(state="disabled") 
    radioTanh.config(state="disabled") 
    radioSigmoid.config(state="disabled") 
    radioLinear.config(state="disabled") 
    radioBinaryCross.config(state="disabled") 
    radioMSE.config(state="disabled") 
    Minibatch_entry.config(state="disabled")
    niterations_entry.config(state="normal")
    num_iterations.set(30)    
    NeuralStyleTransferFlag = True
    binaryclas = False
    MulticlassApp = False
    binaryclasT = False
    binaryclasApp = False
    ImageRecognitionH = False
    ImageRecognitionSigns = False
    signalFit1 = False
    signalFit2 = False
    RNNFlag = False
    SignalFitCSV = False 
    FaceNetFlag = False
    YoloPrediction = False    
    FileLayers = False
    XGBoostFlag = False
    tf.random.set_seed(272) # DO NOT CHANGE THIS VALUE
    #pp = pprint.PrettyPrinter(indent=4)
    img_size = 400
    vgg = tf.keras.applications.VGG19(include_top=False,
                                  input_shape=(img_size, img_size, 3),
                                  weights='support/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

    vgg.trainable = False   
    content_image = np.array(Image.open("support/imagesNST/louvre.jpg").resize((img_size, img_size)))
    content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))
    print(content_image.shape)
    imshow(content_image[0])
    plt.show()        
    style_image =  np.array(Image.open("support/imagesNST/monet.jpg").resize((img_size, img_size)))
    style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))
    print(style_image.shape)
    imshow(style_image[0])
    plt.show()
    
    generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
    generated_image = tf.add(generated_image, noise)
    generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)

    #print(generated_image.shape)
    #imshow(generated_image.numpy()[0])
    #plt.show()    
    
    content_layer = [('block5_conv4', 1)]
    vgg_model_outputs = get_layer_outputs(vgg, nstLib.STYLE_LAYERS + content_layer)
    
    content_target = vgg_model_outputs(content_image)  # Content encoder
    style_targets = vgg_model_outputs(style_image)     # Style encoder
    # Assign the content image to be the input of the VGG model.  
    # Set a_C to be the hidden layer activation from the layer we have selected
    preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    a_C = vgg_model_outputs(preprocessed_content)
    
    # Assign the input of the model to be the "style" image 
    preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
    a_S = vgg_model_outputs(preprocessed_style)    
    
    
def open_TxtLayers():
    global keep_prob_list
    global file_path_txt
    global activationMVar, activationLVar
    # Program to read the entire file (absolute path) using read() function
    
   
    file_path_txt = filedialog.askopenfilename(title="Open Layers File", filetypes=[("Layer file", "*.txt")])      
    file = open(file_path_txt, "r")
    FileLayers = True
    b = []
    Layers = -1
    i = 0
    Drop = np.zeros((20,2))
    Dropout = []
    NLayers = 0
    dropFlag = False
    layer = False
    words = ['tanh', 'relu', 'sigmoid', 'linear']
    activationMVar = 'relu'
    activationLVar = 'sigmoid'
    while True:
        content=file.readline()
        if not content:            
            break
        for word in words:
            if word in content:                
                NLayers = NLayers + 1
                if NLayers == 1:
                    activationMVar = word
                if NLayers > 1:
                    activationLVar = word                    

                
        
       
        
            
        """for line in content:
            if content[i] = "=":
                ActivationM = content[i:i+5]
                print(ActivationM)"""
        
        if content[0:7] != "Dropout":           
            a=re.findall(r'\d+',content)
            Layers = Layers + 1
            #i=0
            #while i < len(content)
            b = b + a
            layer = True
            if Layers > 1 and dropFlag == False:
                Dropout.append(1.0)
            dropFlag = False    
        else:
            
            Drop[i,0] = Layers
            temp=interface.convert_strings_to_floats((re.findall(r'\d+\.\d+',content)))
            result = " ".join(str(t) for t in temp)           
            Drop[i,1] = result
            
            if layer == True:
                Dropout.append(float(result))
                layer = False
                dropFlag = True                
           
            
            i = i + 1
            #rop[i,0] = Layers            
            #Drop[0,i] = (re.findall(r'\d+\.\d+',content))
            #i = i + 1
        #print(a)
    print(activationMVar)
    print(activationLVar) 
    print("-----")
    #keep_prob_list = [0.86, 0.86]
    file.close()
    #print(Layers)        
    #print(Drop)
    #print(Dropout)
    keep_prob_list = Dropout
    #print(b)
    layers_dims = interface.convert_strings_to_ints(b)
    #Drop=interface.convert_strings_to_floats(Drop)
    result = " ".join(str(i) for i in layers_dims) 
    #layers_dims = re.sub(r'[[]]', '', list(layers_dims))
    #print(result)    
    LayerDim.set(result)
   
   
def conv_option():
    if ConvModelVar.get() == True:
        lambDAVar_entry.config(state="disabled")
        Beta_entry.config(state="disabled")
        LayerDim_entry.config(state="disabled")
        TrainSize_entry.config(state="disabled")
        Testsize_entry.config(state="disabled")
        TimeInter_entry.config(state="disabled")
        CheckLDecay.config(state="disabled")
        CheckDataAugmented.config(state="disabled")
        CheckB.config(state="disabled") 
        ConvModelResnet.config(state="normal") 
        Lrate_entry.config(state="normal") 
        radioIniZeros.config(state="disabled") 
        radioIniRandom.config(state="disabled") 
        radioIniHe.config(state="disabled") 
        radioOptGD.config(state="disabled") 
        radioOptMom.config(state="disabled") 
        radioOptAdam.config(state="disabled") 
        radioRelu.config(state="disabled") 
        radioTanh.config(state="disabled") 
        radioSigmoid.config(state="disabled") 
        radioLinear.config(state="disabled") 
        radioBinaryCross.config(state="disabled") 
        radioMSE.config(state="disabled") 
        Minibatch_entry.config(state="normal")
        niterations_entry.config(state="normal")
    else:        
        if XGBoostFlag == True:
            lambDAVar_entry.config(state="disabled")
            Beta_entry.config(state="disabled")
            LayerDim_entry.config(state="disabled")
            TrainSize_entry.config(state="disabled")
            Testsize_entry.config(state="disabled")
            TimeInter_entry.config(state="disabled")
            CheckLDecay.config(state="disabled")
            CheckDataAugmented.config(state="disabled")
            CheckB.config(state="disabled") 
            ConvModelResnet.config(state="disabled") 
            Lrate_entry.config(state="normal") 
            radioIniZeros.config(state="disabled") 
            radioIniRandom.config(state="disabled") 
            radioIniHe.config(state="disabled") 
            radioOptGD.config(state="disabled") 
            radioOptMom.config(state="disabled") 
            radioOptAdam.config(state="disabled") 
            radioRelu.config(state="disabled") 
            radioTanh.config(state="disabled") 
            radioSigmoid.config(state="disabled") 
            radioLinear.config(state="disabled") 
            radioBinaryCross.config(state="disabled") 
            radioMSE.config(state="disabled") 
            Minibatch_entry.config(state="disabled")
            niterations_entry.config(state="disabled")          
        if BinaryClassComparison == True:
            lambDAVar_entry.config(state="disabled")
            Beta_entry.config(state="disabled")
            LayerDim_entry.config(state="disabled")
            TrainSize_entry.config(state="disabled")
            Testsize_entry.config(state="disabled")
            TimeInter_entry.config(state="disabled")
            CheckLDecay.config(state="disabled")
            CheckDataAugmented.config(state="disabled")
            CheckB.config(state="disabled") 
            ConvModelResnet.config(state="disabled") 
            Lrate_entry.config(state="disabled") 
            radioIniZeros.config(state="disabled") 
            radioIniRandom.config(state="disabled") 
            radioIniHe.config(state="disabled") 
            radioOptGD.config(state="disabled") 
            radioOptMom.config(state="disabled") 
            radioOptAdam.config(state="disabled") 
            radioRelu.config(state="disabled") 
            radioTanh.config(state="disabled") 
            radioSigmoid.config(state="disabled") 
            radioLinear.config(state="disabled") 
            radioBinaryCross.config(state="disabled") 
            radioMSE.config(state="disabled") 
            Minibatch_entry.config(state="disabled")
            niterations_entry.config(state="disabled") 
        elif MulticlassApp == True:
            lambDAVar_entry.config(state="normal")
            Beta_entry.config(state="disabled")
            LayerDim_entry.config(state="disabled")
            TrainSize_entry.config(state="disabled")
            Testsize_entry.config(state="disabled")
            TimeInter_entry.config(state="disabled")
            CheckLDecay.config(state="disabled")
            CheckDataAugmented.config(state="disabled")
            CheckB.config(state="disabled") 
            ConvModelResnet.config(state="disabled") 
            Lrate_entry.config(state="normal") 
            radioIniZeros.config(state="disabled") 
            radioIniRandom.config(state="disabled") 
            radioIniHe.config(state="disabled") 
            radioOptGD.config(state="disabled") 
            radioOptMom.config(state="disabled") 
            radioOptAdam.config(state="disabled") 
            radioRelu.config(state="disabled") 
            radioTanh.config(state="disabled") 
            radioSigmoid.config(state="disabled") 
            radioLinear.config(state="disabled") 
            radioBinaryCross.config(state="disabled") 
            radioMSE.config(state="disabled") 
            Minibatch_entry.config(state="normal")
            niterations_entry.config(state="normal")        
        else:        
            lambDAVar_entry.config(state="normal")
            Beta_entry.config(state="normal")
            LayerDim_entry.config(state="normal")
            if signalFit1 == 1 or signalFit2 == 1 or SignalFitCSV == 1 or binaryclas == 1 or binaryclasApp == 1 or binaryclasT == 1:
                TrainSize_entry.config(state="disabled")
                Testsize_entry.config(state="disabled")
            else:
                TrainSize_entry.config(state="normal")
                Testsize_entry.config(state="normal")
            TimeInter_entry.config(state="normal")
            CheckLDecay.config(state="normal")
            if signalFit1 == 1 or signalFit2 == 1 or SignalFitCSV == 1 or binaryclas == 1 or binaryclasApp == 1 or binaryclasT == 1:
                CheckDataAugmented.config(state="disabled")
                ConvModel.config(state="disabled")
                ConvModelResnet.config(state="disabled") 
            else:
                CheckDataAugmented.config(state="normal")
                ConvModel.config(state="normal")
                ConvModelResnet.config(state="normal") 
            CheckB.config(state="normal")         
            radioIniZeros.config(state="normal") 
            radioIniRandom.config(state="normal") 
            radioIniHe.config(state="normal") 
            radioOptGD.config(state="normal") 
            radioOptMom.config(state="normal") 
            radioOptAdam.config(state="normal") 
            radioRelu.config(state="normal") 
            radioTanh.config(state="normal") 
            radioSigmoid.config(state="normal") 
            radioLinear.config(state="normal") 
            radioBinaryCross.config(state="normal") 
            radioMSE.config(state="normal") 
            Minibatch_entry.config(state="normal") 
            Lrate_entry.config(state="normal") 
            niterations_entry.config(state="normal")     
    root.after(1000, conv_option)
   

root = tk.Tk()
root.geometry("1200x800") #You want the size of the app to be 500x500
root.title(" Demo Deep Learning Tool")
root.resizable(0,0)
#root.state('zoomed')



# Cargar el archivo de imagen desde el disco.
icono = tk.PhotoImage(file = r"img\icons8.png") 
# Establecerlo como ícono de la ventana.
root.iconphoto(True, icono)

menu = Menu(root)
root.config(menu=menu)
filemenu = Menu(menu)
menu.add_cascade(label='File', menu=filemenu)
filemenu.add_command(label='Open Data Training...',command=interface.open_datasetTraining)
filemenu.add_command(label='Open Data Test...',command=interface.open_datasetTest)
#filemenu.add_command(label='Open Data Test MNIST...',command=open_datasetMNIST)
filemenu.add_command(label='Open Image ...',command=open_image)
filemenu.add_command(label='Open Layers TXT ...',command=open_TxtLayers)
filemenu.add_separator()
filemenu.add_command(label='Exit', command=root.quit)
Examples = Menu(menu)
menu.add_cascade(label='Examples', menu=Examples)
Examples.add_command(label='1-Image Recognition 64px of Cats',command=ImageRecognition_64px)
Examples.add_command(label='2-Image Recognition 128px Cats & Dogs',command=ImageRecognition_128px)
Examples.add_command(label='3-Image Recognition Signs...',command=ImageRecognitionSigns)
Examples.add_command(label='4.1-Planar Classsifier 1',command=BinaryClass)
Examples.add_command(label='4.2-Planar Classsifier Comparison',command=BinaryClassComparison)
Examples.add_command(label='4.3-Planar Classsifier 2',command=BinaryClassT)
Examples.add_command(label='4.4-Planar Classsifier Application',command=BinaryClassApp)
Examples.add_command(label='4.5-Multi Classsifier Application',command=MultClassApp)
Examples.add_command(label='5-Signal Fit 1...',command=SignalFit1)
Examples.add_command(label='6-Signal Fit 2...',command=SignalFit2)
Examples.add_command(label='7-Signal Fit CSV...',command=SignalFitCSV)
Examples.add_command(label='8-YOLO Img model...',command=YOLOimg)
Examples.add_command(label='9-YOLO Camera model...',command=YOLO)
Examples.add_command(label='10-YOLO Video model...',command=YOLOvideo)
Examples.add_command(label='11-FaceNet...',command=FaceNetFunc)
Examples.add_command(label='12-FaceNet WebCam...',command=FaceNetWebCam)
Examples.add_command(label='13-Age Prediction...',command=AgePrediction)
Examples.add_command(label='14-Neural Style Transfer model...',command=NeuralStyleTransfer)
Examples.add_command(label='15-Recurrent Neural Network model...',command=RNN)
Examples.add_command(label='16-Face Mesh Neural Network model...',command=FaceMesh)
Examples.add_command(label='17-XGBoost Classification model...',command=XGBoost)

helpmenu = Menu(menu)
menu.add_cascade(label='Help', menu=helpmenu)
helpmenu.add_command(label='1-Neural Networks and Deep Learning',command=C1)
helpmenu.add_command(label='2-Improving Deep Neural Networks',command=C2)
helpmenu.add_command(label='3-Structuring Machine Learning Projects',command=C3)
helpmenu.add_command(label='4-Convolutional Neural Networks',command=C4)
helpmenu.add_command(label='5-Sequence Models',command=C5)
#helpmenu.add_command(label='About',command=aboutF)


# Creating a photoimage object to use image 
photo = PhotoImage(file = r"img\play-48.png") 
  
# Resizing image to fit on button 
photoimage = photo.subsample(2, 2) 

# declaring string variable
# for storing name and password
num_iterations=tk.DoubleVar(value=100)
#LayerDim=tk.StringVar(value="1 5 5 1")
LayerDim=tk.StringVar(value="49152 40 10 5 1")

LearningRate=tk.DoubleVar(value=0.0075)
lambDAVar=tk.DoubleVar(value=0.0)
MiniBatchVar = tk.DoubleVar(value=64)
ClassVar1 = tk.DoubleVar(value=1.6)
ClassVar2 = tk.DoubleVar(value=0.88)
BetaVar = tk.DoubleVar(value=0.9)
TimeInterVar = tk.DoubleVar(value=1000)
TrainSizeVar = tk.DoubleVar(value=512)
TestSizeVar = tk.DoubleVar(value=64)


#########
etiq1 = tk.Label(root,text="Epoch_num:")
etiq1.place(x=10, y=100)
niterations_entry = tk.Entry(root,textvariable = num_iterations, font=('calibre',10,'normal'), width=10, justify="right")
niterations_entry.place(x=100, y=100)

etiq2 = tk.Label(root,text="Leearnig Rate:")
etiq2.place(x=10, y=130)
Lrate_entry = tk.Entry(root,textvariable = LearningRate, font=('calibre',10,'normal'), width=10, justify="right", fg="green")
Lrate_entry.place(x=100, y=130)

etiq3 = tk.Label(root,text="Layer Dim:")
etiq3.place(x=10, y=160)
LayerDim_entry = tk.Entry(root, textvariable = LayerDim, font=('calibre',10,'normal'), width=20, justify="right", fg="green")
LayerDim_entry.place(x=100, y=160)

etiq6 = tk.Label(root,text="Beta")
etiq6.place(x=10, y=190)
Beta_entry = tk.Entry(root,textvariable = BetaVar, font=('calibre',10,'normal'), width=10, justify="right", fg="green")
Beta_entry.place(x=100, y=190)

#etiq5 = tk.Label(root,text="Pixel Star", bg = "light cyan", font=("Arial", 12) )
#etiq5.place(x=10, y=220)
etiq4 = tk.Label(root,text="Lambda:")
etiq4.place(x=10, y=220)
lambDAVar_entry = tk.Entry(root,textvariable = lambDAVar, font=('calibre',10,'normal'), width=10, justify="right")
lambDAVar_entry.place(x=100, y=220)
###########

etiq12 = tk.Label(root,text="Train Size max 20000")
etiq12.place(x=200, y=240)
TrainSize_entry = tk.Entry(root,textvariable = TrainSizeVar, font=('calibre',10,'normal'), width=10, justify="right", fg="green")
TrainSize_entry.place(x=200, y=260)

#etiq5 = tk.Label(root,text="Pixel Star", bg = "light cyan", font=("Arial", 12) )
#etiq5.place(x=10, y=220)
etiq13 = tk.Label(root,text="Test Size max 5000")
etiq13.place(x=200, y=280)
Testsize_entry = tk.Entry(root,textvariable = TestSizeVar, font=('calibre',10,'normal'), width=10, justify="right", fg="green")
Testsize_entry.place(x=200, y=300)

etiq7 = tk.Label(root,text="Time interval")
etiq7.place(x=120, y=480)
TimeInter_entry = tk.Entry(root,textvariable = TimeInterVar, font=('calibre',10,'normal'), width=10, justify="right")
TimeInter_entry.place(x=120, y=500)

text_widget = tk.Text(root, wrap=tk.WORD, height=15, width=35)

#########
IniWeights = IntVar()
MiniBatch = IntVar()
optimizer = IntVar()
LDecay = IntVar()
ConvModelVar = IntVar()
Resnet = IntVar()
ActivationM = IntVar()
ActivationL = IntVar()
CostFunction = IntVar()
SignalFit = IntVar()
DataAugmented = IntVar()

IniWeights.set(3)
optimizer.set(1)


radioRelu=tk.Radiobutton(root, 
               text="ReLu",                
               variable=ActivationM,
               pady=5,               
               value=1)
radioRelu.place(x=100, y=5)

radioTanh=tk.Radiobutton(root, 
               text="Tanh",                
               variable=ActivationM,
               pady=5,               
               value=2)
radioTanh.place(x=100, y=35) 

radioSigmoid=tk.Radiobutton(root, 
               text="Sigmoid",                
               variable=ActivationL,
               pady=5,               
               value=1)
radioSigmoid.place(x=160, y=5)

radioLinear=tk.Radiobutton(root, 
               text="Linear",                
               variable=ActivationL,
               pady=5,               
               value=2)
radioLinear.place(x=160, y=35) 

radioBinaryCross=tk.Radiobutton(root, 
               text="binaryCross",                
               variable=CostFunction,
               pady=5,               
               value=1)
radioBinaryCross.place(x=240, y=5)

radioMSE=tk.Radiobutton(root, 
               text="MSE",                
               variable=CostFunction,
               pady=5,               
               value=2)
radioMSE.place(x=240, y=35) 



CheckLDecay=tk.Checkbutton(root, text="Learning Decay", variable=LDecay, 
            onvalue=True, offvalue=False)            
CheckLDecay.place(x=0, y=500) 


CheckDataAugmented=tk.Checkbutton(root, text="Data Augmented", variable=DataAugmented, 
            onvalue=1, offvalue=0)            
CheckDataAugmented.place(x=150, y=350) 

ConvModel=tk.Checkbutton(root, text="Convolutional Model", variable=ConvModelVar, 
            onvalue=True, offvalue=False)           
ConvModel.place(x=150, y=380) 

ConvModelResnet=tk.Checkbutton(root, text="Resnet", variable=Resnet, 
            onvalue=True, offvalue=False)           
ConvModelResnet.place(x=150, y=410) 



CheckB=tk.Checkbutton(root, text="Mini Batch", variable=MiniBatch, 
            onvalue=1, offvalue=0)            
CheckB.place(x=0, y=260) 


etiq5 = tk.Label(root,text="Mini Batch Size:")
etiq5.place(x=0, y=300)
Minibatch_entry = tk.Entry(root,textvariable = MiniBatchVar, font=('calibre',10,'normal'), width=10, justify="right",  fg="green")
Minibatch_entry.place(x=95, y=300)


radioIniZeros=tk.Radiobutton(root, 
               text="Ini Zeros",                
               variable=IniWeights,
               pady=5,               
               value=1)
radioIniZeros.place(x=0, y=5) 

radioIniRandom=tk.Radiobutton(root, 
               text="Ini Random",                
               variable=IniWeights,
               pady=5,          
               value=2)
radioIniRandom.place(x=0, y=35) 

radioIniHe=tk.Radiobutton(root, 
               text="Ini He",                
               variable=IniWeights,
               pady=5,          
               value=3)
radioIniHe.place(x=0, y=65) 


radioOptGD=tk.Radiobutton(root, 
               text="optimizer gd",                
               variable=optimizer,
               pady=5,               
               value=1)
radioOptGD.place(x=0, y=350) 

radioOptMom=tk.Radiobutton(root, 
               text="optimizer momentum",                
               variable=optimizer,
               pady=5,          
               value=2)
radioOptMom.place(x=0, y=380) 

radioOptAdam=tk.Radiobutton(root, 
               text="optimizer Adam",                
               variable=optimizer,
               pady=5,          
               value=3)
radioOptAdam.place(x=0, y=410) 
               


count_button=tk.Button(root, text = '  Train',  anchor="w", image = photoimage, command=train, compound = LEFT)
count_button.place(x=10, y=650)   
count_button.config(width=80, height=30)

Predict_button=tk.Button(root, text = '  Predict', anchor="w", image = photoimage, command=predictT, compound = LEFT)
Predict_button.place(x=10, y=690)   
Predict_button.config(width=80, height=30)

sch_button=tk.Button(root, text = '  Sch NN',  anchor="w", image = photoimage, command=lambda: interface.sch(LayerDim.get(),image_label), compound = LEFT)
sch_button.place(x=10, y=730)   
sch_button.config(width=80, height=30)

sch_button=tk.Button(root, text = '  Info',  anchor="w", image = photoimage, command=Info, compound = LEFT)
sch_button.place(x=140, y=690)   
sch_button.config(width=80, height=30)

sch_button=tk.Button(root, text = '  Results',  anchor="w", image = photoimage, command=plot, compound = LEFT)
sch_button.place(x=140, y=730)   
sch_button.config(width=80, height=30)

OutputText = tk.Text(root, bg = "light cyan", height = 4, width = 30, )
OutputText.place(x=10, y=550)


OutputTextExif = tk.Text(root, bg = "light cyan", height = 40, width = 30, )
OutputTextExif.place(x=940, y=10)


#open_button.place(x=25, y=10)
image_label = tk.Label(root)
image_label.pack(padx=20, pady=20)
status_label = tk.Label(root, text="", padx=20, pady=10)

#open_button2.place(x=200, y=10)
image_label2 = tk.Label(root)
image_label2.pack(padx=20, pady=20)
status_label.pack()

conv_option()

#display_image('nn.png',schVar)

if getattr(sys, 'frozen', False):
    pyi_splash.close()

 
root.mainloop()
