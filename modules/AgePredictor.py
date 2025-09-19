import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model, load_model

import numpy as np # linear algebra
import cv2
from termcolor import colored

#def find_apparent_age(age_predictions: np.ndarray) -> np.float64:
def find_apparent_age(age_predictions):
    """
    Find apparent age prediction from a given probas of ages
    Args:
        age_predictions (?)
    Returns:
        apparent_age (float)
    """
    output_indexes = np.arange(0, 101)
    apparent_age = np.sum(age_predictions * output_indexes)   
    return apparent_age            
            
#def predict(img: np.ndarray) -> np.float64:
def predictAge(faceage_vgg_model, img):
        # model.predict causes memory issue when it is called in a for loop
        # age_predictions = self.model.predict(img, verbose=0)[0, :]
        age_predictions = faceage_vgg_model(img, training=False).numpy()[0, :]
        return find_apparent_age(age_predictions)            

def base_model() -> Sequential:
    global model
    """
    Base model of VGG-Face being used for classification - not to find embeddings
    Returns:
        model (Sequential): model was trained to classify 2622 identities
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(4096, (7, 7), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Conv2D(4096, (1, 1), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Conv2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation("softmax"))

    return model
      



def ageModel(input_shape=(224, 224, 3)):

    vgg_model = base_model()    
    vgg_model.summary()

    classes = 101
    base_model_output = Sequential()
    base_model_output = Conv2D(classes, (1, 1), name="predictions")(vgg_model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation("softmax")(base_model_output)

    faceage_vgg_model = Model(vgg_model.input, base_model_output)
    weight_file ='support/VggFaceAge/age_model_weights.h5'
    faceage_vgg_model.load_weights(weight_file)
    #faceage_vgg_model = weight_utils.load_model_weights(model=faceage_vgg_model, weight_file=weight_file)

    print("Face age model summary.")
    faceage_vgg_model.summary()
    return faceage_vgg_model