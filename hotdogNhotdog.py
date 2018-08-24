
# coding: utf-8

# A binary class version of agaadha-nama
# - Added 'None' support for N_SAMPLE for processing all images in selected class
# - Added LABEL_CLASSES support for selecting classes to categorize by name


# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# #
# # Housekeeping
# # 
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

import os
import glob
import sys

import cPickle as pickle

import numpy as np
import pandas as pd
import os.path as path
from scipy import misc, stats
from skimage.transform import resize
import matplotlib.pyplot as plt
import random
import cv2

import pathlib

from keras.models import Sequential, save_model
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import RMSprop

from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from datetime import datetime
from keras.utils import to_categorical # For keras > 2.0
from keras.callbacks import ReduceLROnPlateau

# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# #
# # DATA PREPROCESSING
# # 
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

PATH = 'dataset/google/*/*'

N_CLASSES = 5 # how many asanas to classify
LABEL_CLASSES = ['Urdhva+Dhanurasana'] # which asanas to classify

N_SAMPLE = None  #None # how many to sample from each class (classes w/ <N will be dropped)
                # if None, automatically selects all observations from each class
TRAIN_TEST_SPLIT = 0.8

IMG_SIZE = (300,300)

def create_filepathsdf(path, n_classes=None, label_classes=None, n_sample=None):
    """ Create dataframe of image labels and file-locations.
    
        Input:  - path to glob for full path of each file.
                    contents of path should be the result of 
                        scrape_images_search.py (image_search)
                    ignores any metadata files (*.json)
                - n_classes number of classes to use in classification
                    if None, takes the number of classes found in path
                - label_classes specific labels of classes to use in classification
                    if None, takes the list of all classes found in path
                Note that the min of n_classes and len(label_classes) takes precedence
                i.e. if   len(label_classes)<=n_classes, use label_classes
                     elif len(label_classes)>n_classes, sample n_classes from label_classes
                - n_sample number of images per class to consider, 
                    Note that any classes w/ < n_sample images will be disregarded. 
                    If None, takes all images from each class
        Output:  - dataframe of labels & paths of all mages to be used in classification
    """
    
    # PATH -> DF
    filepaths = glob.glob(path)
    filepaths = [x for x in filepaths if os.path.splitext(x)[1]!='.json']
    filepaths_df = pd.DataFrame({'path': filepaths, 'label': [x.split('/')[2] for x in filepaths]})

    print("Found %d classes w/ %d images, total." % (len(filepaths_df.label.drop_duplicates()),                                                     len(filepaths_df.path)))
    print
    
    if not n_classes:
        n_classes = len(filepaths_df.label.drop_duplicates())
    if not label_classes:
        label_classes = filepaths_df.label.drop_duplicates().tolist()
        
    if len(label_classes) <= n_classes:
        # use label_classes
        classes = label_classes
    elif len(label_classes) > n_classes:
        # sample n_classes from label_clases
        classes = random.sample(label_classes,n_classes)

    filepaths_df_1 = filepaths_df[filepaths_df['label'].isin(classes)]
    filepaths_df_0 = filepaths_df[~filepaths_df['label'].isin(classes)]

    
    # subset DF to only those containing >= n_sample
    # & sample N_SAMPLE per group
    if n_sample: 
        lbls = cnt_df[cnt_df.path>=n_sample].label.tolist()
        mask = filepaths_df['label'].isin(lbls)
        filepaths_df_1 = filepaths_df[mask]
        filepaths_df_1 = filepaths_df_1.groupby('label').apply(lambda x:                                         x.sample(n_sample)).reset_index(drop=True)
    else:
        n_sample = len(filepaths_df_1.path)
        
    # if only one class (i.e. model is hd-not-hd)
    # sample same number of images from all images for negative class
    if len(classes)==1:
        filepaths_df_0 = filepaths_df_0.sample(n_sample)
        filepaths_df_0.label = "not_"+LABEL_CLASSES[0]
        filepaths_df_1 = filepaths_df_1.append(filepaths_df_0)
    
    cnt_df = filepaths_df_1.groupby('label', as_index=False)['path'].count(                                                ).sort_values(by=['path'], ascending=False)
    cnt_str = ' '.join(cnt_df.to_string(header=False,
                  index=False,
                  index_names=False).split())
   
    print("Classifying %d classes w/ %d images, total:"           % (len(cnt_df.label), sum(cnt_df.path) ))
    print("\t"+cnt_str)
    
    print
    return(filepaths_df_1) 

def load_imgs(df):
    """ Load files from  dataframe of image labels and file-locations.
        Drops from dataframe rows pertaining to failed image loads
    
        Input:  - dataframe of labels & paths of all mages to be used in classification
        Output: - dataframe of labels & paths of all mages to be used in classification
                - list of image arrays
    """
    images=[]
    for i in df.index:
        path = df.path[i]
        try:
            images.append(misc.imread(path))
        except:
            print "Failed to read in: %s, Dropping from dataframe" % path
            df = df.drop(i)
    print("Loaded:" )    
    print("\t %d images!" %len(images))
    print
    return(df, images)

def preproc_imgs(images, img_size):
    """ Preprocess list of image arrays 
        # (1) Scale image arrays s.t range is between 0 and 1 instea dof 0 and 255
        # (2) Resize to be of dim IMG_SIZE (width,height)
        # When the normType is NORM_MINMAX, cv::normalize normalizes _src in such a way that 
        #   the min value of dst is alpha and max value of dst is beta. cv::normalize does its magic 
        #   using only scales and shifts (i.e. adding constants and multiplying by constants).
        # (3) Drop fourth dimmension for PNG images
        # (4) create 3rd dim for greay scale imges
    
        Input:  - images list of image arrays
                - img_size tuple of new image size
        Output: - images list of preprocessed image arrays, with the above modifications
    """

    i=random.randint(0, len(images))
    
    images_sc = [None] * len(images)
    print "Preprocessed"
    for j in range(len(images)):
        if j % 250 == 0:
            print "\t %d images..." % j
        if images[j].all()==None:
            images_sc[j]=None
        else:
            try:
                temp = images[j]
                if len(temp.shape) > 2 and temp.shape[2] == 4: # PNG rgb images have 4 channels
                    temp = cv2.cvtColor(temp, cv2.COLOR_BGRA2BGR)
                elif len(temp.shape) > 2 and temp.shape[2] == 2: # PNG grsc images have 2 channels
                    temp = np.stack((temp[:,:,0],)*3, -1)
                elif len(temp.shape) == 2: # grsc images have 1 channel
                    temp = np.stack((temp,)*3, -1)
                temp = cv2.resize(temp.astype('uint8'), dsize=IMG_SIZE)
                temp = cv2.normalize(temp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,                                dtype=cv2.CV_32F, dst=None)
                images_sc[j] = temp
            except:
                print "Unexpected error:", sys.exc_info()[0]

    print "\t %d images!" % len(images_sc)
    print
    return (images_sc)

def create_testtrain(images, df, train_test_split):
    """ from dataframe image list, create x_ & y_ train and x_ & y_ test.
        
        Input:  - images list of preprocessed image arrays
                - df dataframe of labels & paths of all mages to be used in classification
                = train_test_split float = proportion of images to set aside for training
        Output: - x_train, y_train, x_test, y_test
    """
    n_images = len(images_sc)
    labels = df.label.tolist()
    paths = df.path.tolist()

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_Y = encoder.transform(labels)
    dummy_y = to_categorical(encoded_Y)

    split_index = int(train_test_split * n_images)
    shuffled_indices = np.random.permutation(n_images)
    train_indices = shuffled_indices[0:split_index]
    test_indices = shuffled_indices[split_index:]
    
    print("Split dataset:")

    # Split the images and the labels
    x_train = np.array([images_sc[i] for i in train_indices])
    y_train = np.array([dummy_y[i] for i in train_indices])
    paths_train = [paths[i] for i in train_indices]
    print("\t %d train images; %d train labels" %           (x_train.shape[0],y_train.shape[0]))
    print

    x_test = np.array([images_sc[i] for i in test_indices])
    y_test = np.array([dummy_y[i] for i in test_indices])
    paths_test = [paths[i] for i in test_indices]
    print("\t %d test images; %d test labels" %           (x_test.shape[0],y_test.shape[0]))
    
    return(x_train, y_train, x_test, y_test)

# Preprocess data
filepaths_df = create_filepathsdf(path = PATH, n_classes=N_CLASSES, label_classes= LABEL_CLASSES, n_sample=N_SAMPLE)
filepaths_df, images = load_imgs(df = filepaths_df)
images_sc = preproc_imgs(images = images, img_size = IMG_SIZE)
x_train, y_train, x_test, y_test = create_testtrain(images = images_sc, df = filepaths_df, train_test_split = TRAIN_TEST_SPLIT)


# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# #
# # MODEL FITTING
# # 
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

# Instantiate the model
image_size = x_train[0].shape
n_classes = y_train.shape[1]
img_rows, img_cols = IMG_SIZE

nb_filters_1 = 32 
nb_filters_2 = 64 
nb_filters_3 = 128 
nb_conv = 3 # kernel_size dim
nb_classes = y_train.shape[1]


model1 = Sequential()
model1.add(Conv2D(filters = nb_filters_1, 
                 kernel_size = (nb_conv,nb_conv),
                 padding = 'Same', 
                 activation ='relu', 
                 input_shape = image_size))
model1.add(Conv2D(filters = nb_filters_1, 
                 kernel_size = (nb_conv,nb_conv),
                 padding = 'Same', 
                 activation ='relu'))
model1.add(MaxPool2D(pool_size=(2,2)))
model1.add(Dropout(0.25))
model1.add(Conv2D(filters = nb_filters_2, 
                 kernel_size = (nb_conv,nb_conv),
                 padding = 'Same', 
                 activation ='relu'))
model1.add(Conv2D(filters = nb_filters_2, 
                 kernel_size = (nb_conv,nb_conv),
                 padding = 'Same', 
                 activation ='relu'))
model1.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model1.add(Dropout(0.25))
model1.add(Flatten())
model1.add(Dense(256, activation = "relu"))
model1.add(Dropout(0.5))
model1.add(Dense(nb_classes, activation = "softmax"))
model1.summary()

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model1.compile(optimizer = optimizer ,loss = "categorical_crossentropy", metrics=["accuracy"])

# Training Hyperparamters
EPOCHS = 50
BATCH_SIZE = 200

# Early stopping callback
PATIENCE = 10
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')

# Learning rate annealing callback
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# TensorBoard callback
LOG_DIRECTORY_ROOT = 'logdir'
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")

log_dir = "{}/run-{}/".format(LOG_DIRECTORY_ROOT, now)
tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)

callbacks = [learning_rate_reduction, early_stopping, tensorboard]

# Train the model
model1.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=2)

# Make a prediction on the test set
test_predictions = model1.predict(x_test)
test_predictions = np.round(test_predictions)
print stats.describe(test_predictions)
print
accuracy = accuracy_score(y_test, test_predictions)
print("Accuracy: " + str(accuracy))


# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# #
# # MODEL SAVING
# # 
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##


MODEL_DIRECTORY_ROOT = 'modeldir'
model_dir = "{}/run-{}/".format(MODEL_DIRECTORY_ROOT, now)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
 
save_model(model1, model_dir+'model.h5', overwrite=True,include_optimizer=True)

with open(model_dir+'histr.h5', 'wb') as file:
    pickle.dump(model1.history.history, file)
    
