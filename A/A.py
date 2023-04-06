import tensorflow as tf
import pandas as pd
import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.applications import VGG19
from keras.models import Model
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from keras.applications import VGG19
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

"""
import seaborn as sns
import datetime

#import cv2
from PIL import Image

from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications.vgg16 import VGG16
from collections import Counter
import itertools
import numpy as np
import matplotlib.pyplot as plt
import glob
import shutil
import json
"""

def evaluate(model, test, gen):
    """
    This function print evaluation reuslts of a model.
    Evaluation Metrics: Acc, Precision, Recall, AUC, F1 score
    Args:
        model: the model that need to be evaluated
        test: the testing set
        gen: the testing set created using data generator
    Retrun:
        acc: Accuracy of the model on testing set
    """
    predict = model.predict(gen)
    t = []
    for ele in predict:
        t.append(ele.argmax())
    
    test.label = test.label.astype("int")
    
    eva = tf.keras.metrics.Accuracy()
    eva.update_state(test["label"], t)
    print("The accuracy is", eva.result().numpy())
    acc = eva.result().numpy()

    eva = tf.keras.metrics.Precision()
    eva.update_state(test["label"], t)
    precision = eva.result().numpy()
    print("The precision is", precision)
    
    eva = tf.keras.metrics.Recall()
    eva.update_state(test["label"], t)
    recall = eva.result().numpy()
    print("The recall is", recall)

    eva = tf.keras.metrics.AUC()
    eva.update_state(test["label"], t)
    print("The AUC is", eva.result().numpy())
    
    
    print("The F1 score is", 2*precision*recall/(precision + recall))
    
    return acc


def data_preprocessing():
    """
    This function read the dataset and split to train, validation and test set.

    Retrun:
        train_generator: the training set created by ImageDataGenerator
        validation_generator: the traivalidation ning set created by ImageDataGenerator
        test_generator: the traivalidation ning set created by ImageDataGenerator
        test: a dataframe of testing set for evaluation
    """
    train_img_path = 'Datasets/train_images/'
    data = pd.read_csv('Datasets/train.csv')

    data.label = data.label.astype("str")
    train,temp = train_test_split(data, test_size = 0.2, stratify = data['label'])
    val,test = train_test_split(temp, test_size = 0.5)
    imgh = 224
    imgv = 224
    batchsize = 32

    train_datagen = ImageDataGenerator(preprocessing_function = None,
                                  rescale = 1./255,
                                  rotation_range = 45,
                                  zoom_range = 0.2,
                                  horizontal_flip = True,
                                  vertical_flip = True,
                                  fill_mode = 'nearest',
                                  shear_range = 0.1,
                                  height_shift_range = 0.1,
                                  width_shift_range = 0.1,)

    train_generator = train_datagen.flow_from_dataframe(dataframe = train,
                                                    directory = train_img_path,
                                                    x_col = 'image_id',
                                                    y_col = 'label',
                                                    target_size = (imgh,imgv),
                                                    batch_size = batchsize,
                                                    class_mode = 'sparse'
                                                    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = validation_datagen.flow_from_dataframe(val,
                            directory = train_img_path,
                            x_col = "image_id",
                            y_col = "label",
                            target_size = (imgh,imgv),
                            batch_size = batchsize,
                            class_mode = 'sparse')
    
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_generator = test_datagen.flow_from_dataframe(test,
                            directory = train_img_path,
                            x_col = "image_id",
                            y_col = "label",
                            shuffle = False,
                            target_size = (imgh,imgv),
                            batch_size = batchsize,
                            class_mode = 'sparse')
    return train_generator, validation_generator, test_generator, test

def createCNN():
    """
    This function create the CNN model.
    Retrun:
        model: the CNN model created
    """
    model = Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5, activation='softmax')
    ])
    model.compile(optimizer = 'adam',
                #loss = "categorical_crossentropy",
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])
    return model

def trainCNN(model, train_generator, validation_generator):
    """
    This function train the CNN model and tune with the validation set
    Args:
        model: the model that need to be trained
        train_generator: the training set
        validation_generator: the validation set for tuning
    Retrun:
        the accuracy of the trained model
    """
    earlystop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.001, 
                           patience = 5, mode = 'min', verbose = 1,
                           restore_best_weights = True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint("cassava_224_cnn.h5",
                                        save_best_only=True,
                                        monitor = 'val_loss',
                                        mode='min')
    re_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, 
                                patience = 2, min_delta = 0.001, 
                                mode = 'min', verbose = 1)
    his = model.fit(
        train_generator,
        epochs = 30,
        validation_data = validation_generator,
        callbacks = [earlystop,re_lr],
        )
    
    acc = 0
    for ele in his.history['accuracy']:
        if ele > acc:
            acc = ele
    return acc

def createVGG():
    """
    This function create the VGG19 model.
    Retrun:
        vgg19model: the vgg19 model created
    """

    vgg19 = VGG19(weights = 'imagenet',
                  include_top = False,
                  input_shape = (224, 224,3))
    for layer in vgg19.layers:
        layer.trainable = False
    def topmodel(bottom_model, num_classes):
        top_model = bottom_model.output
        top_model = tf.keras.layers.Flatten(name='flatten')(top_model)
        top_model = tf.keras.layers.Dense(512, activation='relu')(top_model)
        top_model = tf.keras.layers.Dense(1024, activation = 'relu')(top_model)
        top_model = tf.keras.layers.Dense(512, activation = 'relu')(top_model)
        top_model = tf.keras.layers.Dense(num_classes, activation='softmax')(top_model)
        return top_model
    vgg19model = Model(inputs=vgg19.input, outputs=topmodel(vgg19, 5))
    vgg19model.compile(optimizer = 'adam',
                #loss = "categorical_crossentropy",
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])

    return vgg19model

def trainVGG(vgg19model, train_generator, validation_generator):
    """
    This function train the VGG19 model and tune with the validation set
    Args:
        vgg19model: the model that need to be trained
        train_generator: the training set
        validation_generator: the validation set for tuning
    Retrun:
        acc: the accuracy of the trained model
    """
    earlystop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.001, 
                           patience = 5, mode = 'min', verbose = 1,
                           restore_best_weights = True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint("cassava_224_vgg.h5",
                                        save_best_only=True,
                                        monitor = 'val_loss',
                                        mode='min')
    re_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, 
                                patience = 2, min_delta = 0.001, 
                                mode = 'min', verbose = 1)
    history_vgg = vgg19model.fit(train_generator,
                   epochs = 30,
                   steps_per_epoch= 17117/32,
                   validation_data = validation_generator,
                   validation_steps = 2140/32 ,
                   callbacks = [earlystop,re_lr],
                   )
    acc = 0
    for ele in history_vgg.history['accuracy']:
        if ele > acc:
            acc = ele
    
    return acc

def createRES():
    """
    This function create the Res50 model.
    Retrun:
        resnet_model: the Res50 model created
    """
    resnet = ResNet50(include_top=False, input_shape=(224,224,3), weights='imagenet')
    for layer in resnet.layers:
        layer.trainable = False
        
    x = keras.layers.GlobalAveragePooling2D()(resnet.output)
    output = keras.layers.Dense(5, activation='softmax')(x)

    resnet_model = keras.models.Model(inputs=resnet.inputs, outputs=output)

    resnet_model.compile(
        #loss='categorical_crossentropy',
        loss = 'sparse_categorical_crossentropy',
        optimizer = Adam(learning_rate = 0.0005),
        metrics=['accuracy']
    )

    return resnet_model

def trainRES(resnet_model, train_generator, validation_generator):
    """
    This function train the Res50 model and tune with the validation set
    Args:
        resnet_model: the model that need to be trained
        train_generator: the training set
        validation_generator: the validation set for tuning
    Retrun:
        acc: the accuracy of the trained model
    """
    checkpoint = tf.keras.callbacks.ModelCheckpoint("cassava_224_res.h5",
                                        save_best_only=True,
                                        monitor = 'val_loss',
                                        mode='min')
    re_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, 
                                patience = 2, min_delta = 0.001, 
                                mode = 'min', verbose = 1)
    history_res = resnet_model.fit(
        train_generator,
        steps_per_epoch = 17117/32,
        epochs = 30,
        validation_data = validation_generator,
        validation_steps = 2140/32,
        callbacks = [re_lr]
    )

    acc = 0
    for ele in history_res.history['accuracy']:
        if ele > acc:
            acc = ele

    return acc


def createEFF():
    """
    This function create the EfficientNet B0 model.
    Retrun:
        model: the EfficientNet B0 model created
    """
    conv_base = EfficientNetB0(include_top = False, weights = None,
                               input_shape = (224, 224, 3))
    model = conv_base.output
    model = layers.GlobalAveragePooling2D()(model)
    model = layers.Dense(5, activation = "softmax")(model)
    model = Model(conv_base.input, model)
    model.compile(optimizer = "adam",
                  loss = "sparse_categorical_crossentropy",
                  metrics = ["accuracy"])
    return model


def trainEFF(effmodel, train_generator, validation_generator):
    """
    This function train the EfficientNet B0 model and tune with the validation set
    Args:
        effmodel: the model that need to be trained
        train_generator: the training set
        validation_generator: the validation set for tuning
    Retrun:
        acc = the accuracy of the trained model
    """
    earlystop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.001, 
                           patience = 5, mode = 'min', verbose = 1,
                           restore_best_weights = True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint("cassava_224_eff.h5",
                                        save_best_only=True,
                                        monitor = 'val_loss',
                                        mode='min')
    re_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, 
                                patience = 2, min_delta = 0.001, 
                                mode = 'min', verbose = 1)
    history_eff = effmodel.fit(
        train_generator,
        steps_per_epoch = 17117/32,
        epochs = 30,
        validation_data = validation_generator,
        validation_steps = 2140/32,
        callbacks = [ earlystop, re_lr]
    )
    acc = 0
    for ele in history_eff.history['accuracy']:
        if ele > acc:
            acc = ele
    return acc
