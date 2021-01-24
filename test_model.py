# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io as sio
import os

import argparse

from keras.applications.vgg19  import VGG19
from keras.applications.vgg16  import VGG16
import requests

from keras.backend import expand_dims

from keras.models import Model
from keras.utils import Sequence
from keras.layers import Dense, GlobalAveragePooling2D,Flatten,add,concatenate,Dropout,LSTM,Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.layers import Input
from keras import callbacks
from keras.optimizers import Adam
from pyglet.gl.glext_arb import struct__cl_event
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import pickle
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.models import  load_model
import sys

IMAGE_FOLDER = "./images/"
MODEL_FOLDER = "./models/"
def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_google_drive_file(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def download_model(scenarioName,modelName,foldId):
    df = pd.read_csv("models/modelLinks.csv")
    fileName = "{}_{}_{}_network.h5".format(scenarioName,modelName,foldId)
    print(fileName)
    fileID = df[df["Name"]==fileName]["Id"].values[0]
    print(fileID)
    if not os.path.exists(os.path.join(MODEL_FOLDER,fileName)):
        print("Downloading Model " + fileName)
        download_google_drive_file(fileID, MODEL_FOLDER+fileName)
        print("Finished downloading Model.")
    return fileName


def ModelVGG16():
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)

    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in model.layers:
        layer.trainable = True

    opt = Adam(learning_rate=0.00002)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def ModelVGG19():
    base_model = VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in model.layers:
        layer.trainable = True
    opt = Adam(learning_rate=0.00002)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def ConcatedModel():
    vg_model_16 = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    vg_model_19 = VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    x16 = vg_model_16.output
    x16 = GlobalAveragePooling2D()(x16)

    x19 = vg_model_19.output
    x19 = GlobalAveragePooling2D()(x19)

    addLayer = concatenate([x16, x19])

    x = Dense(1024, activation='relu')(addLayer)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(2, activation='softmax')(x)

    for layer in vg_model_16.layers:
        layer.trainable = True
    for layer in vg_model_19.layers:
        layer.trainable = True

    for layer in vg_model_19.layers:
        layer._name = layer._name + str('_C')

    vg_model = Model(inputs=[vg_model_16.input, vg_model_19.input], outputs=output)
    opt = Adam(learning_rate=0.00002)
    vg_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return vg_model

def parse_args():
    parser = argparse.ArgumentParser(description='Testing trained models')
    parser.add_argument('--model', default='VGG16', help="'VGG16', 'VGG19', or 'ConcatedModel'.")
    parser.add_argument('--scenario', default='S1', help="'S1', 'S2','S3' or 'S4'.")
    parser.add_argument('--fold', default='1', help="'1', '2','3','4' or '5' ")
    parser.add_argument('--imageName', default='1.PNG', help="'Image Name'")
    args = parser.parse_args()
    return args.model,args.scenario,args.fold,args.imageName

def BaslikAyarla(klasor,model):
  if (model == "ConcatedModel"):
    return "Proposed Model - "+klasor+"  "
  else:
    return model+" - "+klasor

def classLabels(ScenarioName):
  if ScenarioName == "S1":
    return ["{IBM,PM,DM}","Normal"]
  elif ScenarioName == "S2":
    return ["IBM","Normal"]
  elif ScenarioName == "S3":
    return ["IBM","{PM,DM}"]
  elif ScenarioName == "S4":
    return ["IBM","PM","DM","Normal"]

def ActualClass(ScenarioName,imageName):
    df = pd.read_excel("OrginalAndCroppedFileNames.xlsx")
    label = df[df['Cropped Image Name']=="1.PNG"].values[0][6]
    if label == "D":
        return "DM"
    if label == "P":
        return "PM"
    if label == "I":
        return "IBM"
    if label == "N":
        return "Normal"


IMAGE_FOLDER = "./images"
FILE_LIST_FOLDER = "./FileList"

FOLDS = 5
BATCH_SIZE = 32
IMG_SIZE = (224,224)

if __name__ == "__main__":
    modelType,scenario,fold,imageName = parse_args()
    modelName = download_model(scenario,modelType,fold)
    print(MODEL_FOLDER+modelName)
    if (modelType=="VGG16"):
        model = ModelVGG16()
        print(model.summary())
        model.load_weights(MODEL_FOLDER+modelName)
    elif (modelType=="VGG19"):
        model = ModelVGG19()
        model.load_weights(MODEL_FOLDER+modelName)
    elif (modelType=="ConcatedModel"):
        model = ConcatedModel()
        model.load_weights(MODEL_FOLDER+modelName)
    img = image.load_img(os.path.join(IMAGE_FOLDER, imageName), target_size=(224, 224, 3))
    img = image.img_to_array(img)

    x_test =np.expand_dims(np.array(img),axis=0)
    x_test = x_test / 255.

    if (modelType=="VGG16" or modelType=="VGG19"):
        predictions = model.predict(x_test)
    elif (modelType=="ConcatedModel"):
        predictions = model.predict([x_test, x_test])
    classLabels = classLabels(ScenarioName=scenario)
    i = 0
    for cls in classLabels:
        print(cls, ":", predictions[0][i])
        i+=1
    print("Actual Class :",ActualClass(scenario,imageName))
    print("Predicted Class :", classLabels[np.argmax(predictions[0])])




















