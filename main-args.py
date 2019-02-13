
# import the necessary packages
from keras.applications import ResNet50
from keras.applications import VGG16
from keras.applications import VGG19
from keras_vggface.vggface import VGGFace
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input
from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
AveragePooling2D, Reshape, Permute, multiply, Dropout
from keras.callbacks import Callback
import numpy as np
import argparse
import cv2
import csv
from keras import optimizers

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import pandas as pd
import keras.backend as K

import sys
import time, datetime, os
import matplotlib.pyplot as plt

from LR_adam import  Adam_lr_mult
import _pickle as pickle

from keras import backend as K
from keras.callbacks import TensorBoard
import tensorflow as tf
import json
#-------------------------------------------------------------------------------


# construct the argument parse and parse the arguments
#--image : path to our input image that we wish to classify
# --model :

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-model", "--model", type=str, default="resnet_rcmalli",
	help="name of pre-trained network to use")
ap.add_argument("-classification", "--classification", type=str, default="sex",
	help="name of pre-trained network to use")
args = vars(ap.parse_args())


#-------------------------------------------------------------------------------

# define a dictionary that maps model names to their classes
# inside Keras
MODELS = {
	"vgg16": VGG16,
	"vgg19": VGG19,
	"resnet": ResNet50,
    "resnet_rcmalli": VGGFace,
    "vgg16_rcmalli": VGGFace
}

# esnure a valid model name was supplied via command line argument
if args["model"] not in MODELS.keys():
	raise AssertionError("The --model command line argument should "
		"be a key in the `MODELS` dictionary")

# initialize the input image shape (224x224 pixels) along with
# the pre-processing function (this might need to be changed
# based on which model we use to classify our image)
inputShape = (224, 224)

print("Charging dataset")
#tags_pict = pd.read_csv('datas/clean_pict_tags.csv')
#dirname = "C:/Users/renoult/Documents/BDD_PHOTOS_MANDRILLUS_FACES/MANDRILLS_BKB"
NB_EPOCHS = 10

#-------------------------------------------------------------------------------

# /!\ Metre args : model_choice

def my_model(model_choice):
    # pour le moment essai seulement sur resnet_rcmalli
    Network = MODELS[model_choice]
    if model_choice == "resnet_rcmalli" :
        model = Network(model='resnet50') # architecture renet50 with vggface_weight
    if model_choice == "vgg16_rcmalli" :
        model = Network(model='vgg16') # architecture vgg16 with vggface_weight
    else :
        model= Network(weights="imagenet") #other architectures with imagenet_weight
    return(model)


def train_test_type(Y):
    return {
        "feminity" : np.load('datas/train_test_feminity/sex-feminity-clean01.npz'),
        "sparseness" : np.load('datas/train_test_sparseness/indiv-sparseness-clean01-train.npz'),
        "sex": np.load('datas/train-test_set/sex-clean01_v1.npz'),
		"stage": np.load('datas/train-test_set/stage-clean01_v1.npz'),
		"indiv": np.load('datas/train-test_set/indiv-clean01_v1.npz')
    }.get(Y)


def train_test_type_mini(Y):
    return {
        "sex":  np.load('datas/train-test_set/sex-mini_clean01_v1.npz'),
		"stage": np.load('datas/train-test_set/stage-mini_clean01_v1.npz'),
		"indiv": np.load('datas/train-test_set/indiv-mini_clean01_v1.npz')
    }.get(Y)


class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir):  # add other arguments to __init__ if you need
        self.batch_writer = tf.summary.FileWriter(log_dir)
        self.step = 0
        super().__init__(log_dir=log_dir)
    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


#Train the model using tensorboard instance in the callbacks

def classif(classifytype, model, mini=True):
    if mini == True:
        ts = train_test_type_mini(classifytype)
    else:
        ts = train_test_type(classifytype)
    if classifytype == "sparseness":
        X_train, y_train = ts['X_train'], ts['y_train']
    else:
        X_train, X_test, y_train, y_test = ts['X_train'], ts['X_test'], ts['y_train'], ts['y_test']
    print("End: Split dataset into train et test set")
    x,y = shuffle(X_train, y_train)
    X_learn, X_val, y_learn, y_val = train_test_split(x, y, test_size=0.1)
    ## A REVOIR ICI !!!!!
    #Change last_layer (  softmax depends on class_type)
    ## /!\ add dropout !!!!
    model.layers.pop()
    model.layers.pop()
    model.summary()
    #Add the fully-connected layers
    out = Dropout(0.7)(model.layers[-1].output)
    out = Dense(y_train.shape[1], activation='softmax', name='predictions')(out)
    #Create your own model
    #out = Dense(y_train.shape[1], activation='softmax',name='output_layer')(model.layers[-1].output)
    model_custom = Model( input=model.input, outputs = out)
    print("Architecture custom")
    print(model_custom.summary())
    #Fine-tune: Freeze : CB1 and CB2 (37 firsts layers) #resnet
    #F#Fine-tune: Freeze : CB1 and CB2 (7 firsts layers) #vgg16
    #FCheck the trainable status of the individual layers
    for layer in model_custom.layers[:7]:
        layer.trainable = False
    for layer in model_custom.layers:
        print(layer, layer.trainable)
    #change lr for adam optim
    # Changing learning rate multipliers for different layers :
    # initial : 10e-4
    # apres block4 : 10e-3
    #learning_rate_multipliers = {}
    #learning_rate_multipliers['conv1/7x7_s2'] = 1
    #learning_rate_multipliers['conv4_1_1x1_reduce'] = 10
    #adam_with_lr_multipliers = Adam_lr_mult(multipliers=learning_rate_multipliers)
    #lr_metric = get_lr_metric(adam_with_lr_multipliers)
    #lr_metric = get_lr_metric(optimizers.Adam(lr=0.001))
    #model_custom.compile(loss='categorical_crossentropy',optimizer=adam_with_lr_multipliers, metrics=['accuracy' , lr_metric])
    #learn compile
    model_custom.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'] )
    tensorboard =  LRTensorBoard(log_dir= "results/" + todaystr + '/logs/' )
    #hist = model_custom.fit(x = X_train, y = y_train, batch_size=32, epochs=NB_EPOCHS, verbose=1, validation_split=0.1 ,  callbacks=[tensorboard])
    hist = model_custom.fit(x = X_learn, y = y_learn, batch_size=64, epochs=NB_EPOCHS, verbose=1, validation_data=(X_val, y_val),  callbacks=[tensorboard])
    #(loss, accuracy, lr) = model_custom.evaluate(X_test, y_test, batch_size=10, verbose=1)
    if (X_test and y_test) in locals():
        (loss, accuracy) = model_custom.evaluate(X_test, y_test, batch_size=16, verbose=1)
        print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
    model_json = model_custom.to_json()
    with open('results/' + todaystr + '/model_custom.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model_custom.save_weights('results/' + todaystr + "/model_custom_weights.h5")
    print("Saved model to disk")
    print("Loaded model from disk")
    return(hist)


def plot_loss_acc(hist , todaystr):
    save_path = "results/" + todaystr + "/"
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    #train_lr=hist.history['lr']
    #val_lr=hist.history['val_lr']
    xc=range(NB_EPOCHS)
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    plt.savefig(save_path + 'loss.png')
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
    plt.savefig(save_path + 'acc.png')



#-----------------------MAIN-----------------------------------------
today =  datetime.datetime.now()
todaystr = today.isoformat()
os.mkdir("results/" + todaystr)
log_file_path = "results/" + todaystr + "/log_file.txt"
sys.stdout = open(log_file_path, 'w')

params_file_path =  "results/" + todaystr + "/params.txt"


#tags_pict = pd.read_csv('datas/clean_pict_tags.csv')
#tags_pict = pd.read_csv('datas/minitest_pict_tags.csv')
#tags_pict['sex'] = tags_pict['sex'].astype('category')
#tags_pict['stage'] = tags_pict['stage'].astype('category')
#tags_pict['indiv'] = tags_pict['indiv'].astype('category')
# img_data = load_images(dirname,tags_pict) ON VIRE AJOUT LOAD SELON LA QUALITE DESIRE
#
#model_choice = "resnet_rcmalli"
#classifytype = "sex"


model_choice = args["model"]
model= my_model(model_choice)
print("Architecture model basis")
print(model.summary())
historique = classif(args["classification"], model, mini=False)
print(historique.history)
print(historique.params)

plot_loss_acc(historique, todaystr)


f = open(params_file_path,'w')
f.write("model:" +args["model"] + "\n")
f.write("classification:" + args["classification"] + "\n")
f.write("weights:" + "VGGface" + "\n")
f.write("epochs :" + str(NB_EPOCHS) + "\n")
f.write("batch_size :" + str(64) + "\n" )
f.write("optim :" + "adam" + "\n")
f.write("lr :" + str(0.0001) + "\n")
f.write("freeze :" + "first 7 layers" + "\n")
f.write("dropout :" + "add 0.7 bef last layer" + "\n")
#f.write(json.dumps(historique.history))
f.close()


#------------------------------------------------------------------------
