
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
    "resnet_rcmalli": VGGFace
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
tags_pict = pd.read_csv('datas/clean_pict_tags.csv')
dirname = "C:/Users/renoult/Documents/BDD_PHOTOS_MANDRILLUS_FACES/MANDRILLS_BKB"
NB_EPOCHS = 5


#-------------------------------------------------------------------------------

def load_images(dirname,tags_pict):
    """Load each image into list of numpy array and transform into array
    ----------
    dirname : path to folder with pictures
    tags_pict : pandas dataframe with annotation for pict
    Returns : array
    """
    img_data_list = []
    for p in tags_pict.index :
        # our input image is now represented as a NumPy array of shape
        # (inputShape[0], inputShape[1], 3) however we need to expand the
        # dimension by making the shape (1, inputShape[0], inputShape[1], 3)
        # so we can pass it through thenetwork
        img_path = dirname + '/' + tags_pict.Folder[p] + '/' + tags_pict.pict[p]
        img = load_img(img_path, target_size= inputShape)
        x = img_to_array(img)
        x = np.expand_dims(img, axis=0)
        # pre-process the image using the appropriate function based on the
        # model that has been loaded (i.e., mean subtraction, scaling, etc.)
        x = preprocess_input(x)
        img_data_list.append(x)
    img_data = np.array(img_data_list)
    img_data=np.rollaxis(img_data,1,0)
    img_data=img_data[0]
    print("End : load images")
    np.save('results/img_data_load',img_data)
    return(img_data)


# /!\ Metre args : model_choice

def my_model(model_choice):
    # pour le moment essai seulement sur resnet_rcmalli
    Network = MODELS[model_choice]
    if model_choice == "resnet_rcmalli" :
        model = Network(model='resnet50') # architecture renet50 with vggface_weight
    else :
        model= Network(weights="imagenet") #other architectures with imagenet_weight
    return(model)


# /!\ Metre args pour que l'utilisateur choisisse son type de classification
def classify_type(Y):
    # RETURN ARRAY Y to be encoded
    return {
        # convert class labels to on-hot encoding
        # tags_pict['codes'] = tags_pict.stage.cat.codes
        # labels = np.asarray(list(tags_pict['codes']), dtype ='int64' )
        "sex":np_utils.to_categorical(np.asarray(list(tags_pict.sex.cat.codes), dtype ='int64' ), len(tags_pict.sex.cat.categories)),
		"stage":np_utils.to_categorical(np.asarray(list(tags_pict.stage.cat.codes), dtype ='int64' ), len(tags_pict.stage.cat.categories)),
		"indiv":np_utils.to_categorical(np.asarray(list(tags_pict.indiv.cat.codes), dtype ='int64' ), len(tags_pict.indiv.cat.categories))
    }.get(Y)


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


def classif(classifytype , img_data, model):
    """Construct netw and classify
    classifytype : str
        classifytype = ["sex","stage","indiv"]
    img_data : all loading images
    """
    num_of_samples = tags_pict.shape[0]
    Y = classify_type(classifytype)
    #Make training and testing set
    #randomize array item order for numpy array in unison
    x,y = shuffle(img_data,Y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    print("End: Split dataset into train et test set")
    ## A REVOIR ICI !!!!!
    #Change last_layer ( add dropout + softmax depends on class_type)
    #model.layers.pop()
    #model.layers.pop()
    #x = model.layers[-1].output
    #x = Dropout(0.5)(x)
    #out =  Dense(Y.shape[1], activation='softmax',name='custom_output_layer')(x)
    ## (bug to fix) : add dropout !!!!
    out = Dense(Y.shape[1], activation='softmax',name='output_layer')(model.output)
    model_custom = Model(input=model.input, outputs=out)
    print("Architecture custom")
    print(model_custom.summary())
    #Fine-tune: Freeze : CB1 and CB2 (37 firsts layers)
    for layer in model_custom.layers[:37]:
        layer.trainable = False
    #change lr for adam optim
    # Changing learning rate multipliers for different layers :
    # initial : 10e-4
    # apres block4 : 10e-3
    learning_rate_multipliers = {}
    learning_rate_multipliers['conv1/7x7_s2'] = 1
    learning_rate_multipliers['conv4_1_1x1_reduce'] = 10
    adam_with_lr_multipliers = Adam_lr_mult(multipliers=learning_rate_multipliers)
    #lr_metric = get_lr_metric(adam_with_lr_multipliers)
    lr_metric = get_lr_metric(optimizers.Adam(lr=0.0001))
    #model_custom.compile(loss='categorical_crossentropy',optimizer=adam_with_lr_multipliers, metrics=['accuracy' , lr_metric])
    #learn compile
    model_custom.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy' , lr_metric])
    hist = model_custom.fit(X_train, y_train, batch_size=64, epochs=NB_EPOCHS, verbose=1, validation_data=(X_test, y_test))
    (loss, accuracy, lr) = model_custom.evaluate(X_test, y_test, batch_size=10, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
    return(hist)



def plot_loss_acc(hist , todaystr):
    save_path = "results/" + todaystr + "/"
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    train_lr=hist.history['lr']
    val_lr=hist.history['val_lr']
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
    plt.figure(3,figsize=(7,5))
    plt.plot(xc,train_lr)
    plt.plot(xc,val_lr)
    plt.xlabel('num of Epochs')
    plt.ylabel('learning rate')
    plt.title('train_lr vs val_lr')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
    plt.savefig(save_path + 'lr.png')



#-----------------------MAIN-----------------------------------------
today = datetime.date.today()
todaystr = today.isoformat()
os.mkdir("results/" + todaystr)
log_file_path = "results/" + todaystr + "/log_file.txt"
sys.stdout = open(log_file_path, 'w')

params_file_path =  "results/" + todaystr + "/params.txt"


#tags_pict = pd.read_csv('datas/clean_pict_tags.csv')
tags_pict = pd.read_csv('datas/minitest_pict_tags.csv')
tags_pict['sex'] = tags_pict['sex'].astype('category')
tags_pict['stage'] = tags_pict['stage'].astype('category')
tags_pict['indiv'] = tags_pict['indiv'].astype('category')
img_data = load_images(dirname,tags_pict)
model_choice = args["model"]
model= my_model(model_choice)
print("Architecture model basis")
print(model.summary())
historique = classif(args["classification"], img_data, model)
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
f.write("lr :" + str(0.0001) + " evolue \n")
#f.write(pickle.dumps( historique.params))
f.close()
