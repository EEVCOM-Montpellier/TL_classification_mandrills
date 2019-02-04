import pandas as pd
import numpy as np

import csv
import re
import os

from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import np_utils

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#check curr dir
os.getcwd()

#----------- VARIABLES---------------------------------------------
# Folder with all pictures
dirname = "C:/Users/renoult/Documents/BDD_PHOTOS_MANDRILLUS_FACES/MANDRILLS_BKB"

#Doc excel with pictures annotations
tags_pict_path = "C:/Users/renoult/Documents/BDD_PHOTOS_MANDRILLUS_FACES/Tags_240119.csv"

inputShape = (224, 224)


#----------- FUNCTIONS---------------------------------------------


def list_of_pict(dirName):
    """Get the list of all files in directory tree at given path"""
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        #listOfFiles.append([os.path.join(file) for file in filenames])
        for file in filenames:
            listOfFiles.append(os.path.join(file))
    return(listOfFiles)

#---------if we only have folder and file name (but useless)---------------------

def dict_of_ind(dirname):
    """Dict with {idiNDEX:{Sex: male/femelle/unknown},
                            {Stage : juv...},
                            {Pict : list of all pictures for the individu}
    """
    list_indiv = os.listdir(dirname)
    #clean list
    list_indiv = [ ind for ind in list_indiv if ind[0]!='.' ]
    dictOfInd = {}
    for i in list_indiv:
        indiv = int(i.split('_')[0])
        dictOfInd[i] = {}
        dictOfInd[i]['sex'] = i.split('_')[1]
        dictOfInd[i]['stage'] = i.split('_')[2]
        dictOfInd[i]['pict'] = os.listdir(dirname + "/" + i)
    return(dictOfInd)




def clean_pict(x):
    """Remove from dataset images (depends on quality).
    x :0,1,2
        0: remove images labelled 1FaceQualNoEval and 1FaceQual0
        1:remove images labelled 1FaceQualNoEval and 1FaceQual0 and 1FaceQual1 ...
    Returns : df ( if not choose any option, return the entire dataset)
    """
    #ATTENTION ESPACE, A AMELIORER ICI
    return {
        0: tags_pict[tags_pict.qual != (" 1FaceQualNoEval" and " 1FaceQual0")],
        1: tags_pict[tags_pict.qual != (" 1FaceQualNoEval" and " 1FaceQual0" and
                                            " 1FaceQual1")],
        2: tags_pict[tags_pict.qual != (" 1FaceQualNoEval" and " 1FaceQual0" and
                                            " 1FaceQual1"  and " 1FaceQual2")]
    }.get(x, tags_pict)



def load_images(dirname,tags_pict, savefile):
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
    np.save(savefile ,img_data)
    return(img_data)



def classify_type(Y, tags_pict):
    # RETURN ARRAY Y to be encoded
    return {
        # convert class labels to on-hot encoding
        # tags_pict['codes'] = tags_pict.stage.cat.codes
        # labels = np.asarray(list(tags_pict['codes']), dtype ='int64' )
        "sex":np_utils.to_categorical(np.asarray(list(tags_pict.sex.cat.codes), dtype ='int64' ), len(tags_pict.sex.cat.categories)),
		"stage":np_utils.to_categorical(np.asarray(list(tags_pict.stage.cat.codes), dtype ='int64' ), len(tags_pict.stage.cat.categories)),
		"indiv":np_utils.to_categorical(np.asarray(list(tags_pict.indiv.cat.codes), dtype ='int64' ), len(tags_pict.indiv.cat.categories))
    }.get(Y)



def make_train_test_set(classifytype, tags_pict, img_data,  tablename):
    num_of_samples =img_data.shape[0]
    Y = classify_type(classifytype, tags_pict)
    #Make training and testing set
    #randomize array item order for numpy array in unison
    x,y = shuffle(img_data,Y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    dirfolder = "datas/train-test_set/" + classifytype + "-" +  tablename
    np.savez(dirfolder , X_train = X_train , X_test=X_test, y_train=y_train, y_test=y_test )



def train_test_set_all( tags_pict, img_data,  tablename ):
    for c in ["sex", "stage", "indiv"] :
        make_train_test_set(c, tags_pict , img_data, tablename)

#----------- ANNOTATIONS---------------------------------------------

#lIST OF PICT
list_julien = list_of_pict("C:/Users/renoult/Documents/Mandrill Picture Analyses/learn_all_mini12")
list_pict_BKB = list_of_pict(dirname)
dict_ind_BKB = dict_of_ind(dirname)

#RE-FORMAT AND ANNOTATION CSV
tags_pict = pd.read_csv(tags_pict_path , sep = ';')
tags_pict.rename(columns={"Nom du fichier": "pict" , "Tag2" : "qual"} , inplace=True)
#tags_pict['qual'] = tags_pict['qual'].astype('category')
tags_pict['indiv'] = [i.split('_')[0] for i in list(tags_pict['Folder']) ]
tags_pict['indiv'] = tags_pict['indiv'].astype('category')
#tags_pict['sex'] = [i.split('_')[1] for i in list(tags_pict['Folder'])]
tags_pict['sex'] =[ match.group(1)  if match else "unknown" for match in [re.search(".*(fem|mal|unknown).*", l) for l in list(tags_pict['Folder']) ]]
tags_pict['sex'] = tags_pict['sex'].astype('category')

#tags_pict['stage'] = [i.split('_')[2] for i in list(tags_pict['Folder']) ]
tags_pict['stage'] =[ match.group(1)  if match else "unknown" for match in [re.search(".*(adu|ado|juv|inf|sub).*", l) for l in list(tags_pict['Folder']) ]]
tags_pict['stage'] = tags_pict['stage'].astype('category')


#Check if images are in dataset
tags_pict = tags_pict[tags_pict.pict.isin(list_pict_BKB)]

#DATASET WITHOUT CLEANING
#shape : 8849 * 8
#DATASET WITHOUT noeval - 0 : (8781, 8)
#DATASET WITHOUT noeval - 0 - 1 : (8235, 8)
#DATASET WITHOUT noeval - 0 - 1 - 2 : (4387, 8)

#Save to csv
#clean_pict(0).to_csv('datas/clean0_pict_tags.csv')
#clean_pict(1).to_csv('datas/clean01_pict_tags.csv')
#clean_pict(2).to_csv('datas/clean012_pict_tags.csv')


# prend du temps
#load_images(dirname,clean_pict(2), 'datas/img_data_load_012')
#load_images(dirname,clean_pict(0),'datas/img_data_load_0')
#load_images(dirname,clean_pict(1), 'datas/img_data_load_01')


tags_pict  = clean_pict(1)
load_pict = np.load("datas/img_data_load_01.npy")
train_test_set_all( tags_pict , load_pict, "clean01_v1" )
train_test_set = np.load('datas/train-test_set/sex-clean01_v1.npz')

tags_pict = clean_pict(2)
mini_jdd = tags_pict.sample(frac=0.1, replace=False, random_state=1)
mini_jdd.to_csv('datas/minitest_pict_tags_012.csv')
img_data_mini = load_images(dirname,mini_jdd, 'datas/img_data_load_012_minitest')
train_test_set_all( mini_jdd , img_data_mini, "mini_clean012_v1" )
