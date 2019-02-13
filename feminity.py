# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 11:19:21 2019

@author: sonia
"""

#------------------------Import-----------------------------------------------
import os
from keras.models import model_from_json
import numpy as np
import time
from keras import backend as K
import seaborn as sns
from keras import optimizers
from keras.preprocessing import image
import matplotlib.pyplot as plt
import seaborn as sns
import keract
import re
import json


#----------------Global Var --------------------------------------------------


RESULTS_DIR = "results/2019-02-11/"
MODEL_file = 'model_custom.json'
MODEL_weight = 'model_custom_weights.h5'
IMG_DATASET = 'datas/train_test_feminity/sex-feminity-clean01.npz'

inputShape = (224, 224)

#------------------Load Model ------------------------------------------------
# load json and create model
json_file = open(RESULTS_DIR + MODEL_file, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(RESULTS_DIR + MODEL_weight)
#Re-compilation of model
model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=0.0001))
model.summary()

#LOAD IMAGES_LOAD_DATASET
ts = np.load(IMG_DATASET)
X_train, X_test, y_train, y_test = ts['X_train'], ts['X_test'], ts['y_train'], ts['y_test']



#----------EVALUATE MODEL ------------------------------------------------------

scores=model.evaluate(x=X_test,y= y_test)
print (scores[1])

#---------- PREDICT MODEL ------------------------------------------------------
prediction=model.predict(X_test)

pred_fem = [row[1] for row in prediction]

#dstribution des probas de prédictions de feminités
plt.hist(pred_fem)
plt.show()

#------- IMPORT ATTRACTIVITY DATASET -------------------------------------------
#Dataset with pictures of female with attractiveness score
df = pd.read_csv('datas/train_test_feminity/attract_vars.csv')
dir = "C:/Users/renoult/Documents/BDD_PHOTOS_MANDRILLUS_FACES/MANDRILLS_BKB/"

df['path'] = dir + df['Folder'].astype(str) +'/' + df['pict'].astype(str)

df['proba_fem'] = pred_fem
# Attractiveness label

df['stage'] = tags_pict['stage'].astype('category')


attract = ['Freq_TO_all', 'Freq_TO_alpha', 'Freq_monte_all',
        'Freq_monte_alpha', 'Prop_prox_all', 'Prop_prox_alpha']

import matplotlib
L=[]
for a in attract:
    choice = df[a]
    matplotlib.style.use('ggplot')
    plt.scatter(pred_fem, choice)
    plt.show()
    L.append(np.corrcoef(pred_fem, choice))
