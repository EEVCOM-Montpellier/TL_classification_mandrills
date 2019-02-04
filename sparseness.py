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


RESULTS_DIR = "results/2019-02-01/"
MODEL_file = 'model_custom.json'
MODEL_weight = 'model_custom_weights.h5'
IMG_DATASET = 'datas/train-test_set/sex-clean01_v1.npz'

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
model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=0.001) )
model.summary()

#LOAD IMAGES_LOAD_DATASET
ts = np.load(IMG_DATASET)
X_train, X_test, y_train, y_test = ts['X_train'], ts['X_test'], ts['y_train'], ts['y_test']




#------------------- GET ACTIVATIVTIONS FOR EACH LAYER-------------------------

#100 FIRST IMAGES
layers_activations = keract.get_activations(model, X_train[0:100].reshape(100,224,224,3)) #177

if not os.path.exists(RESULTS_DIR + 'activ_dist/'):
    os.makedirs(RESULTS_DIR + 'activ_dist/')
    
    
#List of activation layer
r = re.compile(".*Relu")
activ_list = list(filter(r.match, [*layers_activations]))

#TAILLE DES LAYERS
for activ in activ_list: #parcourt les couches
    print(activ)
    print(layers_activations[activ].shape)
    if len(layers_activations[activ].shape) != 4 :
        activ_list.remove(activ)

#FUNCTIONS


def list_act_1_neur(layer, neur_x , neur_y , act_map):
    """Give list of all activities(100img) for 1 neuron for 1 layer for 1 act_map"""
    list_act = []
    layer_activ = layers_activations[layer] 
    for i in range(layer_activ.shape[0]):
        list_act.append(layer_activ[i, neur_x, neur_y, act_map])
    return(list_act)
    


def list_act_all_neur(layers_activations, layer, num_filter):
    """Give list of list all activities for all neuron for 1 layer for 1 act_map for 100 images"""
    directory = RESULTS_DIR +  'activ_dist/' + str(num_filter) + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    taille = layers_activations[layer].shape
    if ( taille[1] > 30) :
        pas = 4
    else:
        pas = 1
    list_out = []
    for x in range(0,taille[1], pas):
        for y in range(0,taille[1],pas):
            list_in = list_act_1_neur(layer = layer, neur_x = x , neur_y = y ,  act_map = num_filter)
            #plt.hist(list_in, bins ='auto' )
            #plt.savefig(directory + layer.split('/')[0] +  "-neurone" + '(' + str(x) + ','+ str(y) + ')')
            list_out.append(list_in)
            #plt.close()
    #plt.hist(list_out, bins ='auto' , histtype='stepfilled', alpha=0.3)
    #plt.savefig( directory + layer.split('/')[0] + 'all_neurons.png')
    #plt.close()
    return(list_out)
    




#----------------Lifetime Sparseness Measure --------------------------------

#conv2_2 : (100, 112, 112, 128)

#list pour chaq neuron du filtre de réponse de 100 images 
layer_act_all_neur = list_act_all_neur(layers_activations, 'conv2_2/Relu:0', 1)

# Lifetime kurtosis des 784 neurones sur 12544 pour 1 img


    
#TEST UNIMODAL DIST?????
def lifetime_kurtosis(layer_act_all_neur, neuron):
    """Calculate Lifetime kurtosis 
    M stimuli for 1 neurone
    layer_act_all_neur: list with activities reponse of all neurones for multiples images 
    neuron: int num of neuron of the list
    return: KL value(high value of KL = sparse signal"""
    layer_act_all_neur[neuron]
    M = len(layer_act_all_neur[neuron]) #nb stimuli(nb img)
    KL= 0
    for i in range(M):
        KL += ((layer_act_all_neur[neuron][i] - np.mean(layer_act_all_neur[neuron]))/np.std(layer_act_all_neur[neuron]))**4
    KL = KL/M - 3
    return(KL)
    
def KL_mulitple_neurons(layer_act_all_neur):
    """List all KL of each neurons in a given layer"""
    all_KL = []
    for neuron in range(len(layer_act_all_neur)):
        all_KL.append(lifetime_kurtosis(layer_act_all_neur,neuron))
    return(all_KL)

#KL pour tous les neurones de la couche 2_2
all_KL_conv2_2 = KL_mulitple_neurons(layer_act_all_neur)
min(all_KL_conv2_2 )
max(all_KL_conv2_2 )
np.mean(all_KL_conv2_2 )
np.var(all_KL_conv2_2 )

#----------- Population Sparseness Measures-----------------------------------

img = 1

def population_kurtosis(layer_act_all_neur, img):
    """Calculate Population kurtosis
    1 stimuli(1img) for all neurons in the layer
    layer_act_all_neur: list with activities reponse of all neurones for multiples images 
    img: int num of img 
    return: KP value(high value of KP = sparse signal"""
    KP = 0
    N = len(layer_act_all_neur) #nb neurons
    for neur in range(N): #all neurones in layer
        all_act = [neur[img] for neur in layer_act_all_neur] #response of all neurones for given img
        r = layer_act_all_neur[neur][img]
        KP += ((r - np.mean(all_act))/np.std(all_act))**4
    KP = KP/M - 3
    return(KP)


def KP_mulitple_imgs(layer_act_all_neur):
    """List all KP of each img(signal) in a given layer"""
    all_KP = []
    for img in range(len(layer_act_all_neur[0])):
        all_KP.append(population_kurtosis(layer_act_all_neur, img))
    return(all_KP)
    

def population_TrevesRolls(layer_act_all_neur, img):
    """Calculate Population TrevesRolls
    1 stimuli(1img) for all neurons in the layer
    layer_act_all_neur: list with activities reponse of all neurones for multiples images 
    img: int num of img 
    return: """
    num = 0
    denom = 0
    N = len(layer_act_all_neur) #nb neurons
    for neur in range(N): #all neurones in layer
        r = layer_act_all_neur[neur][img]
        num = r/N
        denom = (r*r)/N
    ST = (num**2)/denom
    return(ST)



#KP pour plusieurs IMGs
all_KP_conv2_2 = KP_mulitple_imgs(layer_act_all_neur)




#Dict with sparseness(KP,KL)  for each layer 
sparseness_dict=dict((el,(0,0)) for el in activ_list)
for k in sparseness_dict.keys():
    print(layers_activations[k].shape)
    print(k)
    layer_act_all_neur = list_act_all_neur(layers_activations, k, 1)
    allKL = KP_mulitple_imgs(layer_act_all_neur)
    allKP = KP_mulitple_imgs(layer_act_all_neur)
    sparseness_dict[k]=(allKL,allKP)


#save dict 
json = json.dumps(sparseness_dic)
f = open(RESULTS_DIR  + "dict_sparseness_perlayer.json","w")
f.write(json)
f.close()