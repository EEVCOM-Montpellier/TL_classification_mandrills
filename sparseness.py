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

truc=[]
#TAILLE DES LAYERS
for activ in  sorted(activ_list): #parcourt les couches
    print(activ)
    print(layers_activations[activ].shape)
    if len(layers_activations[activ].shape) != 4 :
        activ_list.remove(activ)


#FIND BEST PAS
        
    
def find_pas(layers_activations, layer):
    taille = layers_activations[layer].shape
    if (taille[1] > 100) :
        pas = 64
    if (taille[1] > 100) :
        pas = 32
    elif (taille[1] > 50) :
        pas = 16
    elif(taille[1] > 25) :
        pas = 8
    else:
        pas = 2
    return(pas)
    
    

#FUNCTIONS
def list_act_1_neur(layer, neur_x , neur_y , act_map):
    """Give list of all activities(100img) for 1 neuron for 1 layer for 1 act_map"""
    list_act = []
    layer_activ = layers_activations[layer] 
    for i in range(layer_activ.shape[0]):
        list_act.append(layer_activ[i, neur_x, neur_y, act_map])
    return(list_act)
    


def list_act_all_neur(layers_activations, layer, num_filter):
    """Give list of list all activities for all neuron for 1 layer for 1 act_map for 100 images
    directory = RESULTS_DIR +  'activ_dist/' + str(num_filter) + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)"""
    taille = layers_activations[layer].shape
    pas = find_pas(layers_activations, layer)
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
    
    
def lifetime_TrevesRolls(layer_act_all_neur, neuron):
    """Calculate Lifetime TrevesRolls
    1 M stimuli for 1 neurone in the layer
    layer_act_all_neur: list with activities reponse of all neurones for multiples images 
    neuron: int num of neuron of the list 
    return: """
    num = 0
    denom = 0
    M = len(layer_act_all_neur[neuron]) #nb stimuli(nb img)
    for i in range(M):
        r = layer_act_all_neur[neuron][i]
        num = r/M
        denom = (r*r)/M
    ST = (num**2)/denom
    return(ST)



def lifetime_multiple_neurons(layer_act_all_neur , criteria ):
    """List all KL of each neurons for a given layer"""
    if criteria == "Kurtosis" :
        all_KL = []
        for neuron in range(len(layer_act_all_neur)):
            all_KL.append(lifetime_kurtosis(layer_act_all_neur,neuron))
        return(all_KL)
    elif criteria == "TrevesRolls":
        all_TR = []
        for neuron in range(len(layer_act_all_neur)):
            all_TR.append(lifetime_TrevesRolls(layer_act_all_neur,neuron))
        return(all_TR)
    else:
        return(False)        




#----------- Population Sparseness Measures-----------------------------------


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
    KP = KP/N - 3
    return(KP)


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
    
    
def population_multiple_neurons(layer_act_all_neur , criteria ):
    """List all KL of each neurons for a given layer"""
    if criteria == "Kurtosis" :
        all_KP = []
        for img in range(len(layer_act_all_neur[0])):
            all_KP.append(population_kurtosis(layer_act_all_neur, img))
        return(all_KP)
    elif criteria == "TrevesRolls":
        all_TR = []
        for img in range(len(layer_act_all_neur[0])):
            all_TR.append(population_TrevesRolls(layer_act_all_neur,img))
        return(all_TR)
    else:
        return(False)   
        
#-----------------------------DICT ACTIVATIONS, sparseness per layer for each filter -------------------------------------------




#--------- TEMPS EXECUTION POUR 100 IMG 
    
#list pour chaq neuron du filtre de réponse de 100 images 
    
#on prend la derniere couche 100*14*14*512 -> 2mn
import time
debut = time.time()

curr_layer = 'conv5_3/Relu:0'
conv5_3_all_filtres=dict((el,{}) for el in range(layers_activations[curr_layer].shape[3]))
for num_filtre in range(layers_activations[curr_layer].shape[3]):
    layer_act_all_neur = list_act_all_neur(layers_activations, curr_layer, num_filtre)
    conv5_3_all_filtres[num_filtre]['activations'] = layer_act_all_neur
    conv5_3_all_filtres[num_filtre]['lifetimeTR'] = lifetime_multiple_neurons(layer_act_all_neur, criteria = "TrevesRolls")
    conv5_3_all_filtres[num_filtre]['populationTR'] = population_multiple_neurons(layer_act_all_neur, criteria = "TrevesRolls")
    print("--- %s seconds pour 1 filtre ---" % (time.time() - debut))
    

#Dict with sparseness(KP,KL)  for each layer 
debut = time.time()
sparseness_dict=dict((el,{}) for el in activ_list)
for curr_layer in sparseness_dict.keys():
    layer_dict = dict((el,{}) for el in range(layers_activations[curr_layer].shape[3]))
    for num_filtre in range(layers_activations[curr_layer].shape[3]):
        layer_act_all_neur = list_act_all_neur(layers_activations, curr_layer, num_filtre)
        layer_dict[num_filtre]['activations'] = layer_act_all_neur
        layer_dict[num_filtre]['lifetimeTR'] = lifetime_multiple_neurons(layer_act_all_neur, criteria = "TrevesRolls")
        layer_dict[num_filtre]['populationTR'] = population_multiple_neurons(layer_act_all_neur, criteria = "TrevesRolls")
    sparseness_dict[curr_layer] = layer_dict
    print("--- %s seconds pour 1 layer ---" % (time.time() - debut))


#save dict 
json = json.dumps(sparseness_dict)
f = open(RESULTS_DIR  + "dict_sparseness_perlayer.json","w")
f.write(json)
f.close()

truc=0
for k in sparseness_dict.keys():
    truc += len(sparseness_dict[k]))