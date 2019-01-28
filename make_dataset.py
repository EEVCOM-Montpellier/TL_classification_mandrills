import pandas as pd
import csv
import re
import os


#check curr dir
os.getcwd()

#----------- VARIABLES---------------------------------------------
# Folder with all pictures
dirname = "C:/Users/renoult/Documents/BDD_PHOTOS_MANDRILLUS_FACES/MANDRILLS_BKB"

#Doc excel with pictures annotations
tags_pict_path = "C:/Users/renoult/Documents/BDD_PHOTOS_MANDRILLUS_FACES/Tags_240119.csv"


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




list_julien = list_of_pict("C:/Users/renoult/Documents/Mandrill Picture Analyses/learn_all_mini12")
list_pict_BKB = list_of_pict(dirname)
dict_ind_BKB = dict_of_ind(dirname)


#----------- ANNOTATIONS---------------------------------------------

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


tags_pict  = clean_pict(2)

#Check if images are in dataset
tags_pict = tags_pict[tags_pict.pict.isin(list_pict_BKB)]

#Save to csv
tags_pict.to_csv('datas/clean_pict_tags.csv')


mini_jdd = tags_pict.sample(frac=0.1, replace=False, random_state=1)
