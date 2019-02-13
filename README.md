# Transfert Learning: Classification of Mandrills

The objective of this tool is to classify Mandrills Face per : sex, stage and individus.


## Requirements

* Install directly all requirements here (in your env ; ex py35):

```
pip install -r requirements.txt
```


* If using a virtual env (with Anaconda)

```
conda create -n py35 python=3.5 anaconda
activate py35
```

 * Python library

```
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
```

* Model from rc_malli repo

```
pip install keras_vggface
 ```

## Run

Please launch the tool at base.

Run the program
```
 python main-args.py -model MODEL -classification CLASS
 ```

With :

- MODEL: the architecture of neural network with options: resnet_rcmalli(default) , vgg16 , vgg19, resnet
Only resnet_rcmalli is pre-trained with VGGface, the others are train with Imagenet.

- CLASS : Type of classification: sex, stage or individue.
- CLASS (NEW UPDATE) : add : feminity and sparseness
For feminity and sparseness: it takes a dataset without the pictures of the 65 females (with comportmental varaibles) to learn

## Example usage


```
python main-args.py -model resnet_rcmalli -classification stage
```

 The tool will create a folder inside `results/` a folder with the current date `results/2019-01-28/`.
 Inside `results/2019-01-28/` : log_file.txt , params.txt, plots of accuracy and loss.

 ## Folders and scripts

* **main-args.py** : /!\ Change photos directory : ```dirname = "C:/Users/renoult/Documents/BDD_PHOTOS_MANDRILLUS_FACES/MANDRILLS_BKB"```
* **make_dataset.py** : formate meta-datas for images (remove bad quality images)
