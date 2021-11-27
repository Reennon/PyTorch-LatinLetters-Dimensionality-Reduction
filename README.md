## PyTorch LatinLetters Dimensionality Reduction
Description
---
Dimensionality Reduction based on Latin Letters image classification

Model based on AlexNet, that classifies 26 classes of latin laters, based on original [CoMNIST](https://github.com/GregVial/CoMNIST) dataset.\
Transformed CoMNIST dataset is reduced by using UMAP method

Example of classification with ~91% Acuracy:

![зображення](https://user-images.githubusercontent.com/37474734/142851107-cbaac9d4-1570-48a7-9c78-e64857129e5b.png)


Data separation between training and validation datatasets are .85 to .15 with 15k of sample images stored as numpy arrays in csv file
Made by using [CoMNIST](https://github.com/GregVial/CoMNIST) Dataset with latin letters

> CoMNIST on GitHub https://github.com/GregVial/CoMNIST \
> CoMNIST on Kaggle, along with guide how to set it up
> https://www.kaggle.com/gregvial/comnist

UMAP 

Attached few samples of how reduced dataset by the UMAP looks

![зображення](https://user-images.githubusercontent.com/37474734/143688835-ef380d54-0dbc-4700-95cd-8455b8131ef2.png)

![зображення](https://user-images.githubusercontent.com/37474734/143688821-acaf79c0-d29e-43c7-ac28-f34ff3504010.png)

Fot this one I used hyperparameters below:
```
reducer = UMAP(
    n_neighbors=100,
    n_components=3,
    n_epochs=1000,
    min_dist=0.5,
    local_connectivity=10,
    random_state=42
)
```

![зображення](https://user-images.githubusercontent.com/37474734/143690501-74cc49d8-4cf9-4bce-8148-6db17f9d15b2.png)

And, for the one above I used these hyperparameters:
```
reducer = UMAP(
    n_neighbors=50,
    n_components=3,
    n_epochs=1000,
    min_dist=0.5,
    local_connectivity=5,
    random_state=42
)
```


> UMAP Implementation I used https://umap-learn.readthedocs.io/en/latest/index.html

## How to use pretrained model
You can use pretreined model parameters instead of training the model from scratch.
Here is the link to Google Drive with .pth file, containing alexnet's pretreined weights for this dataset.
https://drive.google.com/file/d/1LR42lsfMBW6CP6HvxS2Y0z3Oyi5_1ap7/view?usp=sharing

Steps to load pretreined PyTorch model:
1. download `AlexNet_<version>.pth` file from the Google Drive to the models folder (I attached a [link](https://drive.google.com/file/d/1LR42lsfMBW6CP6HvxS2Y0z3Oyi5_1ap7/view?usp=sharing) above)
2. create a PATH variable, with downloaded `.pth` file, an example is ./models/alexnet/AlexNet_1122_10_09_13.pth (if you are running the ipynb file on root level)
The cell should look like:
```
PATH = './models/alexnet/AlexNet_1122_10_09_13.pth'
```
3. create a cell below the newly created with PATH variable to initialize model, and enter the code below:
    
```
alexnet = AlexNet()
alexnet.load_state_dict(torch.load(PATH))
```

4. use alexnet.eval() to evaluate the model (run in inference) or alexnet.train() to train the model

Also below I attached a sample of how estimately should the model train for CoMNIST dataset, if you decided to train the model on your own

![зображення](https://user-images.githubusercontent.com/37474734/142851130-ef97eede-599a-435e-9420-a9e7ed0e7bf0.png)



## How to use pretrained UMAP reducer
Although, I highly encourage that you to try match hyperparameters by yourself, and see what you got\
To use pretrained UMAP:
1. download `UMAP_<version>` file from the Google Drive to the models folder ([link](https://drive.google.com/file/d/12816s5bCtf41LIpSeYv46uuMMQ3dItnw/view?usp=sharing)
2. create a UMAP_PICKLE_PATH variable, with downloaded `pickle` file, an example is ./models/umap/UMAP_11_26__00_24_51.pth (if you are running the ipynb file on root level)
3. create a cell below the newly created with PATH variable to initialize model, and enter the code below:
```
import pickle

UMAP_PICKLE_PATH = '<TODO: Enter full path to file here>'
with open(UMAP_PICKLE_PATH, 'rb') as pickle_file:
    reducer = pickle.load(pickle_file)
```
4. then you can just call:
```
# The shape should be (BATCH_SIZE, -1), NumPy array type
# I passed a flattened NumPy array, previously converted 
# from PyTorch tensor with shape (BATCH_SIZE, 64, 64)
array_a = np.array([1, 64 * 64])
reducer.transform(array_a)
```
Also you can look for example in reducer in `main.py` file

## The file structure should be as below:
![зображення](https://user-images.githubusercontent.com/37474734/143688108-7acb8789-53ad-4bef-b96e-1a8451fa488f.png)

---

### Pretrained reduced model
I just was curious if I can use NN to classify 26 letters of latin alphabet, having only 3 input features.

- Short answer: Not a good idea
- Long answer: It is a tough task, to teach model to generalize well

It is mainly caused by small input size (3), and huge number of classes (26) \
Model requires a lot of training, however there is a little time between each epoch.
Down below is my pretrained PyTorch model, that has 65% Accuracy on previously REDUCED set. But, if we would DIRECTLY PASS image, previously reduced by the UMAP, accuracy would drop significantly

PyTorch pretrained [model](https://drive.google.com/file/d/1q9pvlbPiP5MZ-5N2YGMIWCsGeYneqLiu/view?usp=sharing)

```
ReducedLatinNet(
  (fc1): Linear(in_features=3, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=256, bias=True)
  (fc3): Linear(in_features=256, out_features=256, bias=True)
  (fc4): Linear(in_features=256, out_features=256, bias=True)
  (fc5): Linear(in_features=256, out_features=26, bias=True)
)
```
