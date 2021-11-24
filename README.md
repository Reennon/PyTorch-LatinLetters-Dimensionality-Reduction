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

### How to use pretrained model
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

