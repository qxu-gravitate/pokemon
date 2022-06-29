# Pokemon

``` 
practice/
├── data
    ├── test
    test_labels.csv
    ├── train
    train_labels.csv
├── logs
├── output
    detector_multiclass.h5
    lb.pickle
    ├── plot
        accs.png
        losses.png
    test_images_multiclass.txt
train.py
predict_model.py
requirements.txt
example
readme.md
``` 

## Step 1. PIP install packages
```
cd practice
pip install -r requirements.txt
```

## Step 2. Model training
```
python train.py
```

## Step 3. Model Predict
For example
```
python predict_model.py --input /practice/data/train/pokemon-2011.jpg
```

## Step 4. Tensorflow Board
```
tensorboard --logdir logs
```
Check from http://localhost:6006/

# Image Data: https://drive.google.com/drive/folders/1Tk9gP4gxiQkZ7_H27nIXtNl9RbF__F30?usp=sharing
