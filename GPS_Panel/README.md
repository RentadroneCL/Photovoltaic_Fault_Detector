
# Panel Enumeration
This algorithm divides the tables and lists the panels of each of these tables.

## Install requirements

In the root project execute the following command to install all dependencies project

```
pip install -r requirements.txt
```

## Example

View example 
```
Enumeration_KML.ipynb
```

# Panel Classifier
This algorithm uses the division of the panels to classify them individually if they are with any fault.

## Train Model 
In the Classifier folder there is the training of the model 
[Train_Classifier_DEEP_SVC](Classifier/Train_Classifier_DEEP_SVC.ipynb)
which is based on the union of a deep learning model pre-trained with Imagenet together with an SVM model.

## Example

View example

```
ClassifierPanel_KML.ipynb

```

## Classification Report


|          | Precision | Recall | F1-score | support | accuracy |
|----------|-----------|--------|----------|---------|----------|
| 0-Normal | 0.95      | 0.97   | 0.96     | 1688    |          |
| 1-Falla  | 0.97      | 0.96   | 0.96     | 2084    | 0.96     |




