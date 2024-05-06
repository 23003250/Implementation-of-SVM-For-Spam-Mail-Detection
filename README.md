# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1.Start

step 2.Import the required packages.

step 3.Import the dataset to operate on.

step 4.Split the dataset.

step 5.Predict the required output.

step 6.Stop.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MIDHUN S
RegisterNumber:  212223240087
*/

import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### data.head():
![exp 9 1](https://github.com/23003250/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139331462/e40d1f85-d42b-44b2-8d83-b0be2d271f79)

### data.info():
![exp 9 2](https://github.com/23003250/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139331462/8589603c-b315-497a-bf65-595a390810b0)

### data.isnull()sum():
![exp 9 4](https://github.com/23003250/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139331462/9e77ce0f-fdcb-4d84-9754-c89b6a88a884)

### y_predict:
![exp 9 5](https://github.com/23003250/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139331462/196d717e-b11b-4357-b4bc-917e2f3c89d6)

### Accuracy:
![exp 9 accuracy](https://github.com/23003250/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139331462/822bb778-68e6-49ea-a4c8-c9ced24ae6ff)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
