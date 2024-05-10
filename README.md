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
from sklearn.model_selection import train_test_split
data
data.shape
x=data['v2'].values
y=data['v1'].values
x.shape
y.shape
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)
x_train
```
```

x_train.shape
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report 
accuracy=accuracy_score(y_test,y_pred)
accuracy
con=confusion_matrix(y_test,y_pred)
print(con)
cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
### data:
![exp 9 data](https://github.com/23003250/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139331462/dff3e622-e7b8-4a15-9c18-e39caf1d06fa)

### data.shape:
![exp 9 data shape](https://github.com/23003250/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139331462/46019e0a-abd4-45ac-9793-ad85d5fc4f81)

### x.shape:
![exp 9 x shape](https://github.com/23003250/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139331462/7bb6f222-2a1b-40ff-a854-5b1f015e1be0)

### y.shape:
![exp 9 x shape](https://github.com/23003250/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139331462/cdbf779c-5abe-426a-9a43-06fabd253930)

### x_train:
![exp 9 x_train](https://github.com/23003250/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139331462/c5d76edd-4d65-4b34-94c8-bd14e53e21e3)

### x_train.shape:
![exp 9 x_train shape](https://github.com/23003250/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139331462/0a132e17-a840-4ae2-955e-b866efcfd4ae)

### y_pred:
![exp 9 y_pred](https://github.com/23003250/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139331462/f3edd0ee-06fb-4594-b99e-06a761fe9956)

### Accuracy:
![exp 9 accuracy](https://github.com/23003250/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139331462/55accff0-f33c-4575-9b70-ab39e7ca51d4)

### confusion_matrix:
![exp 9 confusion_matrix](https://github.com/23003250/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139331462/0196f78d-660e-4592-9b61-165c43dd281d)

### classification_report:
![exp 9 classification report](https://github.com/23003250/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139331462/02df2736-a171-4c0d-8a81-4fd5ce0f2c73)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
