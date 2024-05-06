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
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
```
```

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
![Screenshot 2024-04-29 133208](https://github.com/Aadithya2201/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145917810/baf4bb87-7e16-445d-92bf-2f5a10792c10)
### data.info():
![Screenshot 2024-04-29 133228](https://github.com/Aadithya2201/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145917810/d3d1fcf1-4a59-4eed-b9c1-ef6f624af227)
### data.isnull()sum():
![Screenshot 2024-04-29 133239](https://github.com/Aadithya2201/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145917810/88894413-b530-4622-86c8-544cdbfca4f1)

### y_predict:
![Screenshot 2024-04-29 133246](https://github.com/Aadithya2201/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145917810/5cd851f3-2720-49d6-928c-3d598a68ab66)
### Accuracy:
![Screenshot 2024-04-29 133252](https://github.com/Aadithya2201/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145917810/6da1045f-f1d4-4dca-94b0-881423607e9e)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
