# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
date: 31/8/23
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for Gradient Design.
2.Set variables for assigning dataset values. 
3.Import linear regression from sklearn 4.Assign the points for representing the graph
4.Predict the regression for marks by using the representation of the graph.

## Program:

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Bala Umesh  
RegisterNumber:  212221040024
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()
df.tail()
X = df.iloc[:,:-1].values
X
Y = df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="blue")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="yellow")
plt.plot(X_test,regressor.predict(X_test),color="black")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE= ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE= ",mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```


## Output:
### df.head():
![image](https://github.com/BalaUmesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113031742/64eed93d-39d9-467a-9b64-199fb4fcef1b)


### df.tail():
![image](https://github.com/BalaUmesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113031742/1902b7e2-d0fb-47bf-a763-0e5657097e56)


### array value of x:
![image](https://github.com/BalaUmesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113031742/5a497aa9-df61-4d43-aa68-e593818d7b93)


### array value of y:
![image](https://github.com/BalaUmesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113031742/b2cd49f1-0854-4dd1-bb60-16290573bb73)


### values of y prediction:
![image](https://github.com/BalaUmesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113031742/e32df82a-1ba7-4abc-b0c8-5a9391eeed0f)


### array values of Y test:
![image](https://github.com/BalaUmesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113031742/aa6e7de1-fcf1-48e3-9b16-3f3da7cc4315)


### training set graph:
![image](https://github.com/BalaUmesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113031742/0f8d3690-5561-4c94-9298-204ec570850b)


### test set graph:
![image](https://github.com/BalaUmesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113031742/7a8f57dd-9a0f-4709-8154-1f945b755a92)


### values of mse, mae, rmse:
![image](https://github.com/BalaUmesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113031742/f1936065-d0ec-47df-ac13-e65c8e8b34f2)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
