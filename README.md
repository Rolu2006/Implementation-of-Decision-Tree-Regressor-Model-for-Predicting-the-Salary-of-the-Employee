# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import pandas
2.Import Decision tree classifier
3.Fit the data in the model
4.Find the accuracy score
```

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:somalarajurohini 
RegisterNumber:24000337  
*/
```
```
import pandas as pd
data=pd.read_csv("/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```








## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10 entries, 0 to 9
Data columns (total 3 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   Position  10 non-null     object
 1   Level     10 non-null     int64 
 2   Salary    10 non-null     int64 
dtypes: int64(2), object(1)
memory usage: 372.0+ bytes
Position    0
Level       0
Salary      0
dtype: int64

![Screenshot 2024-11-29 081549](https://github.com/user-attachments/assets/9a7855f7-7e6e-4235-991b-b9c13cf2c651)

0     45000
1     50000
2     60000
3     80000
4    110000
Name: Salary, dtype: int64
array([80000., 45000.])
0.48611111111111116
array([150000.])
```









## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
