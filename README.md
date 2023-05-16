# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required packages.
 
 2.Display the values predicted using scatter plot and predict.

 3.Plot the graph according to the given input.

 4.End the program

## Program:
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: Bairav Skandan Loha

RegisterNumber: 212221230010
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

y_test

plt.scatter(x_train,y_train,color="black")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show

plt.scatter(x_test,y_test,color="yellow")
plt.plot(x_test,regressor.predict(x_test),color="green")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse) 

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)


```

## Output:
df.head()

![](https://raw.githubusercontent.com/KATHIR1611/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/main/ML%20exp%2021.png)

df.tail()

![](https://raw.githubusercontent.com/KATHIR1611/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/main/ML%20exp%2022.png)

Array values of x

![](https://github.com/KATHIR1611/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/ML%20exp%2023.png)

Array values of y

![](https://github.com/KATHIR1611/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/ML%20exp%2024.png)

Values of y prediction

![](https://github.com/KATHIR1611/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/ML%20exp%20210.png)

Array values of y test

![](https://github.com/KATHIR1611/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/ML%20exp%2026.png)

Training set graph

![](https://github.com/KATHIR1611/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/ML%20exp%2027.png)

Test set graph

![](https://github.com/KATHIR1611/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/ML%20exc%2028.png)

Values of MSE,MAE and RMSE

![](https://github.com/KATHIR1611/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/ML%20exp%2029.png)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
