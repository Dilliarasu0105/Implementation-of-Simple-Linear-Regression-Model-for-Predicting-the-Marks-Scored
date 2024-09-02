# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
'''
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.
'''

## Program:
```
/*
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DILLIARASU M
RegisterNumber:  212223230049
*/
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("student_scores.csv")

print(df.tail())
print(df.head())
df.info()

x = df.iloc[:, :-1].values  # Hours
y = df.iloc[:,:-1].values   # Scores

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

print("X_Training:", x_train)
print("X_Test:", x_test)
print("Y_Training:", y_train)
print("Y_Test:", y_test)

reg = LinearRegression()
reg.fit(x_train, y_train)

Y_pred = reg.predict(x_test)

print("Predicted Scores:", Y_pred)
print("Actual Scores:", y_test)

a = Y_pred - y_test
print("Difference (Predicted - Actual):", a)

plt.scatter(x_train, y_train, color="green")
plt.plot(x_train, reg.predict(x_train), color="red")
plt.title('Training set (Hours vs Scores)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test, y_test, color="blue")
plt.plot(x_test, reg.predict(x_test), color="green")
plt.title('Testing set (Hours vs Scores)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mae = mean_absolute_error(y_test, Y_pred)
mse = mean_squared_error(y_test, Y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)  
*/
```

## Output:

![359573548-9de86369-938b-4030-a048-808337d517d4](https://github.com/user-attachments/assets/29f8b60d-8198-4913-9508-2d33b8703477)

![359573769-6a1861ee-dac9-408d-9f7a-faef16902dbd](https://github.com/user-attachments/assets/9c80a676-2bf2-44bf-8627-24487d321d94)
## TRAINING SET INPUT
![359570077-5b02582c-73c3-40e7-bca7-b53d44566ded](https://github.com/user-attachments/assets/7f74590e-6856-4342-9507-60d72b653e01)

![359570509-3b8a3dc9-a4c3-4c62-879e-fe9f1420eba7](https://github.com/user-attachments/assets/98154f0e-3e03-4764-8ddd-d6cd1a115b31)
## TEST SET VALUE
![359571117-5b112037-841a-48c7-a45f-8529261351df](https://github.com/user-attachments/assets/6becf818-a60d-45a5-9516-a13ab4b4c1e1)

![359571413-e5de4423-c71a-434b-a8b3-32d5c317d78d](https://github.com/user-attachments/assets/12b44b95-3e93-4db0-ade6-904062617f59)
## TRAINING SET

![359572132-c550b4a3-7bc4-4365-ad61-03aa1fd3e53a](https://github.com/user-attachments/assets/559747a1-f81f-4684-bfd0-6da3bc8b3c41)
## TEST SET:

![359572417-68d8ecf9-f140-4677-a7f9-9b6609da5987](https://github.com/user-attachments/assets/deada089-1acd-4f95-87d3-24ddc1d1b380)
## MEAN SQUARE ERROR, MEAN ABSOLUTE ERROR AND RMSE:

![image](https://github.com/user-attachments/assets/2d1119c9-0376-4020-bd09-303f32564860)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
