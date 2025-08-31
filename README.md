# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: PRIYANKA S
RegisterNumber: 212224040255

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X for the intercept term (bias)
    X = np.c_[np.ones(len(X1)), X1]

    # Initialize theta (coefficients) with zeros
    theta = np.zeros(X.shape[1]).reshape(-1, 1)

    # Perform gradient descent
    for _ in range(num_iters):
        # Step 1: Calculate predictions
        predictions = X.dot(theta).reshape(-1, 1)

        # Step 2: Calculate errors
        errors = (predictions - y).reshape(-1, 1)

        # Step 3: Update theta using gradient descent formula
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)

    return theta
data = pd.read_csv('50_Startups.csv', header=None)
print (data.head())
# Assuming the last column is your target variable 'y' and the preceding columns are your features
X = (data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform (X1)
Y1_Scaled = scaler.fit_transform(y)
print('Name:PRIYANKA S')
print('Register No.:212224040255' )
print(X1_Scaled)
print(Y1_Scaled)
# Learn model parameters
theta = linear_regression (X1_Scaled, Y1_Scaled)
# Predict target value for a new data point
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
 
*/
```

## Output:

<img width="661" height="836" alt="Screenshot 2025-08-31 105736" src="https://github.com/user-attachments/assets/88fc8d29-1576-467d-bd87-d187f9fbdcc8" />

<img width="511" height="874" alt="Screenshot 2025-08-31 105812" src="https://github.com/user-attachments/assets/f2edd62e-3e18-4bfa-aab7-a71d5f25f7c0" />

<img width="529" height="880" alt="Screenshot 2025-08-31 105935" src="https://github.com/user-attachments/assets/74e18c42-3c33-450c-ac59-db25018ff2e4" />

<img width="582" height="875" alt="Screenshot 2025-08-31 105951" src="https://github.com/user-attachments/assets/35b2733e-1780-4db6-af88-5da59e13076b" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
