import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('multiple_linear_regression_dataset.csv',sep=";")

x = df.iloc[:,[0,2]].values
y = df.maas.values.reshape(-1,1)

multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

print('b0:')
print(multiple_linear_regression.intercept_)

print('b1,b2:')
print(multiple_linear_regression.coef_)

predicted_values = multiple_linear_regression.predict(np.array([[5,35],[10,35]]))

print(predicted_values)