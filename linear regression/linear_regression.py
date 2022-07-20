import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import line
# linear_regression_datasetcsv = open('linear_regression_dataset.csv','r')
df = pd.read_csv("linear_regression_dataset.csv",sep = ';')
plt.scatter(df.deneyim,df.maas)
# print(df.maas)
plt.xlabel('deneyim')
plt.ylabel('maas')
# plt.show()

from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)
# print(x)

linear_reg.fit(x,y)


b0 = linear_reg.predict([[0]]) 
print("b0:")
print(b0)

b0_ = linear_reg.intercept_
print("b0_:")
print(b0_)

b1 = linear_reg.coef_

array_values = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)
predicted_array = linear_reg.predict(array_values)
plt.plot(array_values,predicted_array,color="red")
plt.show()