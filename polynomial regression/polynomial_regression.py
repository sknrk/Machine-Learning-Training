from distutils.errors import LinkError
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('polynomial_regression.csv',sep=';')

x = df.araba_fiyat.values.reshape(-1,1)
y = df.araba_max_hiz.values.reshape(-1,1)

plt.scatter(x,y)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = PolynomialFeatures(degree = 2)
x_polynomial = polynomial_regression.fit_transform(x)


# polynomial_regression = LinearRegression()
# polynomial_regression.predict(x)

linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y)
y_answer = linear_regression2.predict(x_polynomial)
plt.plot(x,y_answer,color = 'black')




plt.plot(x,lr.predict(x),color="red")
plt.show()
