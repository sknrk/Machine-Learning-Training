import pandas as pd
import matplotlib.pyplot as plt
# linear_regression_datasetcsv = open('linear_regression_dataset.csv','r')
df = pd.read_csv("linear_regression_dataset.csv",sep = ';')
plt.scatter(df.deneyim,df.maas)
plt.xlabel('deneyim')
plt.ylabel('maas')
plt.show()

from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

x = df.deneyim.values.reshape(1,-1)
y = df.maas.values.reshape(1,-1)
# print(x)

linear_reg.fit(x,y)
linear_reg.predict(0)