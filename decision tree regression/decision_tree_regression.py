import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt

df = pd.read_csv("decision_tree_regression_dataset.csv",sep=";")

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(x,y)
aranged_x = np.arange(min(x),max(x),0.01).reshape(-1,1)
predicted_values = tree_reg.predict(aranged_x)

plt.scatter(x,y,color="green")
plt.plot(aranged_x,predicted_values,color="red")
plt.show()