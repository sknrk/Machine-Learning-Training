import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import rand

df = pd.read_csv("random_forest_regression_dataset.csv", sep=";")
x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor

random_forest_regressor = RandomForestRegressor(n_estimators=100,random_state=42)

random_forest_regressor.fit(x,y)
# aranged_x = np.arange(min(x),max(x),0.01).reshape(-1,1)
predicted_values = random_forest_regressor.predict(x)


from sklearn.metrics import r2_score

print("r_score: ",r2_score(y,predicted_values))