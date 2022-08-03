import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

data = pd.read_csv("data.csv")
data = data.drop(['Unnamed: 32'],axis=1)

M=data[data.diagnosis == 'M']
B=data[data.diagnosis == 'B']

# plt.scatter(M.radius_mean,M.texture_mean,color="red",label="bad")
# plt.scatter(B.radius_mean,B.texture_mean,color="green",label="good")
# plt.xlabel("radius mean")
# plt.ylabel("texture mean")
# plt.legend()
# plt.show()

data.diagnosis = [1 if each == 'M' else 0 for each in data.diagnosis]

x_data = data.drop(['diagnosis'],axis=1)
y = data.diagnosis.values

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn.ensemble import RandomForestClassifier

RandomForest = RandomForestClassifier(n_estimators= 100, random_state=1)

RandomForest.fit(x_train,y_train)
print(RandomForest.score(x_test,y_test))