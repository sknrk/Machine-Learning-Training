import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("data.csv",sep=",")
data.drop(['Unnamed: 32','id'],axis=1,inplace=True)
data.diagnosis = [1 if each=='M' else 0 for each in data.diagnosis]

y = data.diagnosis.values
x_data = data.drop(['diagnosis'],axis=1)


x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)

x_train = x_train.T 
x_test = x_test.T 
y_train = y_train.T 
y_test = y_test.T 

def initialize_weights_and_bias(dimension):
    weight = np.full((dimension,1),0.01)
    bias = 0.0

    return weight,bias


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train.T,y_train.T)
print("Test Accuracy {}".format(lr.score(x_test.T,y_test.T)))