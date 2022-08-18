import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris = load_iris()

data = iris.data
feature_names = iris.feature_names
y = iris.target

df = pd.DataFrame(data,columns=feature_names)
df['sinir']=y

x = data

from sklearn.decomposition import PCA 

pca = PCA(n_components=2, whiten=True)
pca.fit(x)
x_pca = pca.transform(x)
print('variance_ratio:',pca.explained_variance_ratio_)
print('sum:',sum(pca.explained_variance_ratio_))