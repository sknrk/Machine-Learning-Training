from operator import index
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("HyderabadResturants.csv",sep=",")
df.drop(0 if each=='New' else 1 for each in df.ratings)
i=0
for each in df.ratings:
    i= i +1
    if(each=='New' or each=='-'):
        df.drop(index=i,axis=0,inplace = True)

print(df.ratings)