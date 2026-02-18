import torch
import pandas as pd 
import numpy as np 

df = pd.read_csv('data.csv')
X = torch.tensor(df.drop("Y", axis = 1).to_numpy()).float()
Y = torch.tensor(df["Y"].to_numpy()).reshape(-1,1)

w = torch.tensor([
    [],
    [],
    [],
    []
])

b = torch.tensor([
    []
])

Yhat = X@w + b
r = Yhat-Y
SSE = r.T@r
loss = SSE/5