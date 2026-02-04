import torch
import pandas as pd 

# X = torch.tensor([
#     [2.0],
#     [7.0]
# ])

# Y = torch.tensor([
#     [13],
#     [24]
# ])

# #w = Weight = slope
# w = torch.tensor([
#     [3.0]
# ])

# #b = bias = constant
# b = torch.tensor([
#     [5.0]
# ])

# #Multiply X by weight + bias
# Yhat = X@w + b
# r = Yhat - Y
# SSE = r.T@r
# loss = SSE/2
# print(Yhat)
# print(SSE)
# print(loss)


# Function with 2 variables

# X = torch.tensor([
#     [2.0, 3.0]
# ])

# Y = torch.tensor([
#     [30]
# ])

# #w = Weight = slope
# w = torch.tensor([
#     [4.0],
#     [1.0]
# ])

# #b = bias = constant
# b = torch.tensor([
#     [5.0]
# ])

# #Multiply X by weight + bias
# Yhat = X@w + b
# r = Yhat - Y
# SSE = r.T@r
# loss = SSE/2
# print(Yhat)
# print(SSE)
# print(loss)

# 

# X = torch.tensor([
#     [60.0, 11.0],
#     [24.0, 12.5]
# ])

# Y = torch.tensor([
#     [10.0],
#     [5.0]
# ])

# #w = Weight = slope
# w = torch.tensor([
#     [4.0],
#     [1.0]
# ])

# #b = bias = constant
# b = torch.tensor([
#     [5.0]
# ])

# #Multiply X by weight + bias
# Yhat = X@w + b
# r = Yhat - Y
# SSE = r.T@r
# loss = SSE/2
# print(Yhat)
# print(SSE)
# print(loss)

# Exercise

df = pd.read_csv('data.csv')
X = torch.tensor(df.drop('Y', axis = 1).to_numpy()).float()
Y = torch.tensor(df['Y'].to_numpy()).float().reshape(-1,1)

w = torch.tensor([
    [-2.6],
    [-1.9],
    [-2.1]
]).float()

b = torch.tensor([
    [1.6]
])

Yhat = X@w + b 
r = Yhat-Y
SSE = r.T@r
loss = SSE/10

print(loss)