import torch

x = torch.tensor(7) #Scaler
x = torch.tensor([4,3,7,6,5])   #Vector
x = torch.tensor([
    [3,4],
    [1,2]
])  #Matrix
x = torch.tensor([
    [4,3,7,6,5]
])  #Row Vector

x = torch.tensor([
    [4],
    [3],
    [7],
    [6],
    [5]
])  #Column Vector

A = torch.tensor([
    [1,5],
    [7,4]
])  #Matrix are represented by CAPITAL LETTER

print(A.T)

# Exercise 1

# Q.1 - Shape:5,5      Dimension:2
# Q.2 - Shape:3,      Dimension:1
# Q.3 - Shape:1,      Dimension:1 
# Q.4 - Shape:      Dimension:0 
# Q.5 - Shape:3,1      Dimension:1 
# Q.6 - Shape:5,1      Dimension:1 
# Q.7 - Shape:4,2      Dimension:2 
# Q.8 - Shape:5,      Dimension:1 
# Q.9 - Shape:1,1      Dimension:2 
# Q.10 - Shape:3,2      Dimension:2 