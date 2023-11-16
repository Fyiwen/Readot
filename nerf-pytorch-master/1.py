import torch
import numpy as np

#print(torch.cuda.is_available())
alpha=torch.rand(6,3) # 射线个数，每根采样点个数
print(alpha)
c=torch.ones((alpha.shape[0], 1))#(6,1)
print(c)
b=torch.cat([c, 1.-alpha + 1e-10], -1)(6,4)
print(b)
d=torch.cumprod(b, -1)(6,4)
print(d)
print(d[:, :-1]) #(6,3)
weights = alpha * d[:, :-1]
print(weights) #(6,3)