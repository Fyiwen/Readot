import torch
import numpy as np

print(torch.cuda.is_available())
'''
c2w=np.array([
    [
        -0.9,
        0.1,
        -0.01,
        -0.05
    ],
    [
        -0.01,
        -0.29,
        0.95,
        3.84
    ],
    [
        -4.6,
        0.9,
        0.2,
        1.2
    ],
    [
        0.0,
        0.0,
        0.0,
        1.0
    ],
])
focal=3
s=np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.]))
print(s)
'''