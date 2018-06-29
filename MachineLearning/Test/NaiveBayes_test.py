import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

X=np.array(
    [[0,1,0,1],
     [1,0,1,1],
     [0,0,0,1],
     [1,0,1,0]]
)

y=np.array([0,1,0,1])

for label in np.unique(y):
    print(label)