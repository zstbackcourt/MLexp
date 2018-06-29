import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import mglearn

X,y=mglearn.datasets.make_wave(n_samples=40)
plt.plot(X,y,'o')
plt.ylim(-3,3)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()