from scipy.io import loadmat
import os
import numpy as np
import pandas as pd
current_dir = os.path.dirname(os.path.realpath(__file__))
data = loadmat(current_dir + '/Q/OP-KNN-PG/k10/' + 'AG.mat')

mean = data["Q_mean"]
min = data["min"]
min = pd.DataFrame(min)
min = min.stack()
#min = np.asarray(min)
#min = min.tolist()
print('min:', min)
max = data["max"]
max = pd.DataFrame(max)
max = max.stack()

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot')
x= np.arange(len(mean)) + 1
plt.plot(mean, label='DPGOA')
plt.fill_between(x,min, max, alpha=0.2)
plt.ylabel('Normalized data transmission rate',fontsize=14)  # Normalized Computation Rate
plt.xlabel('Time slot t',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc=7,fontsize=14)
plt.savefig("Q/OP-KNN-PG/k10/temp.png", dpi=500)
plt.show()
