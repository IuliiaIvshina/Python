
from sklearn.neural_network import MLPClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_TC = pd.ExcelFile('TehresursPrice.xlsx')
data_TC_1 = data_TC.parse('Лист1')

DD = np.hstack((data_TC_1.values[:,0:11]))
# print(DD)
D = np.hstack((data_TC_1.values[:,1:2], 
               data_TC_1.values[:,4:7],
               data_TC_1.values[:,8:11])).astype(np.double)
print(D)
Y0 = data_TC_1.values[:,4:5].astype(np.int32).flatten()
print(Y0)
# Y0 = np.hstack((data_TC_1.values[:,4:5])).astype(np.float32)
HP = sorted(set(Y0))
print(HP)

# линейная апроксимация цены
x = np.arange(len(D[:,1]))
y = D[:,1]
print(x)
print(y)
a,b = np.polyfit(x,y,1)
model = a*x + b
print(model)


print(np.min(y),np.max(y))

for i in 4:#range(len(D[0])):
    a, b = np.polyfit(np.sort(D[:,i]),np.linspace(np.min(y),np.max(y),len(D[:,i])),1)
    D[:,i] = D[:,i] * a + b
    # print(D[:,i] * a + b)

print(D)
plt.plot(x, y,'b',x, model,'g',x, D[:,1],'r^')
plt.show()




# шкалирование
for i in range(len(D[0])):
    a, b = np.polyfit(np.sort(D[:,i]),np.linspace(0,1,len(D[:,i])),1)
    D[:,i] = D[:,i] * a + b





Y = np.zeros((len(Y0), len(HP)))
for i, y0 in zip(range(len(Y0)), Y0):
    Y[i,HP.index(y0)] = 1.0
print(Y)

clf = MLPClassifier(
    solver='lbfgs', 
    alpha=1e-5,
    hidden_layer_sizes=(200), 
    random_state=1,
    max_iter=100, 
    warm_start=True
)

for x in range(100):
    clf.fit(D, Y)

Y_ = clf.predict_proba(D)

for a, b in zip(Y, Y_):
    print("Y :", [ round(x) for x in a ], "> Y':", [ round(x) for x in b ])