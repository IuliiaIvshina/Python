from warnings import simplefilter

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

data_TC = pd.ExcelFile("TehresursPrice.xlsx")
data_TC_1 = data_TC.parse("Лист1")

DD = np.hstack((data_TC_1.values[:, 0:11]))
# print(DD)
D = np.hstack((data_TC_1.values[:, 1:2],
               data_TC_1.values[:, 4:7],
               data_TC_1.values[:, 8:11])).astype(np.double)
print("D[0] =", D[0])

Y0 = data_TC_1.values[:, 8:9].astype(np.int32).flatten()
print("Y0[0] =", Y0[0])
HP = sorted(set(Y0))
print(HP)

for i in range(len(D[0])):
    a, b = np.polyfit(np.sort(D[:, i]), np.linspace(0, 1, len(D[:, i])), 1)
    D[:, i] = D[:, i] * a + b
# print(D)

Y = Y0
# Y = np.zeros((len(Y0)))
# for i, y0 in zip(range(len(Y0)), Y0):
#     Y[i] = y0
# print(Y)

simplefilter("ignore", category=ConvergenceWarning)
hl_size = 1
while hl_size <= 1024:
    hl_size *= 2
    print("hidden_layer_size =", hl_size)

    clf = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(hl_size), random_state=1, max_iter=1000, warm_start=True)

    for x in range(100):
        clf.fit(D, Y)

    # Y_ = clf.predict_proba(D)
    Y_ = clf.predict(D)

    # for a, b in zip(Y, Y_):
    #     print("Y :", [ round(x) for x in a ], "> Y':", [ x for x in b ])

    # for a in Y_:
    #     print( "> Y':", [ x for x in a ])

    difference = Y_ - Y
    delta = np.sum(abs(difference))
    print("delta =", delta)
    if delta == 0:
        break

print(Y_)
print(difference)
