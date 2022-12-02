import numpy as np
import pandas as pd

a = 0.1 # темп  обучения
b = -0.4 # торможение (смещение)

# активационная функция
sigm = lambda x: 1 if x > 0 else 0

data_TC = pd.ExcelFile('data.xlsx')
#print(data_TC.sheet_names)

# data_TC_1 = data_TC.parse('Лист1')
# #print(data_TC_1)

# # исходные данные
# D0 = np.hstack((data_TC_1.values[:,2:5],data_TC_1.values[:,6:9]))
# for i in [0,1,2,3,4,5]:
#     a, b = np.polyfit(
#         np.sort(D0[:,i]),
#         np.linspace(0,1,len(D0[:,i])),1
#     )
#     D0[:,i] = D0[:,i] * a + b


# обучающие данные
data_TC_1 = data_TC.parse('Лист4')
D = np.hstack((data_TC_1.values[:,2:5],data_TC_1.values[:,6:9]))
print(D)
#print(range(len(D[0])))
for i in [0,1,2,3,4,5]:
    a, b = np.polyfit(
        np.sort(D[:,i]),
        np.linspace(0,1,len(D[:,i])),1
    )
    D[:,i] = D[:,i] * a + b


Y = np.hstack((data_TC_1.values[:,4:5])).astype(np.float32)
#...:astype
print(D[1])

#print(type(D))
#print(type(Y))


print(set(Y))
# инициализация весов
w0 = np.zeros(len(D[0]))
w1 = np.zeros_like(w0)
w2 = np.zeros_like(w0)
w3 = np.zeros_like(w0)
#print(w1)

# выходной вектор
Y0 = np.zeros_like(Y)
Y0[np.where(Y==1.0)] = 1
Y1 = np.zeros_like(Y)
Y1[np.where(Y==2.0)] = 1
Y2 = np.zeros_like(Y)
Y2[np.where(Y==3.0)] = 1
Y3 = np.zeros_like(Y)
Y3[np.where(Y==4.0)] = 1
print(Y0)
print(Y1)
print(Y2)
print(Y3)

# тело нейрона
def f(x,w):
    s = b + np.sum(x @ w)
    return sigm(s)

# эпоха обучения
def train(w,Y):
    _w = w.copy()
    for x, y in zip(D, Y):
        w += a * (y - f(x,w)) * x
    return (w != _w).any()  


# обучение и тестирование
# while train(w0,Y0) and train(w1,Y1) and train(w2,Y2) and train(w3,Y3):
#     print(w0) 
#     print(w1) 
#     print(w2) 
#     print(w3) 

while train(w0,Y0) :
    print(w0) 
    

while  train(w1,Y1) :
    print(w1) 
    

while train(w2,Y2) :
    print(w2) 
    

while train(w3,Y3):
    print(w3) 

for x in D:
    print(x, f(x,w0), f(x,w1), f(x,w2),f(x,w3))


