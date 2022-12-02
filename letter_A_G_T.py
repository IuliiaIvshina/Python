
# распознавание букв
import numpy as np
import math 

# исходные данные
# распознаем русские буквы а, г, т
D = np.array([
    [1,1,1,
     1,1,1,
     1,0,1],
    [1,1,1,
     1,0,0,
     1,0,0],
    [1,1,1,
     0,1,0,
     0,1,0],
]) 
# выходной вектор
Ya = np.array([1,0,0])
Yg = np.array([0,1,0])
Yt = np.array([0,0,1])

# инициализация весов
wa = np.zeros((9)) 
wg = np.zeros((9)) 
wt = np.zeros((9)) 



a = 0.2 # темп  обучения
b = -0.4 # торможение (смещение)

# активационная функция
sigm = lambda x: 1 if x > 0 else 0

# тело нейрона
def f(x,w):
    s = b + np.sum(x @ w)
    return sigm(s)

# # тело нейрона
# def f1(x):
#     s = b + np.sum(x @ w1)
#     return sigm(s)

# эпоха обучения
def train(D,Y,w):
    _w = w.copy()
    for x, y in zip(D, Y):
        w += a * (y - f(x,w)) * x
    return (w != _w).any()  


# обучение и тестирование
print('обучние буква а')
while train(D,Ya,wa): #and train(D,Yc,wc) and train(D,Yt,wt):
    print(wa) 
    
# обучение и тестирование
print('обучние буква г')
while train(D,Yg,wg): #and train(D,Yc,wc) and train(D,Yt,wt):
    print(wg)
    
# обучение и тестирование
print('обучние буква T')
while train(D,Yt,wt): #and train(D,Yc,wc) and train(D,Yt,wt):
    print(wt)


for x in D:
    print(x, f(x,wa), f(x,wg),f(x,wt))

# искажаем представление русских букв а, г, т
DD = np.array([
    [0,1,0,
     1,1,1,
     1,0,1],
    [1,1,1,
     1,0,1,
     1,0,0],
    [1,1,1,
     0,1,0,
     0,1,1],
]) 
print('искажаенные буквы (с шумом) а, г, т')
for x in DD:
    print(x, f(x,wa), f(x,wg),f(x,wt))


# -------------------------------------------------
Me = np.eye(3)
print(Me)  

# for x in D:
#     D_turn = x[0]
#     print(x, f(x,wa), f(x,wg),f(x,wt))

#  = 

MDDa = np.array([DD[0][0:3],DD[0][3:6],DD[0][6:9]])
MDDg = np.array([DD[1][0:3],DD[1][3:6],DD[1][6:9]])
MDDt = np.array([DD[2][0:3],DD[2][3:6],DD[2][6:9]])


# до 36 градусов можно повернуть изображение
fi = 37* math.pi/180
M_fi = np.array([[math.cos(fi), math.sin(fi), 0],[-math.sin(fi), math.cos(fi), 0],[0, 0, 1]])
MDDa_turn = (np.dot(M_fi,MDDa))
MDDg_turn = (np.dot(M_fi,MDDg))
MDDt_turn = (np.dot(M_fi,MDDt))
# print()

DDturn = (np.array([[MDDa_turn[0][0], MDDa_turn[0][1], MDDa_turn[0][2],
                MDDa_turn[1][0], MDDa_turn[1][1], MDDa_turn[1][2],
                MDDa_turn[2][0], MDDa_turn[2][1], MDDa_turn[2][2]],
               [MDDg_turn[0][0], MDDg_turn[0][1], MDDg_turn[0][2],
                MDDg_turn[1][0], MDDg_turn[1][1], MDDg_turn[1][2],
                MDDg_turn[2][0], MDDg_turn[2][1], MDDg_turn[2][2]],
               [MDDt_turn[0][0], MDDt_turn[0][1], MDDt_turn[0][2],
                MDDt_turn[1][0], MDDt_turn[1][1], MDDt_turn[1][2],
                MDDt_turn[2][0], MDDt_turn[2][1], MDDt_turn[2][2]]]))
# DDturn = np.array([[MDDa_turn[0][0:3], MDDa_turn[1][0:3], MDDa_turn[2],[0:3]],
#                   [MDDg_turn[0],MDDg_turn[1],MDDg_turn[2]],
#                   [MDDt_turn[0],MDDt_turn[1],MDDt_turn[2]]])

# print(DDturn)

print('искажаенные (с шумом и поворотом) буквы  а, г, т')
for x in DDturn:
    print(x, f(x,wa), f(x,wg),f(x,wt))
