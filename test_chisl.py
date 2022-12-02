
"""
# работа с изображениями
import numpy as np
import matplotlib.pyplot as plt

# чтение файла изображения
img = plt.imread('tilimili.jpg')
plt.imshow(img)

# поворот рисунка на 30 градусов
from scipy import ndimage
rotated = ndimage.rotate(img,30,reshape=0)

# охранение изображения
plt.imshow(rotated)
plt.imsave(fname='rotated.png',arr=rotated)

# преобразование для выделения краев
from scipy import ndimage
sob = ndimage.sobel(img)
plt.imsave(fname='sob.png',arr=sob)

# вырезание части рисунка
from scipy import ndimage
lx, ly, c = img.shape
crop = img[lx // 4: -lx // 4, ly // 4: -ly // 4]
plt.imsave(fname='crop.png',arr=crop)

# размытие изображения
from scipy import ndimage
blur = ndimage.gaussian_filter(img, sigma=5)
plt.imsave(fname='blur.png',arr=blur)
"""

# четные и нечентные цифры
# распознавание образов
import numpy as np
w = np.zeros((3)) # инициализация весов

# исходные данные
D = np.array([
    [0,0,0],
    [0,0,1],
    [0,1,0],
    [1,0,0],
    [0,1,1]
]) 
# выходной вектор
Y0 = np.array([0,1,0,0,1])

a = 0.2 # темп  обучения
b = -0.4 # торможение (смещение)

# активационная функция
sigm = lambda x: 1 if x > 0 else 0

# тело нейрона
def f(x):
    s = b + np.sum(x @ w)
    return sigm(s)

# эпоха обучения
def train():
    global w
    _w = w.copy()
    for x, y in zip(D, Y0):
        w += a * (y - f(x)) * x
    return (w != _w).any()  

# обучение и тестирование
while train():
    print(w) 

print(f([1,0,0]))
print(f([1,0,1]))
print(f([1,0,0.2]))
