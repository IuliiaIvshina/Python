import numpy as np
import pandas as pd 
# чтение Excel в фрейме данных 
df = pd.read_excel('w_1_1_1.xlsx') 
 
# преобразовать фрейм данных в 2D-массив Numpy 
M_array = np.asarray(df)

# создание единичной матрицы
M_eye = np.eye(M_array.shape[0])
print(M_eye.shape)
M_product = np.dot(M_eye, M_array)

#запись данных в exe
#data_exl = pd.DataFrame({'TransferBudget':M_product})
np.savetxt('./result.txt',M_product)