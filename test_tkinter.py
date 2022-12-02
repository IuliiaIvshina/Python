from tkinter import *

# создание графического интерфейса
window = Tk()
window.title('Добро пожаловать в приложение')
Label(window, text='Label 1',bg='white').pack(
    fill='both',
    expand=True
)
Label(window, text='Label 2',bg='blue').pack(
    fill='both',
    expand=True
)
Label(window, text='Label 3',bg='red').pack(
    fill='both',
    expand=True
)
window.bind('<Escape>', lambda x: window.destroy())
window.mainloop()

