import tkinter
import random
from tkinter import Tk

def click(event, target):

    print("I was clicked!")
    x = top.winfo_width()
    y = top.winfo_height()
    target.place(x=random.randint(a=0, b=x - 50), y=random.randint(a=0, b=y - 20))

    target.config(text="press me")
    return None

top: Tk = tkinter.Tk()
top.title("一个简单的界面")
top.geometry("300x180+0+0")
top.resizable(width=True, height=True)
top.minsize(width=300, height=180)
Button_1 = tkinter.Button()
Button_1.config(text="press me")
Button_1.bind(sequence="<Button-1>", func=lambda event: click(event, target=Button_1))
Button_1.place(x=20, y=20)
tkinter.mainloop()
