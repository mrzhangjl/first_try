import tkinter as tk
from datetime import datetime
from tkinter import font

from Cython.Shadow import nonecheck

root = tk.Tk()
root.config(bd=12,
            bg='#aaffaa')
root.geometry('800x480')
root.minsize(width=800, height=480)
frame1 = tk.Frame(root)
frame1.config(highlightcolor='#aaffaa',
              highlightbackground='#aaffaa',
              bg='#aaffaa',
              )
frame1.pack(side='top',
            fill='x',
            expand=0,)
frame2 = tk.Frame(root)
frame2.config(bg='#aaffaa',
              )
frame2.pack(side='top',
            fill='both',
            expand=0)

text1 = tk.Text(frame1)
text1.config(width=8,
             height=1,
             bd=2,
             bg='#aaffaa',
             highlightthickness=12,
             highlightcolor='#aaffaa',
             highlightbackground='#aaffaa',
             font=font.Font(size=42))
text1.pack(side='right',
           fill='none',
           expand=0)
text2 = tk.Text(frame2)
text2.config(width=80,
             height=2,
             bd=2,
             bg='#aaffaa',
             highlightthickness=12,
             highlightcolor='#aaffaa',
             highlightbackground='#aaffaa')
text2.pack(side='right',
           fill='both',
           expand=0)


class APP_Menu:
    def __init__(self, root):
        self.root = root
        self.menu = tk.Menu(root)
        self.file_menu = tk.Menu(self.menu, tearoff=0)
        self.file_menu.add_command(
            label='新建(N)',
            command=self.new_file,
            accelerator='Ctrl+N')
        self.file_menu.add_command(
            label='新窗口(W)',
            command=self.new_window,
            accelerator='Ctrl+Shift+W')
        self.file_menu.add_command(
            label='打开(O)...',
            command=self.open_file,
            accelerator='Ctrl+O')
        self.file_menu.add_command(
            label='保存(S)',
            command=self.save_file,
            accelerator='Ctrl+S')
        self.file_menu.add_command(
            label='另存为(A)...',
            command=self.save_as_file,
            accelerator='Ctrl+Shift+S')
        self.file_menu.add_separator()
        self.file_menu.add_command(
            label='页面设置(U)...',
            command=self.pages_attribute_set)
        self.file_menu.add_command(
            label='打印(P)...',
            command=self.print_file,
            accelerator='Ctrl+P')
        self.file_menu.add_separator()
        self.file_menu.add_command(label='退出(X)', command=self.exit)
        self.menu.add_cascade(label='文件(F)', menu=self.file_menu)
        root.config(menu=self.menu)

    def new_file(self):
        pass

    def new_window(self):
        pass

    def open_file(self):
        pass

    def save_file(self):
        pass

    def save_as_file(self):
        pass

    def pages_attribute_set(self):
        pass

    def print_file(self):
        pass

    def exit(self):
        self.root.destroy()


menu = APP_Menu(root)


def flush_data():
    time_now_str = datetime.now().strftime('%H:%M:%S')
    text1.delete(index1=1.0, index2=tk.END)
    text1.insert(index='1.0', chars=time_now_str)
    text2.delete(index1=1.0, index2=tk.END)
    text2.insert(index='1.0', chars=time_now_str)
    root.after(1000, flush_data)


flush_data()
tk.mainloop()
