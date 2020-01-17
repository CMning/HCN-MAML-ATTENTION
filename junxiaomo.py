from tkinter import *

def change_tex():
    my_label.config(text='孙博是小肥猪')


window =Tk()
window.title('fat-pig')

my_label=Label(window,width=50,height=5,text='')
my_label.grid(row=0,column=0)

my_button=Button(window,text='谁是小肥猪',width=10,command=change_tex)
my_button.grid(row=1,column=0)

window.mainloop()