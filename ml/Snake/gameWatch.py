import tkinter as tk 
from tkinter import ttk 
from PIL import Image,ImageTk

from ctypes import windll
windll.shcore.SetProcessDpiAwareness(1)

WINDOW_W = 1200
WINDOW_H = 800
GAME_IMG = None 
GAME_IMG2 = None
GAMES_LIBRARY = {}

def load_game_shot(canvas:tk.Canvas):
    global GAME_IMG
    fname = r"C:\users\steinshark\pictures\out-0.png"
    GAME_IMG = ImageTk.PhotoImage(Image.open(fname))

    print(GAME_IMG.width())
    print(GAME_IMG.height())
    print(canvas.winfo_reqheight())
    print(canvas.winfo_reqwidth())

    canvas.create_image(0,0,anchor="nw",image=GAME_IMG)

#    OPTIONS                                        GAME 
######################################################################################
#                       #
#                       #
#                       #
#                       #
#                       #
#                       #
#                       #
#                       #
#                       #
#                       #
#                       #
#                       #
#                       #
#                       #
#                       #
#                       #
#                       #
#                       #
#                       #
#                       #
######################################################################################
#Create the root frame 
root = tk.Tk()
root.geometry(f'{WINDOW_W}x{WINDOW_H}')
root.resizable = True 
root.title("Game Viwer")

#Create the basic layout
progress_bar    = ttk.Progressbar(root,orient='horizontal',mode='indeterminate',length=200)
canvas          = tk.Canvas(root,width=512,height=512)
fr              = tk.Frame(root)



root.grid() 

progress_bar.grid(column=0,row=0)
fr.grid(column=0,row=1)
canvas.grid(column=1,row=0,rowspan=2)
load_game_shot(canvas)


root.mainloop()



