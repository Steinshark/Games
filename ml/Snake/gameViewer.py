import tkinter as tk 
from tkinter import ttk 
from PIL import Image,ImageTk
from ctypes import windll
import random
import os 
from matplotlib import pyplot as plt 




def plot_game(scores_list=[],steps_list=[],series_names="Empty",x_scales=[],graph_name="NoName",f_name="iterations"):
    fig, axs = plt.subplots(2,1)
    fig.set_size_inches(19.2,10.8)

    for sc,li,x,na in zip(scores_list,steps_list,x_scales,series_names):
        axs[0].plot(x,sc,label=na)
        axs[1].plot(x,li,label=na)

    axs[0].legend()
    axs[0].set_title("Average Score")
    axs[1].legend()
    axs[1].set_title("Average Steps")

    axs[0].set_xlabel("Game Number")
    axs[0].set_ylabel("Score")
    axs[1].set_xlabel("Game Number")
    axs[1].set_ylabel("Steps Taken")
    fig.suptitle(graph_name)
    #Save fig to figs directory
    if not os.path.isdir("figs"):
        os.mkdir("figs")
    fig.savefig(os.path.join("figs",f"{f_name}.png")    )

def load_game_shot(canvas:tk.Canvas,snake_body):
    global GAME_IMG
    global PIXEL_LIST
    for row in range(canv_im_h):
        for col in range(canv_im_w):
            #print(PIXEL_LIST[row,col])
            if random.random() < .25:
                PIXEL_LIST[row,col] = (255,255,255)
    fname = r"C:\users\steinshark\pictures\out-0.png"
    GAME_IMG = ImageTk.PhotoImage(RAW_IMG)

    canvas.create_image(0,0,anchor="nw",image=GAME_IMG)


if __name__ == "__main__":
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
    windll.shcore.SetProcessDpiAwareness(1)
    WINDOW_W = int(1600 * 12/9)
    WINDOW_H = 1200
    GAME_IMG = None 
    GAME_IMG2 = None
    GAMES_LIBRARY = {}


    RAW_IMG = Image.new('RGB',(1024,1024),color=(0,0,0))

    PIXEL_LIST   = RAW_IMG.load()
    canv_im_w, canv_im_h = RAW_IMG.size
    root = tk.Tk()
    root.geometry(f'{WINDOW_W}x{WINDOW_H}')
    root.resizable = True 
    root.title("Game Viwer")

    #Create the basic layout
    progress_bar    = ttk.Progressbar(root,orient='horizontal',mode='indeterminate',length=200)
    canvas          = tk.Canvas(root,width=WINDOW_W,height=WINDOW_H)
    fr              = tk.Frame(root)



    root.grid() 

    progress_bar.grid(column=0,row=0)
    fr.grid(column=0,row=1)
    canvas.grid(column=1,row=0,rowspan=2)
    load_game_shot(canvas,[])


    root.mainloop()



