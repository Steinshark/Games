import tkinter as tk 
from tkinter import ttk
from tkinter import ttk, Button, Entry, Frame, Label, OptionMenu,StringVar

from tkinter.ttk import Combobox
from PIL import Image,ImageTk
import os 
if "win" in os.name:
    from ctypes import windll
import random
from matplotlib import pyplot as plt 
import random 
import pygame
from trainer import Trainer



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


def play_game():
    pass 

def show_game():
    pass

class GameBoard:

    def __init__(self,x,y,grid_x,grid_y,title):
        #Prep pygame
        pygame.init()
        self.WINDOW = pygame.display.set_mode((x,y))
        self.WINDOW.set_caption(title)

        self.red    = (255,0,0)
        self.green  = (1,255,40)
        #Some calculations
        self.grid_w = int(x / grid_x) 
        self.grid_h = int(y / grid_y)

        

    def update_display(self,snake,food):
        self.WINDOW.fill((0,0,0))

        #Draw Snake
        for segment in snake:
            x,y = segment
            upper_l = (x*self.grid_w,y*self.grid_h)
            pygame.draw.rect(self.WINDOW,self.green,pygame.Rect(upper_l[0],upper_l[1],self.grid_w,self.grid_h))
        #Draw Food
        pygame.draw.rect(self.WINDOW,self.green,pygame.Rect(food[0]*self.grid_w,food[1]*self.grid_h,self.grid_w,self.grid_h))

        pygame.display.flip()



class TrainerApp:

    def __init__(self,width,height):

        #Build window 
        self.window         = tk.Tk()
        self.window         .geometry(str(width)+ "x" +str(height))
        self.window.grid()

        #Build general frames
        self.top_frame      = tk.Frame(self.window,height=height/30)
        self.control_frame  = tk.Frame(self.window,width=width/5)
        self.view_frame     = tk.Frame(self.window,width=5*width/5,height=29*height/30)

        self.top_frame.grid(row=0,column=0,columnspan=2,sticky=tk.EW)
        self.control_frame.grid(row=1,column=0,sticky=tk.NSEW)
        self.view_frame.grid(row=1,column=1,sticky=tk.NSEW)

        self.top_frame.configure(background="blue")
        self.control_frame.configure(background="red")
        self.view_frame.configure(background="yellow")
        #Keep track of settings
        self.settings = {   "gameX"     : None,
                            "gameY"     : None,
                            "iters"     : None,
                            "arch"      : None
        }




        self.setting_frames = {

                        "gameX" : Frame(self.control_frame,padx=1,pady=1),
                        "gameY" : Frame(self.control_frame,padx=1,pady=1),
                        "iters" : Frame(self.control_frame,padx=1,pady=1),
                        "te"    : Frame(self.control_frame,padx=1,pady=1),
                        "ps"    : Frame(self.control_frame,padx=1,pady=1),
                        "ss"    : Frame(self.control_frame,padx=1,pady=1),
                        "bs"    : Frame(self.control_frame,padx=1,pady=1),
                        "ep"    : Frame(self.control_frame,padx=1,pady=1),
                        "ms"    : Frame(self.control_frame,padx=1,pady=1),
                        "mx"    : Frame(self.control_frame,padx=1,pady=1),
                        "sf"    : Frame(self.control_frame,padx=1,pady=1),
                        "arch"  : Frame(self.control_frame,padx=1,pady=1),
                        "lo"    : Frame(self.control_frame,padx=1,pady=1),
                        "op"    : Frame(self.control_frame,padx=1,pady=1),
                        "tr"    : Frame(self.control_frame,padx=1,pady=1)
        }
        #Build control Frame 

        label_w         =   15
        label_h         =   1
        self.labels    = {      "gameX"     :   Label( self.setting_frames["gameX"],
                                                    text="Game X",width=label_w,height=label_h),
                                "gameY"     :   Label( self.setting_frames["gameY"],
                                                    text="Game Y",width=label_w,height=label_h),
                                "iters"     :   Label( self.setting_frames["iters"],
                                                    text="Iters",width=label_w,height=label_h),
                                "te"        :   Label( self.setting_frames["te"],
                                                    text="Train Rate",width=label_w),
                                "ps"        :   Label( self.setting_frames["ps"],
                                                    text="Pool Size",width=label_w,height=label_h),
                                "ss"        :   Label( self.setting_frames["ss"],
                                                    text="Sample Size",width=label_w,height=label_h),
                                "bs"        :   Label( self.setting_frames["bs"],
                                                    text="Batch Size",width=label_w,height=label_h),
                                "ep"        :   Label( self.setting_frames["ep"],
                                                    text="Epochs",width=label_w,height=label_h),
                                "ms"        :   Label( self.setting_frames["ms"],
                                                    text="Memory Size",width=label_w,height=label_h),
                                "mx"        :   Label( self.setting_frames["mx"],
                                                    text="Max Steps",width=label_w,height=label_h),
                                "sf"        :   Label( self.setting_frames["sf"],
                                                    text="Smooth Factor",width=label_w,height=label_h),
                                "arch"      :   Label( self.setting_frames["arch"],
                                                    text="Arch",width=label_w-2,height=label_h),
                                "lo"        :   Label( self.setting_frames["lo"],
                                                    text="Loss Function",width=label_w,height=label_h),
                                "op"        :   Label( self.setting_frames["op"],
                                                    text="Optimizer",width=label_w,height=label_h),
                                "tr"        :   Label( self.setting_frames["tr"],
                                                    text="Transfer Rate",width=label_w,height=label_h),
                                        
        }


        arch_options = ["FCN_1","CNN_1","CNN_2","CNN_3","CNN_4","CNN_5"]
        entry_w = 5
        self.fields     = {     "gameX"     :   Entry(self.setting_frames["gameX"],width=entry_w),
                                "gameY"     :   Entry(self.setting_frames["gameY"],width=entry_w),
                                "iters"     :   Entry(self.setting_frames["iters"],width=entry_w),
                                "te"        :   Entry(self.setting_frames["te"],width=entry_w),
                                "gameY"     :   Entry(self.setting_frames["gameY"],width=entry_w),
                                "iters"     :   Entry(self.setting_frames["iters"],width=entry_w),
                                "te"        :   Entry(self.setting_frames["te"],width=entry_w),
                                "ps"        :   Entry(self.setting_frames["ps"],width=entry_w),
                                "ss"        :   Entry(self.setting_frames["ss"],width=entry_w),
                                "bs"        :   Entry(self.setting_frames["bs"],width=entry_w),
                                "ep"        :   Entry(self.setting_frames["ep"],width=entry_w),
                                "ms"        :   Entry(self.setting_frames["ms"],width=entry_w),
                                "mx"        :   Entry(self.setting_frames["mx"],width=entry_w),
                                "sf"        :   Entry(self.setting_frames["sf"],width=entry_w),
                                "arch"      :   Combobox(self.setting_frames["arch"],width=entry_w+2,textvariable=StringVar()),
                                "lo"        :   Entry(self.setting_frames["lo"],width=entry_w),
                                "op"        :   Entry(self.setting_frames["op"],width=entry_w),
                                "tr"        :   Entry(self.setting_frames["tr"],width=entry_w)
        }


        self.fields['arch']['values'] = arch_options
        for i,(frame,b,f) in enumerate(zip(self.setting_frames,self.labels,self.fields)):
            self.labels[f].pack(side=tk.LEFT,anchor=tk.E,padx=0,pady=0)
            self.fields[b].pack(side=tk.RIGHT,anchor=tk.W,padx=0,pady=0)

            self.setting_frames[frame].grid(row=i+1,column=0,pady=1,padx=0)
        
        



    def run_loop(self):
        self.window.mainloop()


    def set_var(self,s_key,b_key):
        self.settings[s_key] = self.fields[b_key].get()

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
    
    ta = TrainerApp(1200,800)

    ta.run_loop()




