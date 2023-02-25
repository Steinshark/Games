#GUI items 
import tkinter
from tkinter.scrolledtext import ScrolledText 

#Tensor Items 
import torch 

#Aux items 
import os 
import sys 
import time 
import random 
import json 
import numpy 
import pprint 

#Pre Classed items 
from SoundBooth import AudioDataSet, Trainer
from config import * 


class Studio:

    #Init app
    def __init__(self,width=1920,height=1080):

        #Settings 
        settings_col_weight     = 1 
        console_col_weight      = 1
        telemety_col_weight     = 5 
        title_row_weight        = 1
        work_row_weight         = 16

        self.window = tkinter.Tk()
        self.window.geometry(f"{width}x{height}")
        self.window.resizable()
        
        #CONFIG COLS 
        self.window.columnconfigure(0,weight=settings_col_weight)
        self.window.columnconfigure(1,weight=console_col_weight)
        self.window.columnconfigure(2,weight=telemety_col_weight)

        #CONFIG ROWS 
        self.window.rowconfigure(0,weight=title_row_weight)
        self.window.rowconfigure(1,weight=work_row_weight)

        #Keep track of all values 
        self.values = {}    
        self.frames = {}
        self.console: ScrolledText

        #Create layout 
        self.create_layout()

        #Run the main loop 
        self.window.mainloop()


    #Create the standard layout 
    def create_layout(self):
        
        create_settings_frame(self)
        create_console_frame(self)
        create_telemetry_frame(self)
    
    def print(self,output,end="\n"):
        self.console.insert(tkinter.END,output+end)

    

#Create a window 
if __name__ == "__main__":
    
    #Create window with a set size 
    s   = Studio()