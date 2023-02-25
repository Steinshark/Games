import tkinter 
from tkinter import Button, Frame, Label
import tkinter.filedialog
import tkinter.scrolledtext 
from tkinter.ttk import Combobox, Progressbar, Checkbutton
import cleandata
import torch 
from sandboxG import buildBestMod1,buildBest
import torch 
#GENERATOR VALUES 
_G_CONFIG_VALUES        = {
    "init_fn":          None,
}

_G_INIT_KWARGS          = {
    "ncz":              None,
    "Leak":             None,
    "Kernel_ver":       None,
    "Factor_ver":       None,
    "Enable CUDA":      None,
    "Out_ch":           None,
    "Verbose":          None
}

_G_OPTIM_VALUES         = {
    "init_fn":          None,
}

_G_OPTIM_KWARGS         = {
    "params":           None,
    "lr":               None,
    "wd":               None,
    "momentum":         None 
}


#DISCRIMINATOR VALUES 
_D_CONFIG_VALUES        = {
    "init_fn":          None,
} 

_D_INIT_KWARGS          = {
    "ncz":              None,
    "leak":             None,
    "kernel_ver":       None,
    "factor_ver":       None,
    "device":           None,
    "ver":              None,
    "out_ch":           None,
    "verbose":          None
}

_D_OPTIM_VALUES         = {
    "init_fn":          None,
}

_D_OPTIM_KWARGS         = {
    "params":           None,
    "lr":               None,
    "wd":               None,
    "momentum":         None 
}


#COLORS 
Model_Background        = "#0b1517"
Gen_Background          = "#17abe6"
ConfButton_Background   = "#738b91"
Console_Background      = "#0f161c"
Console_Text            = "#9ac1e3"

#FONTS 
MODEL_FONT              = ("Fixedsys",25)
CONSOLE_FONT            = ("Terminal",15)

#VARS 
DEF_DIR                 = "C:/gitrepos/projects/ml/music/models/"
BOOL_VARS               = {"Verbose":None,"Enable CUDA":None}
#Selections 
Gen_inits               = {
        "buildBest":        buildBest,
        "buildBestMod1":    buildBestMod1}

#Save olds 
G_CONF_VALS             = {    
    "ncz":              512,
    "Leak":             .02,
    "Kernel_ver":       0,
    "Factor_ver":       0,
    "Enable CUDA":      None,
    "Out_ch":           1,
    "Verbose":          None,
    "init_fn":          None}

def create_g_panel(app_ref):
    root:tkinter.Frame
    root            = app_ref.frames["generator"]
    root.rowconfigure(0,weight=1)
    root.rowconfigure(1,weight=1)
    root.rowconfigure(2,weight=1)
    root.rowconfigure(3,weight=1)
    root.rowconfigure(4,weight=1)
    root.rowconfigure(5,weight=1)
    root.columnconfigure(0,weight=1)
    root.configure(background=ConfButton_Background)

    #Configure 
    config          = Frame(root,padx=2,pady=1)
    config.configure(background=ConfButton_Background)
    config.rowconfigure(0)
    config.columnconfigure(0,weight=2)
    config.columnconfigure(1,weight=1)
    config_label    = Label(config,text="Config G")
    config_button   = Button(config,text="Conf.",command=lambda: setup_generator(app_ref),padx=1,pady=1)
    config_label.grid(row=0,column=0,sticky=tkinter.EW)
    config_button.grid(row=0,column=1,sticky=tkinter.EW)
    
    config.grid(row=0,column=0,sticky=tkinter.EW)

    #G init file 
    gfile          = Frame(root,padx=2,pady=1)
    gfile.configure(background=ConfButton_Background)
    gfile.rowconfigure(0)
    gfile.columnconfigure(0,weight=2)
    gfile.columnconfigure(1,weight=1)
    gfile_label    = Label(gfile,text="G Init File")
    gfile_button   = Button(gfile,text="Chose",command=lambda: tkinter.filedialog.askopenfile(initialdir=DEF_DIR))
    gfile_label.grid(row=0,column=0,sticky=tkinter.EW)
    gfile_button.grid(row=0,column=1,sticky=tkinter.EW)
    
    gfile.grid(row=1,column=0,sticky=tkinter.EW)

    #Build 
    build           = Frame(root,padx=2,pady=1)
    build.configure(background=ConfButton_Background)
    build.rowconfigure(0)
    build.columnconfigure(0,weight=2)
    build.columnconfigure(1,weight=1)
    build_label    = Label(build,text="Build G")
    build_button   = Button(build,text="Build",command=lambda: build_generator(app_ref))
    build_label.grid(row=0,column=0,sticky=tkinter.EW)
    build_button.grid(row=0,column=1,sticky=tkinter.EW)
    
    build.grid(row=2,column=0,sticky=tkinter.EW)

def create_d_panel(app_ref):
    pass 
    root:tkinter.Frame
    root            = app_ref.frames["discriminator"]
    root.rowconfigure(0,weight=1)
    root.rowconfigure(1,weight=1)
    root.rowconfigure(2,weight=1)
    root.rowconfigure(3,weight=1)
    root.rowconfigure(4,weight=1)
    root.rowconfigure(5,weight=1)
    root.columnconfigure(0,weight=1)
    root.configure(background=ConfButton_Background)

    #Configure 
    config          = Frame(root,padx=2,pady=1)
    config.configure(background=ConfButton_Background)
    config.rowconfigure(0)
    config.columnconfigure(0,weight=2)
    config.columnconfigure(1,weight=1)
    config_label    = Label(config,text="Config D")
    config_button   = Button(config,text="Conf.",command=lambda: setup_generator(app_ref),padx=1,pady=1)
    config_label.grid(row=0,column=0,sticky=tkinter.EW)
    config_button.grid(row=0,column=1,sticky=tkinter.EW)
    
    config.grid(row=0,column=0,sticky=tkinter.EW)

    #G init file 
    gfile          = Frame(root,padx=2,pady=1)
    gfile.configure(background=ConfButton_Background)
    gfile.rowconfigure(0)
    gfile.columnconfigure(0,weight=2)
    gfile.columnconfigure(1,weight=1)
    gfile_label    = Label(gfile,text="D Init File")
    gfile_button   = Button(gfile,text="Chose",command=lambda: tkinter.filedialog.askopenfile(initialdir=DEF_DIR))
    gfile_label.grid(row=0,column=0,sticky=tkinter.EW)
    gfile_button.grid(row=0,column=1,sticky=tkinter.EW)
    
    gfile.grid(row=1,column=0,sticky=tkinter.EW)

    #Build 
    build           = Frame(root,padx=2,pady=1)
    build.configure(background=ConfButton_Background)
    build.rowconfigure(0)
    build.columnconfigure(0,weight=2)
    build.columnconfigure(1,weight=1)
    build_label    = Label(build,text="Build D")
    build_button   = Button(build,text="Build",command=lambda: build_discriminator(app_ref))
    build_label.grid(row=0,column=0,sticky=tkinter.EW)
    build_button.grid(row=0,column=1,sticky=tkinter.EW)
    
    build.grid(row=2,column=0,sticky=tkinter.EW)
#Create the settings frame of window 
def create_settings_frame(app_ref):
    settings_frame                  = Frame(app_ref.window)   
    settings_frame.columnconfigure(0,weight=1)

    settings_frame.rowconfigure(0,weight=1)
    settings_frame.rowconfigure(1,weight=8)
    settings_frame.rowconfigure(2,weight=1)
    settings_frame.rowconfigure(3,weight=8)

    app_ref.frames["settings"]      = settings_frame

    #Create the generator handle 
    app_ref.frames["g_header"]      = Label(settings_frame,text="GENERATOR",font=MODEL_FONT)
    app_ref.frames["g_header"].configure(background=Gen_Background)
    app_ref.frames["g_header"].grid(row=0,column=0,sticky=tkinter.NSEW)

    app_ref.frames["generator"]     = Frame(settings_frame)
    create_g_panel(app_ref)
    app_ref.frames["generator"].grid(row=1,column=0,sticky=tkinter.NSEW)
    
    
    #Create the generator handle 
    app_ref.frames["d_header"]      = Label(settings_frame,text="DISCRIMINATOR",font=MODEL_FONT)
    app_ref.frames["d_header"].configure(background=Gen_Background)
    app_ref.frames["d_header"].grid(row=2,column=0,sticky=tkinter.NSEW)

    app_ref.frames["discriminator"] = Frame(settings_frame)
    create_d_panel(app_ref)
    app_ref.frames["discriminator"].grid(row=3,column=0,sticky=tkinter.NSEW)

    app_ref.frames["settings"].grid(column=0,row=1,sticky=tkinter.NSEW)
    app_ref.window.grid() 

def create_console_frame(app_ref):
    app_ref.console     = tkinter.scrolledtext.ScrolledText(app_ref.window)
    app_ref.console.configure(background=Console_Background,foreground=Console_Text,font=CONSOLE_FONT)
    app_ref.console.grid(row=1,column=1,sticky=tkinter.NSEW)

def create_telemetry_frame(app_ref):
    pass  

#save value retrieval from a tkinter frame 
def grab_entry_item(frame_ref,val_type):

    #Grab item in the frame 
    raw_val     = frame_ref.get()

    #Attempt type cast 
    raw_val     = val_type(raw_val)

    #Ensure correct type 
    if not isinstance(raw_val,val_type):
        raise TypeError("improper type discovered in entry")
    else:
        return 

def build_generator(app_ref,state_dict_file=None):

    kwargs:dict 
    init_fn:function 

    kwargs      = {k.lower(): _G_INIT_KWARGS[k] for k in _G_INIT_KWARGS}
    kwargs["device"] = torch.device('cuda') if kwargs["enable cuda"] else torch.device("cpu")
    del kwargs["enable cuda"]

    print(f"going with vals:")
    import pprint 
    pprint.pp(kwargs)
    init_fn     = _G_CONFIG_VALUES['init_fn']

    #Check init_fn
    if not init_fn:
        app_ref.print("No init_fn for Generator supplied!")
        return
    #Attempt instantiation 
    generator:torch.nn.Module
    generator   = init_fn(**kwargs)

    if state_dict_file:
        state   = torch.load(state_dict_file)
        generator.load_state_dict(state)

    app_ref.G   = generator
    app_ref.print(f"Created Generator Model - {(sum([p.numel()*p.element_size() for p in generator.parameters()])/1000000):.2f}MB")

def build_discriminator(state_dict_file=None):
    Disciminator    = cleandata._D

    if state_dict_file:
        state   = torch.load(state_dict_file)
        Disciminator.load_state_dict(state)

    return Disciminator.to(_D_INIT_KWARGS["device"])

def return_vals(window:tkinter.Tk,valuelist,app_ref):
    global  G_CONF_VALS,_G_CONFIG_VALUES
    vals    = {f:0 for f in valuelist}
    change_flag = False 
    for entry in valuelist:
        if entry in ["Leak"]:
            try:
                val = float(valuelist[entry].get())
                _G_INIT_KWARGS[entry]   = val 
            except ValueError:
                app_ref.print(f"bad val '{valuelist[entry].get()}' found in {entry}")
                return
        elif entry in ["ncz","Kernel_ver","Factor_ver","Ver","Out_ch"]:
            try:
                val = int(valuelist[entry].get())
                _G_INIT_KWARGS[entry]   = val 
            except ValueError:
                app_ref.print(f"bad val '{valuelist[entry].get()}' found in {entry}")
                return
        elif entry in ["Enable CUDA","Verbose"]:
            val = BOOL_VARS[entry].get()
            _G_INIT_KWARGS[entry]   = val 
        elif entry in ["init_fn"]:
            val = valuelist[entry].get()
            _G_CONFIG_VALUES['init_fn'] = Gen_inits[val]

        
        vals[entry]             = val
    G_CONF_VALS = vals 
    G_CONF_VALS["init_fn"] = list(Gen_inits.keys()).index(vals["init_fn"])

    window.destroy()
    app_ref.temp_vals = vals 
    app_ref.print("Set Generator initialization values")

def setup_generator(app_ref):
    global G_CONF_VALS, BOOL_VARS
    label_w     = 20 
    popup_geo   = "300x400"

    popup       = tkinter.Tk()
    popup.geometry(popup_geo)
    for i in range(len(_G_INIT_KWARGS)+1):
        popup.rowconfigure(i,weight=1)

    popup.columnconfigure(0,weight=1)
    popup.configure(background="#545C6A")

    frames  = {key : tkinter.Frame(popup,width=100,pady=1) for key in _G_INIT_KWARGS}
    frames["init_fn"] = tkinter.Frame(popup,width=100,pady=1)

    values  = {}

    for i,frame in enumerate(frames):
        print(frame)
        frames[frame].columnconfigure(0,weight=5)
        frames[frame].columnconfigure(1,weight=3)
        frames[frame].rowconfigure(0,weight=1)

        label   = tkinter.Label(frames[frame],text=frame,width=label_w)
        label.grid(row=0,column=0)


        entry   = tkinter.Entry(frames[frame])
        if not G_CONF_VALS[frame] is None:
            entry.delete(0,tkinter.END)
            entry.insert(0,G_CONF_VALS[frame])
        
        if frame == "init_fn":
            entry           = Combobox(frames[frame],textvariable=tkinter.StringVar(),state="readonly")
            entry['values'] = list(Gen_inits.keys())
            entry.current(0)
            if G_CONF_VALS:
                entry.current(G_CONF_VALS[frame])
        elif frame in ["Enable CUDA","Verbose"]:
            BOOL_VARS[frame]    = tkinter.BooleanVar()
            entry           = Checkbutton(frames[frame],variable=BOOL_VARS[frame],onvalue=True,offvalue=False)

        
            
        entry.grid(row=0,column=1)            
            
        values[frame]   = entry 
        frames[frame].grid(row=i,column=0,padx=0,pady=1,sticky=tkinter.EW)
    submit_frame    = tkinter.Button(popup,text="Submit",command=lambda : return_vals(popup,values,app_ref))
    submit_frame.grid(row=i+1,column=0)
    popup.grid()
    popup.mainloop()

if __name__ == "__main__":
    setup_generator(None)
