import random 
import math 
import time 
import torch 





def build_snake_img(snake_list,food_loc,board_size,img_w=1280,img_h=720,colors={"snake":(36/255,199/255,104/255),"food":(207/255,29/255,29/255)},dev=torch.device('cuda')):

    vect_init_type              = torch.float16
    #create base tensor in CxWxH 
    w                           = board_size[0]
    h                           = board_size[1]

    frame_repr_tensor           = torch.zeros(size=(3,img_h,img_w),dtype=vect_init_type)

    snake_color                 = torch.tensor(colors['snake'],dtype=vect_init_type)
    food_color                  = torch.tensor(colors['food'],dtype=vect_init_type)

    #Find limiting scale  
    board_ar                    = w / h 
    if board_ar > img_w/img_h:                                     #If too wide, scale to sides 
        square_sf                   = int(img_w / w) 
        top_l                       = (0, int((img_h - h*square_sf) / 2))
        bot_r                       = (img_w,int(img_h - (img_h - h*square_sf) / 2))    
        frame_repr_tensor[:,0:top_l[1],:]                   = torch.ones(size= (3,top_l[1],img_w),dtype=vect_init_type)
        frame_repr_tensor[:,bot_r[1]:img_h,:]               = torch.ones(size= (3,top_l[1],img_w),dtype=vect_init_type)
    else:  
        square_sf                   = int(img_h / h) 
        top_l                       = (int((img_w - w*square_sf) / 2), 0)
        bot_r                       = (img_w - int((img_w - w*square_sf) / 2), img_h)

        frame_repr_tensor[:,:,:top_l[0]]                     = torch.ones(size= (3,img_h,top_l[0]),dtype=vect_init_type)
        frame_repr_tensor[:,:,bot_r[0]:img_w]               = torch.ones(size= (3,img_h,top_l[0]),dtype=vect_init_type)

    for sq in snake_list:
        sq_topl_x                       = int(top_l[0] + sq[0]*square_sf) 
        sq_topl_y                       = int(top_l[1] + sq[1]*square_sf) 
        

        for y_pixl in range(sq_topl_y,sq_topl_y+square_sf):
            for x_pixl in range(sq_topl_x,sq_topl_x+square_sf):
                 
                frame_repr_tensor[:,y_pixl,x_pixl]                = snake_color
                #frame_repr_tensor[1,y_pixl,x_pixl]                = colors["snake"][1]
                #frame_repr_tensor[2,y_pixl,x_pixl]                = colors["snake"][2]
    
    sq_topl_x                    = int(top_l[0] + food_loc[0]*square_sf) 
    sq_topl_y                    = int(top_l[1] + food_loc[1]*square_sf) 

    for y_pixl in range(sq_topl_y,sq_topl_y+square_sf):
        for x_pixl in range(sq_topl_x,sq_topl_x+square_sf):
                
            frame_repr_tensor[:,y_pixl,x_pixl]                = food_color
            #frame_repr_tensor[1,y_pixl,x_pixl]                = colors["food"][1]
            #frame_repr_tensor[2,y_pixl,x_pixl]                = colors["food"][2]
    
    return frame_repr_tensor.to(dev)


