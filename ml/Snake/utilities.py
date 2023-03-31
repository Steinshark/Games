import random 
import math 
import time 
import torch 
import numpy 


TIME_MULT   = 0 
VECT_TYPE   = torch.float16
SNAKE_SQ    = torch.ones(size=(1,1))
FOOD_SQ     = torch.ones(size=(1,1))
TOP_L       = (0,0)
BOT_R       = (0,0)
SQUARE_SF   = 0 
DEV         = torch.device('cuda')

def init_utils(board_size,img_w,img_h,vect_type,colors={"snake":(36/255,199/255,104/255),"food":(207/255,29/255,29/255)},device=torch.device('cuda')):
    global TOP_L,BOT_R,FOOD_SQ,SNAKE_SQ,VECT_TYPE,SQUARE_SF,DEV

    w                           = board_size[0]
    h                           = board_size[1]
    board_ar                    = w / h 
    DEV                         = device
    VECT_TYPE                   = vect_type

    if board_ar > img_w/img_h:                                     #If too wide, scale to sides 
        SQUARE_SF                   = int(img_w / w)
        TOP_L                       = (0, int((img_h - h*SQUARE_SF) / 2))
        BOT_R                       = (img_w,int(img_h - (img_h - h*SQUARE_SF) / 2))

    else:  
        SQUARE_SF                   = int(img_h / h) 
        TOP_L                       = (int((img_w - w*SQUARE_SF) / 2), 0)
        BOT_R                       = (img_w - int((img_w - w*SQUARE_SF) / 2), img_h)

    SNAKE_SQ                        = torch.zeros(size=(3,SQUARE_SF,SQUARE_SF))
    SNAKE_SQ[0]                     = torch.full(size=(SQUARE_SF,SQUARE_SF),fill_value=colors['snake'][0])                 
    SNAKE_SQ[1]                     = torch.full(size=(SQUARE_SF,SQUARE_SF),fill_value=colors['snake'][1])
    SNAKE_SQ[2]                     = torch.full(size=(SQUARE_SF,SQUARE_SF),fill_value=colors['snake'][2])

    FOOD_SQ                         = torch.zeros(size=(3,SQUARE_SF,SQUARE_SF))
    FOOD_SQ[0]                      = torch.full(size=(SQUARE_SF,SQUARE_SF),fill_value=colors['food'][0])                 
    FOOD_SQ[1]                      = torch.full(size=(SQUARE_SF,SQUARE_SF),fill_value=colors['food'][1])
    FOOD_SQ[2]                      = torch.full(size=(SQUARE_SF,SQUARE_SF),fill_value=colors['food'][2])

      
def build_snake_img(snake_list,food_loc,board_size,img_w=1280,img_h=720):

    vect_init_type              = torch.float16
    #create base tensor in CxWxH 
    w                           = board_size[0]
    h                           = board_size[1]

    frame_repr_tensor           = torch.zeros(size=(3,img_h,img_w),dtype=vect_init_type)

    #Find limiting scale  

    for sq in snake_list:
        sq_topl_x                       = int(TOP_L[0] + sq[0]*SQUARE_SF) 
        sq_topl_y                       = int(TOP_L[1] + sq[1]*SQUARE_SF) 
        

        sq_topl_x                       = int(TOP_L[0] + sq[0]*SQUARE_SF) 
        sq_topl_y                       = int(TOP_L[1] + sq[1]*SQUARE_SF) 
        frame_repr_tensor[:,sq_topl_y:sq_topl_y+SQUARE_SF,sq_topl_x:sq_topl_x+SQUARE_SF]      = SNAKE_SQ
    
    sq_topl_x                    = int(TOP_L[0] + food_loc[0]*SQUARE_SF) 
    sq_topl_y                    = int(TOP_L[1] + food_loc[1]*SQUARE_SF) 

    frame_repr_tensor[:,sq_topl_y:sq_topl_y+SQUARE_SF,sq_topl_x:sq_topl_x+SQUARE_SF]          = FOOD_SQ
    
    return frame_repr_tensor.to(DEV)


def step_snake_img(game_vector,snake_list,food_loc,board_size,img_w=1280,img_h=720,colors={"snake":(36/255,199/255,104/255),"food":(207/255,29/255,29/255)},dim_fact=.33,vect_init_type=torch.float16,min_thresh=.03):    
    global TIME_MULT,SNAKE_SQ,FOOD_SQ,TOP_L,BOT_R
    w                           = board_size[0]
    h                           = board_size[1]

    MIN     = torch.nn.Threshold(min_thresh,0,True) 


    #Dim playable surface and set threshold of 5 for pixels
    game_vector[:,TOP_L[1]:BOT_R[1],TOP_L[0]:BOT_R[0]] *= dim_fact 
    MIN(game_vector)
    
    t0 = time.time()
    for sq in snake_list:
        sq_topl_x                       = int(TOP_L[0] + sq[0]*SQUARE_SF) 
        sq_topl_y                       = int(TOP_L[1] + sq[1]*SQUARE_SF) 
        game_vector[:,sq_topl_y:sq_topl_y+SQUARE_SF,sq_topl_x:sq_topl_x+SQUARE_SF]      = SNAKE_SQ
    TIME_MULT += time.time()-t0
    sq_topl_x                    = int(TOP_L[0] + food_loc[0]*SQUARE_SF) 
    sq_topl_y                    = int(TOP_L[1] + food_loc[1]*SQUARE_SF) 

    game_vector[:,sq_topl_y:sq_topl_y+SQUARE_SF,sq_topl_x:sq_topl_x+SQUARE_SF]          = FOOD_SQ
                
    
    #Add current snake 
    return game_vector
        
def reduce_arr(arr,newlen):

    #Find GCF of len(arr) and len(newlen)
    gcf         = math.gcd(len(arr),newlen)
    mult_fact   = int(newlen / gcf) 
    div_fact    = int(len(arr) / gcf) 

    new_arr     = numpy.repeat(arr,mult_fact)


    return [sum(list(new_arr[n*div_fact:(n+1)*div_fact]))/div_fact for n in range(newlen)]





if __name__ == "__main__":
    pass