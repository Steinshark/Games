import urllib.request
from PIL import Image
import urllib.error
import torch 
import random 
from  torch.utils.data import Dataset,DataLoader
import os 
from torch.nn.functional import interpolate 
from torchvision import transforms
import socket 
import numpy 
import json 
import ssl 

from matplotlib import pyplot as plt 
socket.setdefaulttimeout(1)

save_loc  	= r"C:/data/images/converted_tensors/"
MULT        = 2/255
CORRECTOR   = 0.7142857313156128
AVGS        = []



#This function is used to convert the tensor from 3*y*x tensor to 3*512*768 for use in our network training
def crop_to_ar(img:torch.Tensor,ar=1,target_ar=1,target_dim:int=512):

    
    #Check img dims compatable 
    if not len(img.shape) == 3:
        raise ValueError(f"bad shape {img.shape} must be 2d img")
    
    img_x   = img.shape[2]
    img_y   = img.shape[1]

    #Fix AR
    if not (img_x / img_y == target_ar):
        ar          = img_x / img_y
        removed_x   = 'l'
        removed_y   = 't'
        while not (ar == target_ar):
            
            #Remove a side 
            if ar > target_ar:
                if removed_x == 'l':
                    img     = img[:,:,1:]
                    removed_x  = 'r'
                else:
                    img     = img[:,:,:-1]
                    removed_x  = 'l'
            elif ar < target_ar:
                if removed_y == 't':
                    img     = img[:,:-1,:]
                    removed_y   = 'b'
                else:
                    img     = img[:,1:,:]
                    removed_y   = 't'
            img_x   = img.shape[2]
            img_y   = img.shape[1]
            ar          = img_x / img_y 
    #INFO


    img     = img.unsqueeze_(dim=0).float()
    img     = interpolate(img,size=(target_dim,target_dim*target_ar))[0].half()
    return img


def downsample(img:torch.Tensor,stage=0):
    if stage == 0:
        return interpolate(img,size=(32,48))
    elif stage == 1:
        return interpolate(img,size=(64,96))
    elif stage == 2:
        return interpolate(img,size=(128,192))
    elif stage == 3:
        return interpolate(img,size=(256,384))
    elif stage >= 4:
        return img


def fix_img(img:torch.Tensor,mode="tanh"):
    if mode == "tanh":
        img += 1 
        img /= 2 

    return img  


def prep_img(img:torch.Tensor,final_dim:int=400):
    
    #Convert to float16 if not done so
    #print(f"PREimg max is {torch.max(img):.3f}\tmin is {torch.min(img):.3f}")  
    img         = img.half()

    #Convert between [-1,1]
    img         = img*MULT - 1
    #print(f"POSTimg max is {torch.max(img):.3f}\tmin is {torch.min(img):.3f}")  
    
    #Dummy check
    assert torch.max(img) <= 1 and torch.min(img) >= -1

    AVGS.append(img.mean())
    return img


#This function is used to create the dataset used for training. It loads "source_file" which is a file provided 
#by Google with urls to images for download. There are 9 total, each containing 1 million URLS. 1 should suffice
# Download here : https://storage.googleapis.com/cvdf-datasets/oid/open-images-dataset-train0.tsv 
# Just pass the file path as "source_file" and it will download and pre-process the imgs
# ***MAKE SURE TO SET "save_loc" (located at top of file) to your dataset path
def online_grab(source_file:str,start_index:int=0,final_dim:int=400,history_fname:str="C:/data/images/download_history.txt"):
    dataset             = open(source_file,"r").readlines()
    saved               = 1
    t_b                 = 0 
    t_saved_b           = 0

    #Get file downoad history
    try:
        history:dict        = json.loads(open(history_fname,'r').read())
    except FileNotFoundError:
        print(f"\thistory not yet created")
        history             = dict()
    #already             = set(save_loc + l for l in os.listdir(save_loc))   

    xfrms               = transforms.Compose([transforms.PILToTensor(),transforms.Lambda(lambd=crop_to_ar)])

    for i,line in enumerate(dataset[1+start_index:]):
        url         = line.split("\t")[0]
        img_path    = url.split("/")[-1].split(".")[0] + ".pytensor"
        n_bytes     = int(line.split("\t")[1])
        t_b         += n_bytes    
        img_name    = save_loc+img_path

        if img_name in history:
            print(f"\tskipping\t{img_path[:7]}...")
            continue
        else:
            history[img_name]   = n_bytes
        
        try:
            urllib.request.urlretrieve(url,"test_img")
            try:
                img = Image.open("test_img")
            except Image.DecompressionBombError:
                print(f"\tbomb!")
                continue
        except urllib.error.HTTPError:
            print(f"\terr")
            continue
        except urllib.error.ContentTooShortError:
            print(f"\terr")
            continue
        except urllib.error.URLError:
            print(f"\terr")
            continue
        except TimeoutError:
            print(f"\terr")
            continue
        except ssl.SSLWantReadError:
            print(f"\terr")
            continue

        if img.height > 600:

            #Tranform and prep
            tensor  = xfrms(img)
            tensor  = prep_img(tensor)

            
            #Check images
            #plt.imshow(numpy.transpose(((tensor+1)/MULT).int().numpy(),[1,2,0]))
            #plt.show()

            #Convert grayscales 
            if tensor.shape[0] == 1:
                tensor  = torch.cat([tensor,tensor,tensor]).half()
                qual    = ' (Grayscale)'
            elif not tensor.shape[0] == 3:
                print(f"stange shape: {tensor.shape}")
                continue
            else:
                qual    = ''

                
            print(f"\tsaving\t{img_path[:8]}...\tshape:{tensor.shape}{qual}")
            torch.save(tensor,img_name)
            saved += 1 
            t_saved_b += n_bytes

        #training_data.append((n_bytes,was_saved))


        if (i % 100 == 0 and saved > 1):
            print(f"\tchecked {i} imgs, saved {saved}\tavg n_bytes: {t_b/(i+1):.1f} avg saved n_bytes: {t_saved_b/saved:.1f}")

        if (i%20 == 0):
            with open(history_fname,'w') as file:
                print(f"\trewriting history")
                file.write(json.dumps(history))


def local_grab():
    saved               = 1
    t_b                 = 0 
    t_saved_b           = 0
    training_data       = []

    already             = set(save_loc + l for l in os.listdir(save_loc))

    for i,img_path in enumerate(os.listdir("C:/gitrepos/train/train2017")):

        path        = "C:/gitrepos/train/train2017/" + img_path

        img_name    = save_loc+img_path

        if img_name in already:
            continue
        


        img = Image.open(path)


        was_saved   = False 
        if abs((img.width / img.height)-1.5) < .2 and img.width > 500:
            #print(f"saving {img_name}: {img.width}x{img.height}")
            img.save(img_name)
            saved += 1 
            was_saved   = True 


        training_data.append((n_bytes,was_saved))


        if (i % 200 == 0) and (not i == 0):
            print(f"\tchecked {i} imgs, saved {saved}\tavg n_bytes: {t_b/i:.1f} avg saved n_bytes: {t_saved_b/saved:.1f}")
  
        
def fix_local():
    root    = "F:/images/converted_tensors/" 

    for i,file in enumerate(os.listdir(root)):
        fname   = root + file


        #Load tensor 
        tensor  = torch.load(fname) / CORRECTOR
        tensor  = tensor.type(torch.float16)
        torch.save(tensor,fname) 

        if i % 1000 == 0:
            print(f"converted {i} tensors")


# EXAMPLE USAGE (i renamed the google source files to data0.tsv and data11.tsv)
for file in ["C:/data/images/imgs0.tsv"]:
    online_grab(file)

