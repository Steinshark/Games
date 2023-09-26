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
socket.setdefaulttimeout(1)

save_loc  	= r"F:/images/converted_tensors/"
MULT        = 2/255
CORRECTOR   = 0.7142857313156128


#This function is used to convert the tensor from 3*y*x tensor to 3*512*768 for use in our network training
def crop_to_ar(img:torch.Tensor,ar=1):

    
    #Check img dims compatable 
    if not len(img.shape) == 3:
        raise ValueError(f"bad shape {img.shape} must be 2d img")
    
    img_x   = img.shape[2]
    img_y   = img.shape[1]

    #Fix AR
    if not (img_x / img_y == 1.5):
        ar          = img_x / img_y
        removed_x   = 'l'
        removed_y   = 't'
        while not (ar == 1.5):
            
            #Remove a side 
            if ar > 1.5:
                if removed_x == 'l':
                    img     = img[:,:,1:]
                    removed_x  = 'r'
                else:
                    img     = img[:,:,:-1]
                    removed_x  = 'l'
            elif ar < 1.5:
                if removed_y == 't':
                    img     = img[:,:-1,:]
                    removed_y   = 'b'
                else:
                    img     = img[:,1:,:]
                    removed_y   = 't'
            img_x   = img.shape[2]
            img_y   = img.shape[1]
            ar          = img_x / img_y 
            #print(f"ar={ar}\t{img.shape}")
    #INFO


    img     = img.unsqueeze_(dim=0).type(torch.float)
    img     *= MULT
    img     -= 1  
    img     = interpolate(img,size=(512,768))[0]
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

#This function is used to create the dataset used for training. It loads "source_file" which is a file provided 
#by Google with urls to images for download. There are 9 total, each containing 1 million URLS. 1 should suffice
# Download here : https://storage.googleapis.com/cvdf-datasets/oid/open-images-dataset-train0.tsv 
# Just pass the file path as "source_file" and it will download and pre-process the imgs
# ***MAKE SURE TO SET "save_loc" (located at top of file) to your dataset path
def online_grab(source_file):
    dataset     = open(source_file,"r").readlines()
    saved               = 1
    t_b                 = 0 
    t_saved_b           = 0
   
    already             = set(save_loc + l for l in os.listdir(save_loc))

    xfrms               = transforms.Compose([transforms.PILToTensor(),transforms.Lambda(lambd=crop_to_ar),transforms.Normalize(mean=0,std=1.4)])

    for i,line in enumerate(dataset[1:]):
        if i < 134_500:
            continue
        url         = line.split("\t")[0]
        img_path    = url.split("/")[-1].split(".")[0] + ".pytensor"
        bytes       = int(line.split("\t")[1])
        t_b         += bytes    
        img_name    = save_loc+img_path

        if img_name in already:
            continue
        
        try:
            urllib.request.urlretrieve(url,"test_img")
            try:
                img = Image.open("test_img")
            except Image.DecompressionBombError:
                #print(f"bomb")
                continue
        except urllib.error.HTTPError:
            #print(f"err")
            continue
        except urllib.error.ContentTooShortError:
            #print(f"err")
            continue
        except TimeoutError:
            pass

        if abs((img.width / img.height)-1.5) < .4 and img.width > 700:
            tensor  = (xfrms(img) / CORRECTOR).type(torch.float16)
            #print(f"saving {img_name}: {tensor.shape}")
            if tensor.shape[0] == 3:
                torch.save(tensor,img_name)
            #Show interpolated img 
            converted   = fix_img((tensor.float()).unsqueeze_(dim=0))
            #print(f"attempting {converted.shape}")
            #new_tensor  = downsample(converted,3)[0]
            #transforms.ToPILImage()(new_tensor).show()
            #img.save(img_name)
            saved += 1 
            t_saved_b += bytes

        #training_data.append((bytes,was_saved))


        if (i % 100 == 0 and saved > 1):
            print(f"\tchecked {i} imgs, saved {saved}\tavg bytes: {t_b/(i+1):.1f} avg saved bytes: {t_saved_b/saved:.1f}")
            print(f"\tsaved shape {tensor.shape}")


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


        training_data.append((bytes,was_saved))


        if (i % 200 == 0) and (not i == 0):
            print(f"\tchecked {i} imgs, saved {saved}\tavg bytes: {t_b/i:.1f} avg saved bytes: {t_saved_b/saved:.1f}")
  
        
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
for file in ["F:/source/data0.tsv","F:/source/data1.tsv"]:
    online_grab(file)

