import urllib.request
from PIL import Image
import urllib.error
import torch 
import random 
from  torch.utils.data import Dataset,DataLoader
import os 

save_loc  	= r"//FILESERVER/S Drive/Data/images/all/"
def online_grab():
    dataset     = open("data0.tsv","r").readlines()
    save_loc  	= r"//FILESERVER/S Drive/Data/images/all/"
    saved               = 1
    t_b                 = 0 
    t_saved_b           = 0
    training_data       = []


    false_negatives     = 0
    correct_negatives   = 0 
    totals              = 0  

    effectiveness       = 0
    danger              = 1
    already             = set(save_loc + l for l in os.listdir(save_loc))

    for i,line in enumerate(dataset[1:]):

        url         = line.split("\t")[0]
        img_format  = url.split(".")[-1] 
        img_path    = url.split("/")[-1]
        bytes       = int(line.split("\t")[1])
        t_b         += bytes    
        img_name    = save_loc+img_path

        if img_name in already:
            continue
        

        try:
            urllib.request.urlretrieve(url,"test_img")
            img = Image.open("test_img")
        except urllib.error.HTTPError:
            continue

        was_saved   = False 
        if abs((img.width / img.height)-1.5) < .2 and img.width > 700:
            #print(f"saving {img_name}: {img.width}x{img.height}")
            img.save(img_name)
            saved += 1 
            t_saved_b += bytes
            was_saved   = True 


        training_data.append((bytes,was_saved))


        if (i % 1000 == 0):
            print(f"\tchecked {i} imgs, saved {saved}\tavg bytes: {t_b/(i+1):.1f} avg saved bytes: {t_saved_b/saved:.1f}")


def local_grab():
    save_loc  	= r"//FILESERVER/S Drive/Data/images/all/"
    saved               = 1
    t_b                 = 0 
    t_saved_b           = 0
    training_data       = []


    false_negatives     = 0
    correct_negatives   = 0 
    totals              = 0  

    effectiveness       = 0
    danger              = 1
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
        


online_grab()