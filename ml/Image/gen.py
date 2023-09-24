import torch
import torchvision 
from torchvision import transforms
from torchvision.utils import make_grid
from torch.nn.functional import interpolate 
from torch.utils.data import Dataset,DataLoader
import time 
from models import * 
from PIL import Image
import os 
DEV     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device {DEV}")
import math 
MULT    = 2/255
#a comment 

def convert_to_0_1(input_tensor:torch.Tensor)->torch.Tensor:

    tensor_range    = torch.max(input_tensor) - torch.min(input_tensor)
    returner        = (input_tensor + abs(torch.min(input_tensor))) / tensor_range 
    #print(f"range is ({torch.min(returner)},{torch.max(returner)})")
    return returner

class preload_ds(Dataset):

    def __init__(self,fname_l,processor):
        self.data       = fname_l
        self.processor  = processor

    #Return tensor, fname
    def __getitem__(self,i):
        tensor  = self.processor(torch.load(self.data[i]))

        if not tensor.shape == (3,512,512+256):
            return torch.randn(size=(3,512,512+256),dtype=torch.float)
        try:
            return torch.load(self.data[i])
        except RuntimeError as e:
            try:
                return torch.load(self.data[i])
            except RuntimeError as e:
                return torch.randn(size=(3,512,512+256),dtype=torch.float)
    def __len__(self):
        return len(self.data)

#Crop everything to 1.5
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
    img     = img.unsqueeze_(dim=0).type(torch.float)
    img     *= MULT
    img     -= 1  
    img     = interpolate(img,size=(500,750))[0]
    return img

def load_dataset(bs=8,temp_limit=2000,tensor_lib="F:/images/converted_tensors"):
    tensors         = [] 

    #Build tensors locally or from network
    print(f"generating dataset")
    #files   = set(os.listdir(local_lib))
    i       = 0 
    saved   = 0 
    for file in os.listdir(tensor_lib):
        tensor  = torch.load(f"{tensor_lib}{file}")

        #tensors.append(tensor)
        i += 1 
        if i % 1000 == 0:
            print(f"\tloaded {i} - saved {saved}")
        if i >= temp_limit:
            break
    
    #Generate dataset
    dataset     = preload_ds(tensors)
    dataloader  = DataLoader(dataset,batch_size=bs)
    return dataloader


#This method is used to load the dataset that was downloaded using the "scratch.py" program. see scratch.py for details 
# To create and store the dataset
# Pass the root of the image dataset you downloaded in scratch.py as the "local_dataset_path" arg and it will create a 
# Dataset for you to use 
def load_locals(bs=8,processor=lambda x: x,local_dataset_path="F:/images/converted_tensors/"):
    dataset     = preload_ds([local_dataset_path+f for f in os.listdir(local_dataset_path)],processor=processor)
    return DataLoader(dataset,batch_size=bs,shuffle=True)

def convert_locals(bs=8):
    xfrms       = transforms.Compose([transforms.PILToTensor(),transforms.Lambda(lambd=crop_to_ar),transforms.Normalize(mean=0,std=1.4)]) 
    root        = r"//FILESERVER/S Drive/Data/images/all/"
    save        = "C:/data/images/converted_tensor/"
    names       = set(os.listdir(save))

    i       = 0 
    saved   = 0
    for img in os.listdir(root):
        img_dir     = root+img
        
        img_name    = img.split('.')[0]
        save_dir    = save+img_name

        if not img_name in names:
            tensor      = xfrms(Image.open(img_dir)) 
            torch.save(tensor,save_dir)
            saved += 1 
        
        i += 1 

        if i % 200 == 0:
            print(f"i={i}\nsaved {saved}")

def fix_tanh():
    img_lib     = r"//FILESERVER/S Drive/Data/images/"
    sav_lib     = r"//FILESERVER/S Drive/Data/converted_tensor/"
    xfrms       = transforms.Compose([transforms.PILToTensor(),transforms.Lambda(lambd=crop_to_ar),transforms.Normalize(mean=0,std=1.4)]) 
    dataset     = torchvision.datasets.ImageFolder(img_lib,transform=xfrms)

    i = 0 
    for tensor,fname in zip(dataset,os.listdir(r"//FILESERVER/S Drive/Data/images/all")):
        tensor = tensor[0]
        for ind_tensor in tensor:
            torch.save(tensor.type(torch.float16),f"{sav_lib}{fname.split('.')[0]}")
        i += 1 
        if i % 500 == 0:
            print(f"saved {i}")
    print(f"saved all tensors")

def fix_sigmoid():
    img_lib     = r"//FILESERVER/S Drive/Data/images/"
    sav_lib     = r"//FILESERVER/S Drive/Data/converted_tensor/"
    xfrms       = transforms.Compose([transforms.PILToTensor(),transforms.Lambda(lambd=crop_to_ar),transforms.Normalize(mean=0,std=1.4)]) 
    dataset     = torchvision.datasets.ImageFolder(img_lib,transform=xfrms)

    i = 0 
    for tensor,fname in zip(dataset,os.listdir(r"//FILESERVER/S Drive/Data/images/all")):
        tensor = tensor[0]
        for ind_tensor in tensor:
            torch.save(tensor.type(torch.float16),f"{sav_lib}{fname.split('.')[0]}")
        i += 1 
        if i % 500 == 0:
            print(f"saved {i}")
    print(f"saved all tensors")




#VARIABLES
bs                  = 8
update_batch        = 50
n_imgs              = 25 
display_n           = 50
n_row               = int(math.sqrt(n_imgs))

#MODELS 
model               = torchvision.models.googlenet(weights=torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1).to(DEV)
model.eval()
gen                 = generator_lg().to(DEV)
err_fn              = torch.nn.MSELoss()
optimizer           = torch.optim.Adam(gen.parameters(),lr=.0002,betas=(.5,.99))

#DATA
dataloader          = load_locals(bs=bs,processor=convert_to_0_1)

#STORAGE
# - SET THESE VARIABLES TO STORE YOUR MODEL PROGRESS AND SAMPLE IMAGES
model_save_root     = "C:/gitrepos/projects/ml/image/models/"
img_save_root       = "C:/gitrepos/projects/ml/image/tests"
def fix_img(img:torch.Tensor,mode="tanh"):
    if mode == "tanh":
        img += 1 
        img /= 2 

    return img  

for ep in range(20):
    print(f"\n\nEpoch {ep}\tbegin training on {len(dataloader)*bs} images")

    prev_imgs   = []
    losses      = [] 
    t_start     = time.time()

    for i,item in enumerate(dataloader):
        t0=time.time()
        if item is None:
            continue
        img             = item.to(DEV).type(torch.float)

        #print(f"max: {torch.max(img)} min: {torch.min(img)}")
        #print(f"img shape {img.shape}")
        optimizer.zero_grad()
        
        #Run through google filters
        with torch.no_grad():
            model(img)

        #Generate 
        generator_out   = gen.forward(model.pre_flatten)
        #input(f"out size is {generator_out.shape}")
        if i % int(display_n / n_imgs) == 0:
            prev_imgs.append(fix_img(generator_out[0].to(torch.device('cpu')),mode="tanh"))

        #Calc loss 

        loss            = err_fn(generator_out,img)
        loss.backward() 
        losses.append(float(loss.mean().item()))

        optimizer.step()

        del generator_out
        del img

        if i % update_batch == 0:
            print(f"\tbatch [{i}]\tloss={losses[-1]:.3f}\tavg loss={sum(losses)/len(losses):.3f}\tt={(time.time()-t0)/bs:.2f}s/img\tt tot={(time.time()-t_start):.2f}s\tn_imgs={int(bs*i)}")

        if i % display_n == 0 and i > 0:
            root    = f"{img_save_root}/ep{ep}/"
            if not os.path.exists(root):
                os.mkdir(root)

            grid    = make_grid(prev_imgs[-25:],nrow=n_row)
            display:Image     = transforms.ToPILImage()(grid)
            #display.show()
            display.save(root+f"test{i}.jpg")
            prev_imgs = [] 

    print(f"\tbatch [{i}]\tloss={losses[-1]:.3f}\tavg loss={sum(losses)/len(losses):.3f}\tt={(time.time()-t0)/bs:.2f}s/img\tt tot={(time.time()-t_start):.2f}s\tn_imgs={int(bs*i)}")
    save_loc     = f"{model_save_root}sm_ep_{ep}.model"
    torch.save(model.state_dict,save_loc)
    #Reload data
    del dataloader 
    dataloader  = load_locals(bs=bs)