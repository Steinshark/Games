from matplotlib import pyplot as plt 
import os 
import numpy 
import random 
from networks import AudioDiscriminator2,AudioDiscriminator3
from torch.utils.data import DataLoader,Dataset
import torch 
from utilities import weights_initD
import json 
import time 

_DEV        = torch.device('cuda')
_KERNELS    = [3,5,7,11,9,7,5,3]
_PADDINGS   = [int(k/2) for k in _KERNELS]
_OUTSIZE    = (1,int(529200/21))

_D          = AudioDiscriminator2(  channels=[_OUTSIZE[0],32,64,128,256,256,512,1],
                                    kernels=[3,5,7,11,9,7,5],mp_kernels=list([5,7,5,4,4,3,3,3]),
                                    paddings=_PADDINGS,
                                    device=torch.device('cuda'),
                                    final_layer=5,
                                    verbose=False,)
class classDataSet(Dataset):
    def __init__(self,fnames_good,fnames_bad):
        
        self.data = {"x":[],"y":[]}

        #Get good files 
        for fname_g in fnames_good:
            arr = numpy.load(fname_g)[0]
            arr = numpy.array([arr])
            arr = torch.from_numpy(arr).type(torch.float)

            if not arr.shape == (1,176400):
                print(f"bad len {fname_g} : {arr.shape}")

            self.data['x'].append(arr)
            self.data['y'].append(1)

        #Get bad files 
        for fname_b in fnames_bad:
            arr = numpy.load(fname_b)[0]
            arr = numpy.array([arr])
            arr = torch.from_numpy(arr)

            self.data['x'].append(arr)
            self.data['y'].append(0)
        
    def __getitem__(self, index):
        return self.data['x'][index],self.data['y'][index]
    
    def __len__(self):
        return len(self.data['x'])


def parse_files():

    if os.path.exists("data.txt"):
        data = open("data.txt").read()
        data = json.loads(data)
        good_files = data['good']
        bad_files  = data['bad']

        print(f"loaded:\n\tgood:{len(good_files)}\n\tbad{len(bad_files)}")
    else:
        good_files = []
        bad_files = []

    #create bad and good bins  
    root = "C:/data/music/dataset/LOFI_sf5_t20_c1_redo2"
    files = [os.path.join(root,f) for f in os.listdir(root)]
    random.shuffle(files)


    plt.ion()
    plt.show()

    for file in files:
        if file in good_files or file in bad_files:
            print("skip")
        arr = numpy.load(file)[0][::-1]
        plt.plot(arr,color='red')
        plt.title(f"{file.split('/')[-1]}")
        plt.draw()
        plt.pause(.001)
        verdict = input('verdict: ')

        while not verdict in ['g',"b","quit","l",'p']:
            verdict = input("verdict (again): ")
        
        if verdict == 'g':
            if not file in good_files:
                good_files.append(file)
        elif verdict == "b":
            if not file in bad_files:
                bad_files.append(file)
        elif verdict == 'p':
            plt.cla()
            continue
        elif verdict == 'l':
            from dataset import upscale,reconstruct
            arr = upscale(numpy.array([arr[::-1],arr[::-1]]),5)
            reconstruct(arr,"SNUP.wav")
            verdict = input("verdict: ")

        elif verdict == 'quit':
            dump = r"C:/gitrepos/projects/ml/music/data.txt"
            good = json.dumps({"good":good_files,"bad":bad_files})
            with open(dump,"w") as f:
                f.write(good)
            print(f"saved:\n\tgood:{len(good_files)}\n\tbad{len(bad_files)}")
            exit()
        

        plt.cla()
    

def audit(fnames,threshold=.98):

    #Load the latest state_dict 
    _D.load_state_dict(torch.load("class_models/latest"))

    good_files  = [] 

    #Load only scorers above threshold
    for fname in fnames:
        arr_as_tensor   = torch.from_numpy(numpy.load(fname)).to(torch.device('cuda')).type(torch.float32)
        if fname == fnames[0]:
            print(f"returning tensors size {arr_as_tensor.shape}")
        if _D.forward(arr_as_tensor.view((1,1,-1))).item() > threshold:
            good_files.append(arr_as_tensor.to(torch.device('cpu')))
    
    return good_files


def load_state(fname="class_models/latest"):
    _D.load_state_dict(torch.load(fname))


def check_tensor(tensor,threshold=.98):
    return _D.forward(tensor.view(1,1,-1).to(torch.device('cuda'))).item() > threshold
        

def train():
    torch.backends.cudnn.benchmark = True
    outsize = (1,int(529200/3))

    fnames      = json.loads(open("data.txt","r").read())
    i_train_good        = 1302
    i_train_bad         = 1302
    validation_size     = 100
    save_thresh         = .08


    print(f"{len(fnames['good'])} good\n{len(fnames['bad'])} bad")
    random.shuffle(fnames['good'])
    random.shuffle(fnames['bad'])
    
    good_train  = fnames['good'][:i_train_good]
    bad_train   = fnames["bad"][:i_train_bad]

    good_test   = fnames['good'][i_train_good:i_train_good+validation_size]
    bad_test    = fnames['bad'][i_train_bad:i_train_bad+validation_size]


    #Set verbose for hp tuning 
    verbose = True 


    eps     = 100
    bs      = 3
    lr      = .0001
    # #skip = [.000001]
    # lrs = [.0001,.000005]
    # bss  = [32,64,128]
    # optims = [torch.optim.SGD,torch.optim.Adam,torch.optim.AdamW,torch.optim.RMSprop]
    # fig, axs = plt.subplots(nrows=len(lrs),ncols=len(bss))
    # for y,bs in enumerate(bss):
    #     for x,lr in enumerate(lrs):
            
            #print(f"Testing bs {bs}, lr {lr}")
            # for optim in optims:
                # if lr in skip or bs in skip:
                #     gl_losses= [.5]*eps
                #     gl_validations = [.5]*eps 
                #     continue
    fake    = torch.zeros(size=(1,1,int(529200/3))).to(_DEV)
    print(f"D out {_D.forward(fake).shape}")
    _D.apply(weights_initD)
    gl_losses = [] 
    gl_validations = []
    loss_fn     = torch.nn.BCELoss()
    optim = torch.optim.SGD(_D.parameters(),lr=lr,weight_decay=lr/10,momentum=.9)

    dataset = classDataSet(good_train,bad_train)
    dataloader = DataLoader(dataset,bs,shuffle=True)

    testset = classDataSet(good_test,bad_test)
    testloader = DataLoader(testset,64,shuffle=True)

    for ep in range(eps):
        losses = [] 
        validations = []
        t0 = time.time()
        for i,data in enumerate(dataloader):

            #Clear grad 
            for p in _D.parameters():
                p.grad = None 
            
            x_set   = data[0].to(torch.device("cuda")).type(torch.float)
            y_set   = torch.reshape(data[1].to(torch.device("cuda")),shape=(len(data[1]),1)).type(torch.float)
            
            preds   = _D.forward(x_set)

            loss    = loss_fn(preds,y_set)
            loss_num    = loss.item()
            losses.append(loss_num)
            loss.backward()
            optim.step()


            #Validate 
            with torch.no_grad():
                data2       = iter(testloader).__next__()
                x_test      = data2[0].to(torch.device('cuda')).type(torch.float)
                y_test      = torch.reshape(data2[1].to(torch.device("cuda")),shape=(len(data2[1]),1)).type(torch.float)
                preds2      = _D.forward(x_test)
                validation  = loss_fn(preds2,y_test)
                validations.append(validation.item())

        #Update losses 
        gl_losses.append(sum(losses) / len(losses))
        gl_validations.append(sum(validations) / len(validations))

        if gl_validations[-1] < save_thresh:
            if not os.path.isdir("class_models"):
                os.mkdir("class_models")
            torch.save(_D.state_dict(),os.path.join("class_models",f"model_{gl_validations[-1]:.4f}"))

            print(f"reached acc of {save_thresh} after {ep} epochs")



        if verbose:
            print(f"EPOCH {ep}  \t- loss: {gl_losses[-1]:.4f}\t- vld't: {gl_validations[-1]:.4f}\t- time: {(time.time()-t0):.2f}s")

    plt.plot(gl_losses,label=f"x_train_{str(optim).split('(')[0]}",color='darkorange')
    plt.plot(gl_validations,label=f"x_test_{str(optim).split('(')[0]}",color='dodgerblue')
    #plt.set_title(f"BS: {bs} LR: {lr}")
    plt.legend()
    plt.show()

    if not os.path.isdir("class_models"):
        os.mkdir("class_models")
    torch.save(_D.state_dict(),os.path.join("class_models",f"CLASS_lm_state_dict"))


def get_tests():
    dev = torch.device("cuda")
    outsize = (1,int(529200/3))
    state_dict = torch.load("class_models/model_0.0532")
    
    _D.load_state_dict(state_dict)

    root = "C:/data/music/dataset/LOFI_sf5_t20_c1_peak1"
    files = [os.path.join(root,f) for f in os.listdir(root)]
    random.shuffle(files)

    files = files[:1000]
    dataset = classDataSet(files,[])
    print(f"dataset is len {dataset.__len__()}")
    dataloader = DataLoader(dataset,1)

    outs = [] 
    plt.ion()
    plt.show()
    nums = []
    with torch.no_grad():
        for data in dataloader:
            arr: torch.Tensor
            arr = data[0].to(dev)
            pred    = _D.forward(arr)
            plt.plot(arr[0].cpu().numpy()[0])
            plt.draw()
            plt.pause(.001)
            input(f"Score was {pred.item()}")
            plt.cla()

            #nums.append(pred.item())
            #outs.append((pred.item(),arr[0].cpu().numpy()))
        
    nums = sorted(nums)
    plt.plot(nums)
    plt.show()


    

    good_thresh         = float(input("Good Thresh:"))
    bad_thresh          = float(input("Bad Thresh:"))

    goods = 0 
    bads = 0 
    for score,arr in outs:
        if score > good_thresh:
            goods += 1
        elif score < bad_thresh:
            bads += 1
    print(f"{goods} goods\n{bads} bads\nGOODS")
    
    

     
    input("GOOD:")
    plt.ion()
    plt.show()

    for item in outs:
        if item[0] > good_thresh:
            plt.plot(item[1][0])
            plt.draw()
            plt.pause(.001)
            input()
            plt.cla()
    print("Bad:")
    for item in outs:
        if item[0] < bad_thresh:
            plt.plot(item[1][0])
            plt.draw()
            plt.pause(.001)
            input()
            plt.cla()


def divide_dataset(load_path,store_path,threshold=.95):
    
    load_state(fname="class_models/model_0.0532")
    size        = len(os.listdir(load_path))
    saved       = 0

    #Ensure store path exists
    if not os.path.exists(store_path):
        os.mkdir(store_path)

    with torch.no_grad():
        for i, fname in enumerate([os.path.join(load_path,f) for f in os.listdir(load_path)]):
            
            #Check for duplicate 
            if os.path.exists(fname.replace(load_path,store_path)):
                continue 
                
            #Load array
            arr_og = numpy.load(fname)[0] 
            arr = numpy.array([arr_og])
            arr = torch.from_numpy(arr).type(torch.float)

            #Check array 
            if _D.forward(arr.to(_DEV).view(1,1,-1)).item() > threshold:
                numpy.save(fname.replace(load_path,store_path),arr_og)
                saved += 1
            
            #Telemetry
            if i % 1000 == 0:
                print(f"Checked {i}/{size} - saved {saved}")






if __name__ == "__main__":
    import sys 
    if sys.argv[1] == "-p":
        parse_files()
    elif sys.argv[1] == "-t":
        train()
    elif sys.argv[1] == '-s':
        get_tests()
    
    elif sys.argv[1] == '-d':
        divide_dataset("C:/data/music/dataset/LOFI_sf5_t20_c1_peak1","C:/data/music/dataset/LOFI_sf5_t20_peak1_thrsh.9")

        