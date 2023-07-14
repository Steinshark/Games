from torchvision.datasets import CIFAR10, CIFAR100 
from torchvision import transforms
from torch.utils.data import DataLoader
import torch 
from matplotlib import pyplot as plt 


#ok that sounds great


def accuracy(preds,labels):
    _,preds     = torch.max(preds,dim=1)
    return torch.sum((preds == labels)).item() / len(preds)

def run_train(bs,n_ch,mom,lr,nest,p,var):
    title   = f"bs:{bs}-ch:{n_ch}-mom:{mom}-lr:{lr}-n:{nest}-p:{p}-var:{var}"
    accuracies      = []
    model           = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=3,out_channels=n_ch,kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(n_ch),

                torch.nn.Conv2d(in_channels=n_ch,out_channels=n_ch*2,kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(n_ch*2),

                torch.nn.Conv2d(in_channels=n_ch*2,out_channels=n_ch*4,kernel_size=5,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(n_ch*4),
                torch.nn.MaxPool2d(2),

                torch.nn.Conv2d(in_channels=n_ch*4,out_channels=n_ch*8,kernel_size=5,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(n_ch*8),
                torch.nn.MaxPool2d(2),

                torch.nn.Flatten(),
                torch.nn.Linear(n_ch*8*36,512),
                torch.nn.Dropout(p*2),
                torch.nn.ReLU(),

                torch.nn.Linear(512,128),
                torch.nn.Dropout(p),
                torch.nn.ReLU(),

                torch.nn.Linear(128,10),
                torch.nn.Sigmoid(),
                #torch.nn.Softmax(dim=0)
    )
    optimizer       = torch.optim.SGD(model.parameters(),lr=lr,momentum=mom,nesterov=nest)
    loss            = torch.nn.CrossEntropyLoss()

    shuffle         = True 
    tforms          = transforms.Compose([transforms.ToTensor(),transforms.Normalize((.48,.45,40),(var,var,var))])
    data_train      = CIFAR10("/home/steinshark/Downloads/datasets/cifar10",train=True,download=True,transform=tforms)
    data_test       = CIFAR10("/home/steinshark/Downloads/datasets/cifar10",train=False,download=True,transform=tforms)
    load_train      = DataLoader(data_train,batch_size=bs,shuffle=shuffle)
    load_test       = DataLoader(data_test,batch_size=bs,shuffle=False)
    classes         = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #16,2,.9,

    print(f"SERIES {title}")
    for epoch in range(32):
        train_losses    = []
        accr            = [] 
        for i,batch in enumerate(load_train):

            #Clear grad
            for p in model.parameters():
                p.grad  = None 

            img,img_class   = batch 
            output_vals     = model(img)
            err             = loss(output_vals,img_class)
            train_losses.append(err.mean().item())
            err.backward()

            accr.append(accuracy(output_vals,img_class)*100)

            optimizer.step()
        
        print(f"\tTrain accur: {sum(accr)/len(accr):.2f}\t",end="")
        
        accs            = [] 
        with torch.no_grad():
            for j, batch in enumerate(load_test):
                img,img_class   = batch 
                outs            = model(img)

                accs.append(accuracy(outs,img_class)*100)
            print(f"\tTest accur: {sum(accs)/len(accs):.2f}%")
        accuracies.append(sum(accs)/len(accs))
    print(f"\n")
    return title,accuracies
########################################################
#                   SETTINGS
########################################################



series   = {}
for bs in [16]:
    for n_ch in [2]:
        for mom in [.75,.9]:
            for lr in [.001]:
                for nest in [True,False]:
                    for p in [.1,.25]:
                            for var in [.1,.25,.5]:
                                key,val         = run_train(bs,n_ch,mom,lr,nest,p,var)
                                series[key]     = val 


for i,key in enumerate(series):
    plt.plot(series[key],label=f"{key}")
plt.legend()
plt.show()



