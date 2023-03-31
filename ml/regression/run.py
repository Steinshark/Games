import torch 
from torch.nn import Sequential,Linear,ReLU,MSELoss
from torch.optim import Adam, SGD 
from torch.utils.data import DataLoader, Dataset
import pandas as pd 
from sklearn.model_selection import train_test_split

#Load data 
datafile            = pd.read_csv("FuelData.csv")
#YEAR,MAKE,MODEL,VEHICLE CLASS,ENGINE SIZE,CYLINDERS,TRANSMISSION,FUEL,FUEL CONSUMPTION,HWY (L/100 km),COMB (L/100 km),COMB (mpg),EMISSIONS
for col in ["MAKE","MODEL","VEHICLE CLASS","TRANSMISSION","FUEL"]:
    datafile[col]   = pd.factorize(datafile[col])[0]


train_x,train_y,test_x,test_y          = train_test_split( datafile.drop("HWY",axis=1).drop("COMB_L",axis=1).drop("COMB_G",axis=1),
                                        datafile["COMB_G"],
                                        train_size=.8)


class RDataset(Dataset):

    def __init__(self,x:pd.DataFrame,y):
        self.data   = {"x":[],"y":[]}
        for x_i,y_i in zip(x.iterrows(),y.iterrows()):
            self.data['x'].append(  torch.tensor(list(x_i[1])))
            self.data['y'].append(  torch.tensor(list(y_i[1])))
        
    def __getitem__(self, i):
        return self.data['x'][i], self.data['y'][i]
    
    def __len__(self):
        return len(self.data['x'])
    

#HYPERPARAMETERS 
bs              = 32 


dataset         = RDataset(train_x,train_y)
dataloader      = DataLoader(dataset=dataset,batch_size=bs,shuffle=True)

model           = 








