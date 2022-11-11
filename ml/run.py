from net import FullyConnectedNetwork
import random
from torch import nn as nn 
import torch 

if __name__ == "__main__":
    dataset_in = open("/home/steinshark/Downloads/cardata.csv").readlines()
    datalist = []
    
    classes = {4:[],5:[],6:[],7:[]}
    min_val = 10000000
    for line in dataset_in[1:]:
        data = line.split(",")

        # Convert to ints 
        for i in [1,2,3]:
            data[i] = int(data[i])
        #Find last year
        if data[1] < min_val:
            min_val = data[1]
        #Check for distinct classes
        for i in [4,5,6,7]:
            if not data[i] in classes[i]:
                classes[i].append(data[i])
        datalist.append(data)
    dataset = {"x":[],"y":[]}

    #Build our dataset 
    for line in datalist:
        y = line[2]
        x = [0 for i in range(6+sum([len(l) for l in classes.values()]))] 
        #Name will be 0
        x[0] = 0#line[0]
        x[1] = line[1] - min_val
        x[2] = line[2]
        x[3] = line[3]
        x[4+classes[4].index(line[4])] = 1
        x[4+len(classes[4])+classes[5].index(line[5])] = 1
        x[4+len(classes[4])+len(classes[5])+classes[6].index(line[6])] = 1
        x[4+len(classes[4])+len(classes[5])+len(classes[6])+classes[7].index(line[7])] = 1
        dataset["x"].append(x)
        dataset["y"].append(y)
    


    indices = []
    not_indices = [i for i in range(len(dataset["x"]))]
    r_indices = [random.randint(0,len(dataset["x"])-1) for i in range(int(.1053*len(dataset["x"])))]  
    indices = []
    for i in r_indices:
        if i not in indices:
            indices.append(i)
            not_indices.remove(i)
    
    train_data = {"x": torch.tensor([dataset["x"][i] for i in not_indices],dtype=torch.float), "y":torch.tensor([[dataset["y"][i]] for i in not_indices],dtype=torch.float)}
    test_data = {"x": torch.tensor([dataset["x"][i] for i in indices],dtype=torch.float), "y":torch.tensor([[dataset["y"][i]] for i in indices],dtype=torch.float)}

    
    batches = [1,16,128,512]
    callbacks = [3,5]
    lrates = [.001,.0001,.00001,.000001]
    losses = {}
    for b in batches:
        for l in lrates:
            for c in callbacks:
                print(f"testing config: {(b,l,c)}")
                network = FullyConnectedNetwork(len(dataset["x"][0]),1,loss_fn= nn.MSELoss,optimizer_fn=torch.optim.Adam ,lr=l,architecture=[128,64,8])
                loss = network.train(train_data["x"],train_data["y"],epochs=1000,verbose=True,show_steps=10,batch_size=b,show_graph=False,callback_on=c)
                res = network.forward(test_data["x"])
                diff = network.loss(res,test_data["y"])
                losses[str((b,l,c))] = torch.sqrt(diff).item() 
    print(f"models had rMSE performance of")
    import pprint 
    pprint.pp(losses)
