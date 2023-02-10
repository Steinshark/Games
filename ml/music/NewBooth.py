
import numpy 
from matplotlib import pyplot as plt 
from scipy.fftpack import fft, fftfreq,ifft
import cmath 
import torch 
from torch.nn import Sequential,Tanh,LSTM
from torch.utils.data import DataLoader,Dataset
import os 
import random 
import pprint

CONST_COMPLX_TYPE       = torch.float32

class LSTMDataset(Dataset):
    def __init__(self,data):
        self.data = data
    
    def __len__(self):
        return len(self.data["x"])
    
    def __getitem__(self, index):
        return self.data['x'][index],self.data['y'][index]

def complex_mse_loss(input, target,exp=2):
    """
    Computes the mean squared error between the input and target complex tensors.
    
    Parameters:
    input (torch.Tensor): Complex tensor of shape (N, ...)
    target (torch.Tensor): Complex tensor of shape (N, ...)
    
    Returns:
    loss (torch.Tensor): Scalar loss value.
    """
    diff = input - target
    diff_real = diff[..., 0] # Real part
    diff_imag = diff[..., 1] # Imaginary part
    loss = torch.mean(torch.pow(diff_real, exp) + torch.pow(diff_imag, exp))
    return loss

class modelG(torch.nn.Module):

    def __init__(self,bs,window,memory_size,dev=torch.device("cuda")):
        super(modelG,self).__init__()

        self.L1         = LSTM(input_size=window,hidden_size=window,num_layers=2,batch_first=True).type(CONST_COMPLX_TYPE).to(dev)
        #self.act        = Tanh().to(dev).type(CONST_COMPLX_TYPE)
    
        
    def forward(self,x):

        input(x.shape)
        lstm_outs,(h,c)     = self.L1(x)
        predicted_val       = h[-1]


        #predicted_activated = self.act(predicted_val)

        return predicted_val  



def get_fft(signal,sample_rate=44100/5,freq_max=8000):

    n       = len(signal)
    n_half  = int(n/2)

    fft_y   = fft(signal)
    fftreal = fft_y
    fft_y   = [numpy.real(power) for power in fft_y[:n_half]]
    
   
    fft_x   = fftfreq(n,d=1/sample_rate)
    fft_x   = [abs(freq) for freq in fft_x[:n_half]]

    return fft_x,fft_y,fftreal


def difft(x,thresh=0):
    N = x.shape[0]
    n = numpy.arange(N)
    k = n.reshape((N, 1))
    M = numpy.exp(-2j * numpy.pi * k * n / N)
    return numpy.dot(M, x)


def fft_to_waveform(freqs,powers,duration,sample_rate=int(44100/5),thresh=0,original=None):
    pi              = 3.14159265358979
    time_series     = numpy.linspace(0,duration,int(sample_rate*duration))
    waveform        = numpy.linspace(0,0,int(sample_rate*duration))
    for i,power in enumerate(powers): 
        #wave = powers[i]*numpy.cos(2*pi*freq*time_series)
        wave = power * numpy.exp(2j * pi * i * len(powers))
        waveform = waveform + wave

    waveform = numpy.real(waveform)
    plt.plot(time_series,original,label="Original")
    plt.plot(time_series,waveform/len(freqs),label="Reconstructed")
    plt.legend()
    plt.show()
    print("\n\n\n\n")
    return waveform


def reconstruct_time_series(data,thresh=1,qual=50):

    if qual == "all":
        inds = range(data.shape[0])
    elif qual == "rand":
        import random
        inds = random.sample(range(data.shape[0]),1024)
    else:
        inds = numpy.argpartition(data,-qual)[-qual:]
    # Create an empty array to store the reconstructed time series
    time_series = numpy.zeros(data.shape[0])
    # Loop over all frequency components
    for i in inds:
        # Multiply each frequency component by its corresponding amplitude
        time_series = time_series + data[i] * numpy.exp(2j * numpy.pi * i * numpy.arange(data.shape[0]) / data.shape[0])
    
    # Return the reconstructed time series
    return numpy.real(time_series)


def filter_powers(powers,thresh=2000):
    return [pow(numpy.maximum(p**4-thresh,0),.25) for p in powers]


def visualize():
    vector  = numpy.load("C:/data/music/dataset/LOFI_sf5_t20_c1_redo/0f03823a5c_0.npy")[0]
    ranges = range(10,100)
    samples = [0]*len(ranges)
    window = int(44100/(5*5))
    powers = [0] *len(ranges)
    freqs = [0] *len(ranges)

    fr_max = 16000
    for i in ranges:
        samples[ranges.index(i)] = vector[window*i:window*i+window]
        #plt.plot(samples[i])
        freqs[ranges.index(i)],powers[ranges.index(i)],ft_real = get_fft(samples[ranges.index(i)],freq_max=fr_max)


    thresh = 1000
    for i in ranges:
        #plt.plot(freqs[i],powers[i],label=f"{i} seconds")
        plt.plot(freqs[ranges.index(i)],[pow(numpy.maximum(p**4-thresh,0),.25) for p in powers[ranges.index(i)]],label=f"{i} seconds")
    
    #plt.plot([10+i for i in ifft(ft_real)])
    plt.legend()


def reconstruct():
    vector  = numpy.load("C:/data/music/dataset/LOFI_sf5_t20_c1_redo/0f03823a5c_0.npy")[0]

    window = int(44100/(5*441))
    new_audio = []
    real_audio = [] 
    end = int(len(vector)/window)
    ranges = range(0,end)

    for i in ranges:
        sample = vector[window*i:window*i+window]
        freq,power,raw = get_fft(sample)
        #power = filter_powers(power)

        #fft_to_freq_list(freq,power,.1,original=sample)    
        new_audio += list(reconstruct_time_series(raw,qual=20))
        real_audio += list(sample)
    from dataset import reconstruct,upscale
    arr = numpy.array([new_audio,new_audio])
    #print(f"max val in arr is {numpy.amax(arr[0])}")
    #arr[0] /= numpy.amax(arr[0])
    #arr[1] /= numpy.amax(arr[1])
    arr /= window
    arr = upscale(arr,5)
    plt.plot(arr[0][:int(44100/2)],label="TEST",color="r")
    reconstruct(arr,"FT_TEST.wav")

    arr = numpy.array([real_audio,real_audio])
    arr = upscale(arr,5)
    plt.plot(arr[0][:int(44100/2)],label="BASE",color="b")
    reconstruct(arr,"FT_BASE.wav")

    plt.legend()
    plt.show()


def LSTM_poc(bs,window,memory_size):
    vector  = numpy.load("C:/data/music/dataset/LOFI_sf5_t20_c1_redo/0f03823a5c_0.npy")[0]

    inputs = numpy.array([[vector[i*window:i*window+window] for i in range(memory_size)]])
    olds = inputs
    G       = torch.nn.LSTM(input_size=441,hidden_size=64)
    inputs  = torch.from_numpy(inputs).type(torch.float32)
    inp     = torch.nn.utils.rnn.pack_sequence(inputs)
    outp    =G(inp)
    print(f"in_shape: {inp.data.shape}")
    print(f"out_shape: {outp[0].shape}")



def prep_data(bs,chunk_size,lookback,fnames):

    data = {"x":[],"y":[]} 

    for file in fnames:
        arr     = numpy.load(file)[0]
        arr     = [arr[chunk_size*i:chunk_size*i+chunk_size] for i in range(int(arr)/chunk_size)]

        for i in range(len(arr) - lookback -1 ):
            data['x'].append(arr[i:i+lookback])
            data['y'].append(arr[i+lookback])
        
    return data



def train(data,model:torch.nn.Module,epochs=100,bs=32,lr=.0001,betas=(.9,.999),dev=torch.device('cuda')): 
    dataset     = LSTMDataset(data)
    dataloader  = DataLoader(dataset,batch_size=bs,shuffle=True,pin_memory=True)

    #Train stuff 
    optim       = torch.optim.Adam(model.parameters(),lr=lr,betas=betas,weight_decay=lr)
    loss_fn     = torch.nn.MSELoss()

    for ep in range(epochs):
        losses  = [] 

        for batch,data in enumerate(dataloader):

            x       = data[0].to(dev)
            y       = data[1].to(dev)

            pred    = model.forward(x)

            loss    = complex_mse_loss(pred,y,exp=1.5)
            losses.append(loss.item())

            loss.backward()

            optim.step()

        print(f"EPOCH:\t{ep} - loss: {(sum(losses)/len(losses)):.3f} - loss(r): {numpy.real(sum(losses)/len(losses)):.3f}")    




if __name__ == "__main__":
    sr      = int(44100/5) 
    window  = int(8820/60)
    memory  = 60
    dev     = torch.device('cpu')
    bs      = 128
    G       = modelG(bs,window,memory,dev=dev)

    #base = 'C:/data/music/dataset/LOFI_sf5_t20_c1_redo'
    #fnames = [os.path.join(base,f) for f in os.listdir(base)]
    #fnames = random.sample(fnames,5)
    #Prep data 

    LSTM_poc(1,441,5)