from pytube import YouTube 
from pydub import AudioSegment
from pydub.utils import make_chunks
import os 
import sys 
from binascii import hexlify
from pydub.exceptions import CouldntDecodeError
import ctypes
import numpy 
#from networks import AudioGenerator,AudioDiscriminator
#import torch 
import binascii
import hashlib
import time 
from http.client import IncompleteRead
import math 
import random 
import torchaudio 
import torch 


#Inizialize directories properly
if "linux" in sys.platform:
    DOWNLOAD_PATH           = r"/media/steinshark/stor_sm/music/downloads"
    CHUNK_PATH              = r"/media/steinshark/stor_sm/music/chunked"
    DATASET_PATH            = r"/media/steinshark/stor_lg/music/dataset"
else:
    DOWNLOAD_PATH           = r"//FILESERVER/S Drive/Data/music/downloads"
    CHUNK_PATH              = r"//FILESERVER/S Drive/Data/music/chunked"
    DATASET_PATH            = r"//FILESERVER/S Drive/Data/music/dataset"

MINUTES                 = 0 

#Normalize audio levels 
def normalize_level(sound:AudioSegment,normalization_level):
    delta_dBFS  =   normalization_level - sound.dBFS
    return sound.apply_gain(delta_dBFS)

#Normalize to  [-peak,peak]
def normalize_peak(arr,peak=1):
    max_ampl    = numpy.amax(arr)
    if max_ampl == 0:
        raise ValueError
    arr         = arr * (peak/max_ampl) 
    return arr

#Returns a deterministic hash of a string
def hash_str(input_str:str):
    return str(hashlib.md5(input_str.encode()).hexdigest())

#Returns whether or not the file has been chunked
def is_chunked(fname,output_path):
    fhash           = hash_str(fname)[:10]

    #Check that file has not been chunked 

    fname_check     = os.path.join(output_path,f"{fhash}_0.wav")
    return os.path.exists(fname_check)

#Returns whether or not the file has been converted to numpy 
def is_processed(filename,category):
    audio_output_path       = os.path.join(DATASET_PATH,category)
    output_full_path        = os.path.join(audio_output_path,f"{filename.replace('.wav','.npy')}")

    return os.path.exists(output_full_path)

#Downloads 1 URL 
def download_link(url:str,output_path:str,index=None,total=None): 

    #Check if already exists 
    out_path_full   = os.path.join(output_path,f"{hash_str(url)[:10]}_final.mp4")
    if os.path.exists(out_path_full):
        print(f"video  done {url[:50]}")
        return
    #Grab video 
    yt_handle       = YouTube(url, use_oauth=True, allow_oauth_cache=True)
    print(f"yt handle is {yt_handle}")

    #Telemetry
    if not index is None:
        tel_msg         = f"downloading {url[:50]}\t[{index}/{total}]"
    else:
        tel_msg         = f"downloading {url[:50]}"
    print(tel_msg,end='',flush=True)
    try:
        #filepath        = yt_handle.streams.filter(progressive=True,only_audio=True).first().download(output_path=output_path)
        streams         = yt_handle.streams
        print(f"looking for streams")

        print(f"found streams\n{streams}")
        filepath    = ""

        #os.rename(filepath,out_path_full)
        print(f"\t- success")
    except IncompleteRead:
        print(f"\t -failed")

#Convert big-endian to little-endian
def big_to_little(hex,con=True):
    little_repr = bytearray.fromhex(hex)[::-1].hex()
    return int(little_repr,base=16) if con else little_repr

#Convert little-endian to big-endian
def little_to_big(hex,con=False):
    try:
        big_repr = bytearray.fromhex(hex)[::-1].hex()
    except TypeError:
        big_repr = bytearray.fromhex(str(hex))[::-1].hex()
    return int(big_repr,base=16) if con else big_repr

#Convert a wav file to a 2-channel numpy array
def read_wav(filename,outname,sf,prescale_outputsize,mode="dual-channel",peak_norm=True):
    file_hex        = open(filename,"rb").read().rstrip().hex()
    
    file_header     = file_hex[:8]
    chunk_size      = big_to_little(file_hex[8:16])
    format          = file_hex[16:24]

    subchk1_ID      = file_hex[24:32]
    subchk1_size    = big_to_little(file_hex[32:40])
    audio_fmt       = big_to_little(file_hex[40:44])
    num_channels    = big_to_little(file_hex[44:48])
    sample_rate     = big_to_little(file_hex[48:56])
    byte_rate       = big_to_little(file_hex[56:64])
    block_align     = big_to_little(file_hex[64:68])
    bits_per_sample = big_to_little(file_hex[68:72])
    subchk2_ID      = file_hex[72:80]    
    subchk2_size    = big_to_little(file_hex[80:88])

    data            = big_to_little(file_hex[96:],con=False)


    hex_per_sample  = int(num_channels*(bits_per_sample/8)*2)
    hex_per_channel = int(hex_per_sample/2)
    n_samples       = int(subchk2_size/(num_channels* (bits_per_sample/8)))


    #UPDATE 
    n_samples       = int(len(data) / 8)
    #input(f"\nactually {n_samples}")
    #input(data[-20:])
    max_val         = pow(2,(bits_per_sample)-1)

    ch1     = [0]   *   n_samples       # Pre-allocate arrays for decoded file
    ch2     = [0]   *   n_samples       # Pre-allocate arrays for decoded file


    #Decode file by reading hex, converting from 2's complement, 
    #and adding to proper channel 

    for i,sample in enumerate(range(n_samples)):
        # try:
            sample_start    = int(i * hex_per_sample)
            sample_end      = int(sample_start +  hex_per_sample)

            #Convert hex to int value
            c1 = int(data[sample_start:sample_start+hex_per_channel],base=16)
            c2 = int(data[sample_start+hex_per_channel:sample_end],base=16)

            #Convert hex to 2s complement
            if c1&0x8000:
                c1 = c1 - 0x10000
            if c2&0x8000:
                c2 = c2 - 0x10000

            ch1[i] = c1/max_val
            ch2[i] = c2/max_val

        # except ValueError as ve:

        #     if ((n_samples - sample) / n_samples) < .001:
        #         pass
        #     else:
        #         print(f" {ve}\n-\tBAD")

    #Ensure output is correct length
    count = 0 
    flag = False 
    while len(ch1) < prescale_outputsize:
        if len(ch1) == 0:
            print("- empty file")
            return 
        ch1.append(ch1[-1])
        ch2.append(ch2[-1])
        count += 1 
        if count > 10000 and not flag :
            print(f"\t - bad len on {filename} - {len(ch1)-10000}/{prescale_outputsize}")
            flag = True

    if len(ch1) > prescale_outputsize:
        ch1 = ch1[:prescale_outputsize]
    if len(ch2) > prescale_outputsize:
        ch2 = ch2[:prescale_outputsize]

    #Create and save the numpy array
    if mode == 'single-channel':
        arr_avg = [(ch1[i]+ch2[i])/2 for i in range(len(ch1))]
        arr = numpy.array([arr_avg])
        if peak_norm:
            try:
                arr = normalize_peak(arr)
            except ValueError:
                return 
    elif mode == 'dual-channel':
        arr = numpy.array([ch1,ch2],dtype=float)
    else:
        print(f"bad mode specified: {mode}")
        exit()

    if sf > 1:
        arr = downscale(arr,sf,mode)
    numpy.save(outname,arr)

#Chunks a file into 'chunk_length' millisecond-sized chunks
def chunk_file(fname:str,chunk_length:int,output_path:str,normalize:bool): 


    #Check that file has not been chunked 
    fout_name       = os.path.basename(fname).replace("_final","").replace(".mp4","")
    fname_check     = os.path.join(output_path,f"{fout_name}_0.wav")
    print(f"Converting {fout_name}.wav",end='',flush=True)
    if os.path.exists(fname_check):
        print("\t- Audio has already been chunked!")  
        return 0


    #Chunk audio
    full_audio  = AudioSegment.from_file(fname,"mp4")
    chunks      = make_chunks(full_audio,chunk_length=chunk_length)

    #Save files
    for i,chunk in enumerate(chunks):

        if normalize:
            chunk = normalize_level(chunk,-20)

        full_path   = os.path.join(output_path,f"{fout_name}_{i}.wav")
        chunk.export(full_path,format="wav")
    
    print(f"\t- saved {i} chunks")
    return i*(chunk_length/60)

#Downloads a list of files 
def download_all(filenames):

    for url in filenames:

        if "CATEGORY" in url:
            #Get category being downloaded
            cat                 = url.replace("CATEGORY","").replace("|","")
            download_out_path   = os.path.join(DOWNLOAD_PATH,cat)

            #Create path dir if it does not exist
            if not os.path.exists(download_out_path):
                os.mkdir(download_out_path)
                print(f"created path for category: {cat}")
            
        else:
            download_link(url,download_out_path)

#Chunks the audio from a given category 
def chunk_all(chunk_length:int,category:str,outpath:str,only="",normalize=False):
    
    global MINUTES

    audio_base_path = os.path.join(DOWNLOAD_PATH,category)
    audio_out_path  = os.path.join(CHUNK_PATH,outpath)

    #Ensure the output path exists 
    if not os.path.exists(CHUNK_PATH):
        os.mkdir(CHUNK_PATH)
    if not os.path.exists(audio_out_path):
        os.mkdir(audio_out_path)
        print(f"created path for audio outputs: {audio_out_path}")

    #Chunk the files 
    flist = os.listdir((audio_base_path))
    random.shuffle(flist)
    for fname in flist:

        #Dont chunk incomplete files
        if not "_final" in fname:
            continue
        if not only =='':
            if only == 'both':
                pass 
            elif only == 'even':
                if not fname[0] in "01234567":
                    continue 
            elif only == 'odd':
                if not fname[0] in "89abcdef":
                    continue

        #Chunk the file 
        audio_source_path   = os.path.join(audio_base_path,fname) 
        MINUTES += chunk_file(audio_source_path,1000*chunk_length,audio_out_path,normalize=normalize)
    
    print(f"added {MINUTES} to dataset")

#Convert all wav files to numpy
def read_all(category:str,sf=1,start=-1,end=-1,prescale_outputsize=529200,worker=0,numworkers=0,mode="dual-channel",peak_norm=False,verbose=False):

    #Get workers ready
    split = math.ceil(16 / numworkers) 
    worker_split = ["0123456789abcdeffff"[i*split:i*split+split] for i in range(numworkers)][worker]
    print(f"my split: {worker_split}")
    #Ensure dataset path exists 
    if not os.path.exists(DATASET_PATH):
        os.mkdir(DATASET_PATH)
    
    #build source and save paths
    audio_source_path       = os.path.join(CHUNK_PATH,category)
    audio_output_path       = os.path.join(DATASET_PATH,category)
    if not os.path.exists(audio_output_path):
        os.mkdir(audio_output_path)

    #Get chunks to convert 
    filenames   = os.listdir(audio_source_path)
    if not start == -1:
        filenames = filenames[start:end]
    total = len(filenames)
    
    #Create arrays
    for i,filename in enumerate(filenames):
        t1 = time.time()
        if not filename[0] in worker_split:
            continue
        output_full_path    = os.path.join(audio_output_path,f"{filename}.npy").replace(".wav","")
        input_full_path     = os.path.join(audio_source_path,filename)
        #Check for existing 
        print(f"Converting {input_full_path}\t{i}/{total}",end='')

        if os.path.exists(output_full_path):
            print(f" -already existed!")
            continue 
        #try:
        read_wav(input_full_path,output_full_path,sf=sf,prescale_outputsize=prescale_outputsize,mode=mode,peak_norm=peak_norm)
        # except ValueError as e:
        #     print(f" got bad value reading")
        #     input(e)
        #     pass
        #
        # Remove file after it has been processed - DONT 
        #os.remove(input_full_path)
        if verbose:
            end = f" t={(time.time()-t1):.2f}s\n"
        else:
            end = "\n"
        print(f" - success",end=end)
    return 

#Convert to a 2s comlement 
def reg_to_2s_compl(val,bits):
    """compute the 2's complement of int value val"""
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val      

#Convert a numpy array to a wav file 
def reconstruct(arr_in:numpy.array,output_file:str,outlen=529200*5):

    #Bring all values back up to 32000
    arr_in *= 32768
    arr_in = numpy.minimum(arr_in,32766)
    arr_in = numpy.maximum(arr_in,-32766)

    #Convert to python list 
    if len(arr_in) == 1:
        ch1     = list(arr_in[0][:outlen])
        ch2     = ch1 
    else:
        ch1     = list(arr_in[0])[:outlen]
        ch2     = list(arr_in[1])[:outlen]

    data    = []
    #Convert values back to 2s complement
    for c1_i,c2_i in zip(range(len(ch1)),range(len(ch2))):
        val1            = int(ch1[c1_i])
        hex_repr_2s_1   = str(binascii.hexlify(val1.to_bytes(2,byteorder='big',signed=True)))[2:-1]

        val2            = int(ch2[c2_i])
        hex_repr_2s_2   = str(binascii.hexlify(val2.to_bytes(2,byteorder='big',signed=True)))[2:-1]

        data.append(hex_repr_2s_1)
        data.append(hex_repr_2s_2)
    

    data = "".join(data)

    header  = "52494646a4ff420157415645666d7420100000000100020044ac000010b10200040010006461746180ff420100000000"


    data    = little_to_big(data)


    file = open(output_file,"wb") 
    file.write(bytes.fromhex(header))
    file.write(bytes.fromhex(str(data)))
    file.close()

#Scale a numpy array down 
def downscale(arr_in:numpy.array,sf:int,mode="dual-channel"):

    import time 
    ch1_split   = [arr_in[0][i*sf:(i+1)*sf] for i in range(int(len(arr_in[0])/sf))]
    if mode == 'dual-channel':
        ch2_split   = [arr_in[1][i*sf:(i+1)*sf] for i in range(int(len(arr_in[1])/sf))]
    
    ch1_avg     = [sum(item)/sf for item in ch1_split]
    if mode == 'dual-channel':
        ch2_avg     = [sum(item)/sf for item in ch2_split]
    
    arr_out     = numpy.array([ch1_avg,ch2_avg] if mode == "dual-channel" else [ch1_avg])

    return arr_out

#Scale a numpy array back up
def upscale(arr_in,sf):
    return numpy.repeat(arr_in,sf,axis=1)

def reduce_arr(arr,newlen):

    #Find GCF of len(arr) and len(newlen)
    gcf         = math.gcd(len(arr),newlen)
    mult_fact   = int(newlen / gcf) 
    div_fact    = int(len(arr) / gcf) 

    new_arr     = numpy.repeat(arr,mult_fact)


    return [sum(list(new_arr[n*div_fact:(n+1)*div_fact]))/div_fact for n in range(newlen)]

def upscale(tensor):
    upsampler           = torchaudio.transforms.Resample(1024,44100)
    new_tensor          = upsampler(tensor)

    torchaudio.save("AudioOut.wav",torch.stack([new_tensor,new_tensor]),44100)

def build_dataset(path_to_wav,path_to_save):

    downsampler         = torchaudio.transforms.Resample(44100,1024)

    for fname in os.listdir(path_to_wav):
        filename            = path_to_wav + fname 
        
        dual_channel_audio  = downsampler(torchaudio.load(filename)[0])
        single_channel      = dual_channel_audio[0]


        window              = 16384
        cur_i               = 0 
        length              = single_channel.shape[0]
        #Create chunks 
        while cur_i+window < length:
            save_path           = path_to_save + fname.replace(".wav","") + f"{cur_i}.tsr"
            torch.save(single_channel[cur_i:cur_i+window],save_path)

            cur_i += random.randint(2,200)


if __name__ == "__main__" and False:
    mode = sys.argv[1]

    category    = "LOFI_32s"
    ####################################################################################    
    #                                      DOWNLOAD                                    # 
    ####################################################################################    
    if mode == "-d":
        links   = open("links.txt","r").read().split()
        links   = list({l : 1 for l in links}.keys())
        print("Downloading ",len(links)-1, " links ")
        download_all(links)
    ####################################################################################    
    #                                       CHUNKS                                     # 
    ####################################################################################    
    elif mode == "-c":
        if not len(sys.argv) > 2:
            input("even,odd, or both?")
        while True:
            chunk_all(16,category="lofi",outpath=category,only=sys.argv[2],normalize=True)
            print("\n"*100)
            print("waiting for job")
            time.sleep(30)
    
    ####################################################################################    
    #                                       CHUNKS                                     # 
    ####################################################################################
    elif mode == "-r":
        if not len(sys.argv) > 3:
            print("need worker and numworkers")
        while True:
            read_all(category,sf=35,prescale_outputsize=int(5292000/6),worker=int(sys.argv[2]),numworkers=int(sys.argv[3]),mode="single-channel",peak_norm=True,verbose=True)
            print("\n"*10)
            print("waiting for job")
            time.sleep(30)
    
    elif mode == "-t":
        for i in range(214):
            fname = f"0f03823a5c_{i}.npy"
            input_vect  = numpy.load(rf"C:\data\music\dataset\LOFI_sf5_t20_c1\{fname}",allow_pickle=True)

            #input_vect[0] = (input_vect[0]+input_vect[1])/2
            #input_vect[1] = input_vect[0]
            outsout     = reconstruct(upscale(input_vect,5),f"outs2\{fname.replace('.npy','')}.wav")

    elif mode == "-u":
        root = open("links.txt").readlines()[1:]
    
        for file in root:
            if "108ee" in hash_str(file.rstrip())[:50]:
                print(f"BAD URL: {file.rstrip()}")
    else:
        print("chose from -d (download), -c (chunk), or -r (read)")
    

if __name__ == "__main__" and True:

    #build_dataset("C:/data/music/wavs/","C:/data/music/dt2/")
    upscale(torch.load("C:/data/music/dt2/alittlelonely1093.tsr"))
    exit()
    load_root           = f"{DATASET_PATH}/LOFI_sf5_t20_peak1_thrsh.95"
    stor_root           = f"{DATASET_PATH}/LOFI_sf35_t20_peak1_thrsh.95"
    if not os.path.exists(stor_root):
        os.mkdir(stor_root)
    fnames              = [f"{load_root}/{f}" for f in os.listdir(load_root)]

    newlen              = int(len(numpy.load(fnames[0])) / 7)
    for i,fname in enumerate(fnames):
        numpy_arr           = numpy.load(fname)
        numpy_arr_reduced   = reduce_arr(numpy_arr,newlen)
        numpy.save(f"{fname.replace(load_root,stor_root)}",numpy_arr_reduced)
        
        if i % 100 == 0:
            print(i)
    