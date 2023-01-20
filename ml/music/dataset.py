from pytube import YouTube 
from pydub import AudioSegment
from pydub.utils import make_chunks
import os 
import sys 
from binascii import hexlify
from pydub.exceptions import CouldntDecodeError
import ctypes
import numpy 
from networks import AudioGenerator,AudioDiscriminator
import torch 
import binascii
import hashlib
import time 
from http.client import IncompleteRead
import math 
import random 

#Inizialize directories properly
if "linux" in sys.platform:
    DOWNLOAD_PATH           = r"/media/steinshark/stor_sm/music/downloads"
    CHUNK_PATH              = r"/media/steinshark/stor_sm/music/chunked"
    DATASET_PATH            = r"/media/steinshark/stor_lg/music/dataset"
else:
    DOWNLOAD_PATH           = r"C:/data/music/downloads"
    CHUNK_PATH              = r"C:/data/music/chunked"
    DATASET_PATH            = r"C:/data/music/dataset"

MINUTES                 = 0 


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
    yt_handle       = YouTube(url)

    #Telemetry
    if not index is None:
        tel_msg         = f"downloading {url[:50]}\t[{index}/{total}]"
    else:
        tel_msg         = f"downloading {url[:50]}"
    print(tel_msg,end='',flush=True)
    try:
        filepath        = yt_handle.streams.filter(only_audio=True).first().download(output_path=output_path)

        os.rename(filepath,out_path_full)
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
def read_wav(filename,outname,sf,prescale_outputsize,mode="dual-channel"):
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


    max_val         = pow(2,(bits_per_sample)-1)

    ch1     = [0]   *   n_samples       # Pre-allocate arrays for decoded file
    ch2     = [0]   *   n_samples       # Pre-allocate arrays for decoded file


    #Decode file by reading hex, converting from 2's complement, 
    #and adding to proper channel 
    for i,sample in enumerate(range(n_samples)):
        try:
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

        except ValueError:

            if ((n_samples - sample) / n_samples) < .001:
                pass
            else:
                print("-\tBAD")

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
        arr = numpy.array([arr_avg,arr_avg])
    elif mode == 'dual-chanel':
        arr = numpy.array([ch1,ch2],dtype=float)
    else:
        print(f"bad mode specified: {mode}")
        exit()

    if sf > 1:
        arr = downscale(arr,sf)
    numpy.save(outname.replace(".wav",""),arr)

#Chunks a file into 'chunk_length' millisecond-sized chunks
def chunk_file(fname:str,chunk_length:int,output_path:str): 


    #Check that file has not been chunked 
    fout_name       = os.path.basename(fname).replace("_final","").replace(".mp4","")
    fname_check     = os.path.join(output_path,f"{fout_name}_0.wav")
    print(f"Converting {fout_name}.wav",end='',flush=True)
    if os.path.exists(fname_check):
        print("\t- Audio has already been chunked!") 
        return 0
    fname_check     = os.path.join(DATASET_PATH,output_path.split("/")[-1], f"{fout_name}_0.npy")
    if os.path.exists(fname_check):
        print("\t- Audio has already been chunked!") 
        return 0


    #Chunk audio
    full_audio  = AudioSegment.from_file(fname,"mp4")
    chunks      = make_chunks(full_audio,chunk_length=chunk_length)

    #Save files
    for i,chunk in enumerate(chunks):
        full_path   = os.path.join(output_path,f"{fout_name}_{i}.wav")
        chunk.export(full_path,format="wav")
    
    print(f"\t- saved {i} chunks")
    return i*1000*chunk_length

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
def chunk_all(chunk_length:int,category:str,outpath:str,only=""):
    
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
                if not fname[0] in "01234678":
                    continue 
            elif only == 'odd':
                if not fname[0] in "9abcdef":
                    continue



        #Chunk the file 
        audio_source_path   = os.path.join(audio_base_path,fname) 
        MINUTES += chunk_file(audio_source_path,1000*chunk_length,audio_out_path)
    
    print(f"added {MINUTES} to dataset")

#Convert all wav files to numpy
def read_all(category:str,sf=1,start=-1,end=-1,prescale_outputsize=529200,worker=0,numworkers=0,mode="dual-chanel"):

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
        if not filename[0] in worker_split:
            continue
        output_full_path    = os.path.join(audio_output_path,f"{filename}.npy")
        input_full_path     = os.path.join(audio_source_path,filename)
        #Check for existing 
        print(f"Converting {input_full_path}\t{i}/{total}",end='')
        if os.path.exists(output_full_path):
            print(f" -already existed!")
            continue 
        try:
            read_wav(input_full_path,output_full_path,sf=sf,prescale_outputsize=prescale_outputsize,mode=mode)
        except ValueError:
            print(f" got bad value reading")
            pass
        #
        # Remove file after it has been processed
        os.remove(input_full_path)
        print(f" - success")
    return 

#Convert to a 2s comlement 
def reg_to_2s_compl(val,bits):
    """compute the 2's complement of int value val"""
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val      

#Convert a numpy array to a wav file 
def reconstruct(arr_in:numpy.array,output_file:str):

    #Bring all values back up to 32000
    arr_in *= 32768
    arr_in = numpy.minimum(arr_in,32766)
    arr_in = numpy.maximum(arr_in,-32766)

    #Convert to python list 
    ch1     = list(arr_in[0])[:5291999]
    ch2     = list(arr_in[1])[:5291999]

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
def downscale(arr_in:numpy.array,sf:int):

    import time 
    ch1_split   = [arr_in[0][i*sf:(i+1)*sf] for i in range(int(len(arr_in[0])/sf))]
    ch2_split   = [arr_in[1][i*sf:(i+1)*sf] for i in range(int(len(arr_in[1])/sf))]
    
    ch1_avg     = [sum(item)/sf for item in ch1_split]
    ch2_avg     = [sum(item)/sf for item in ch2_split]
    
    arr_out     = numpy.array([ch1_avg,ch2_avg])

    return arr_out

#Scale a numpy array back up
def upscale(arr_in,sf):
    return numpy.repeat(arr_in,sf,axis=1)



if __name__ == "__main__":
    mode = sys.argv[1]

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
            chunk_all(60,"LOFI","LOFI_sf5_t60_c1",only=sys.argv[2])
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
            read_all("LOFI_sf5_t60_c1",sf=5,prescale_outputsize=5292000/2,worker=int(sys.argv[2]),numworkers=int(sys.argv[3]),mode="single-channel")
            print("\n"*100)
            print("waiting for job")
            time.sleep(30)
    
    elif mode == "-t":
        input_vect  = numpy.load(r"C:\data\music\dataset\LOFI_sf5_t60_c1\3dc6d2931d_102.npy",allow_pickle=True)

        #input_vect[0] = (input_vect[0]+input_vect[1])/2
        #input_vect[1] = input_vect[0]
        outsout     = reconstruct(upscale(input_vect,5),"Test.wav")
    else:
        print("chose from -d (download), -c (chunk), or -r (read)")

