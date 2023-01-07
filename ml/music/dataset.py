from pytube import YouTube 
from pydub import AudioSegment
from pydub.utils import make_chunks
import os 
from binascii import hexlify
from pydub.exceptions import CouldntDecodeError
import ctypes
import numpy 
from networks import AudioGenerator,AudioDiscriminator
import torch 
import binascii
DOWNLOAD_PATH           = r"C:\data\music\downloads"
CHUNK_PATH              = r"C:\data\music\chunked"
DATASET_PATH            = r"C:\data\music\dataset"

MINUTES                 = 0 
def download_link(url:str,path:str): 

    #Grab video 
    yt_handle               = YouTube(url)
    print(f"downloading {url}")
    link_audio              = yt_handle.streams.filter(only_audio=True).first().download(output_path=path)
    print(f"\tsuccess")

def rename_files():
    for cat in os.listdir(DATABASE_PATH):
        for file in os.listdir(os.path.join(DATABASE_PATH,cat)):
            old_path   = os.path.join(os.path.join(DATABASE_PATH),cat,file) 
            fname = file.replace(" ","").replace("mp4","mp3")
            new_path   = os.path.join(os.path.join(DATABASE_PATH),cat,fname) 

            os.rename(old_path,new_path)

def big_to_little(hex,con=True):
    little_repr = bytearray.fromhex(hex)[::-1].hex()
    return int(little_repr,base=16) if con else little_repr

def little_to_big(hex,con=False):
    try:
        big_repr = bytearray.fromhex(hex)[::-1].hex()
    except TypeError:
        big_repr = bytearray.fromhex(str(hex))[::-1].hex()
    return int(big_repr,base=16) if con else big_repr


def read_wav(filename):
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

    count = 0 
    flag = False 
    while len(ch1) < 5291999:
        ch1.append(ch1[-1])
        ch2.append(ch2[-1])
        count += 1 
        if count > 10000 and not flag :
            print(f"bad len on {filename}")
            flag = True
    
    if len(ch1) > 5291999:
        ch1 = ch1[:5291999]
    if len(ch2) > 5291999:
        ch2 = ch2[:5291999]
    #Create and save the numpy array
    arr = numpy.array([ch1,ch2],dtype=float)
    numpy.save(os.path.join(DATASET_PATH,str(filename.__hash__())[:10]),arr)

#Chunks a file into 'chunk_length' millisecond-sized chunks
def chunk_file(file_name:str,chunk_length:int,save_path:str): 
    full_audio  = AudioSegment.from_file(file_name,"mp4")
    chunks      = make_chunks(full_audio,chunk_length=chunk_length)

    for i,chunk in enumerate(chunks):
        root_name = str(file_name.__hash__())[:10]
        fname   = f"{root_name}_{i}.wav"
        print(f"saving chunk {fname}")
        full_path   = os.path.join(save_path,fname)
        if os.path.exists(full_path):
            print("\tfile existed")
        else:
            chunk.export(full_path,format="wav")

def download_all():
    cur_cat     = ""
    cur_path    = ""
    downloaded = [] 

    for line in open("links.txt").readlines():
        if line in downloaded:
            continue 
        else:
            downloaded.append(line)
        if "CATEGORY" in line:
            cat = line.replace("CATEGORY","").replace("|","").rstrip()
            full_path = os.path.join(DOWNLOAD_PATH,cat)
            if not os.path.exists(full_path):
                os.mkdir(full_path)
                print(f"created path for category: {cat}")
            cur_cat = cat
            cur_path = full_path
        else:
            download_link(line.rstrip(),cur_path)

def chunk_all():
    global MINUTES
    for cat in os.listdir(DOWNLOAD_PATH):
        for file in os.listdir(os.path.join(DOWNLOAD_PATH,cat)):
            try:
                full_file_path      = os.path.join(os.path.join(DOWNLOAD_PATH),cat,file) 
                print(f"opening {full_file_path}")
                chunk_file(full_file_path,1000*60*2,CHUNK_PATH)
                MINUTES += 2
            except CouldntDecodeError:
                pass
    print(f"created dataset {MINUTES} minutes long")

def read_all():
    total = len(os.listdir(CHUNK_PATH))
    for i,filename in enumerate(os.listdir(CHUNK_PATH)):
        print(f"{i}/{total}")
        read_wav(os.path.join(CHUNK_PATH,filename))

def reg_to_2s_compl(val,bits):
    """compute the 2's complement of int value val"""
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val      

def reconstruct(arr_in:numpy.array,output_file):

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
        val1        = int(ch1[c1_i])
        hex_repr_2s_1   = str(binascii.hexlify(val1.to_bytes(2,byteorder='big',signed=True)))[2:-1]

        val2        = int(ch2[c2_i])
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

if __name__ == "__main__":

    #download_all()
    #chunk_all()
    read_all()
