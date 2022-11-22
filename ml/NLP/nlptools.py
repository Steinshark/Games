from newspaper import Article
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, LongformerTokenizer, LongformerConfig, LongformerModel,logging
logging.set_verbosity_error()
import torch 
import torch.nn as nn 
import numpy 
from transformers import AdamW
import pprint
import sys 
sys.path.insert(0,r"D:\code\projects\ml")
sys.path.insert(1,r"D:\code\projects\networking")

import database
import tlib
import networkmodels
import os 
import time 

class LanguageModel:
    
    def __init__(self,encoding="utf-8"):
        self.encoding=encoding
        self.corpus = {} 

    #Adds a corpus to use as the base langauge for this 
    #model
    #Corpus shall be a list of filenames 
    def append_file(self,corpus):
        
        #Open all files and add to corpus
        for f_name in corpus:
            try:
                f_text = open(f_name,encoding="utf-8")
                self.corpus[f_name] = f_text
            except UnicodeDecodeError as UDE:
                print(f"encoding type {self.encoding} unsucessfull for file {f_name}. Was index {corpus.index(f_name)}/{len(corpus)}.")

    #Appends to corpus dictionary.
    #addls is a list of filenames
    def append_text(self,addls):

        if not isinstance(addls,dict):
            print(f"argument 0 must be a dict. found type {type(addls)}")
        #Attempts to add all strings to file
        for textname in addls:
            try:
                utf_encoded_str = addls[textname].encode(encoding="utf-8").decode()
                self.corpus[textname] = utf_encoded_str
            except UnicodeDecodeError as UDE:
                 print(f"encoding type {self.encoding} unsucessfull for {textname}.")



#A news worker will handle downloading and working will news data
class Predictor:

    def __init__(self,tokenizer_name="allenai/led-base-16384",model_name="allenai/led-base-16384",max_pos_emb=16384,vocab_size=50265,attn_win=1024,max_enc_emb=16384):

        #Create the NLP Tools
        self.Tokenizer = LongformerTokenizer.from_pretrained(tokenizer_name)
        config = LongformerConfig(max_position_embeddings=16384,vocab_size=vocab_size,attention_window=attn_win,max_encoder_position_embeddings=max_enc_emb)
        self.EmbeddingModel = LongformerModel.from_pretrained(model_name,config=config)
        self.embedding_optim = AdamW(self.EmbeddingModel.parameters())
        #Create the market data tools 
        #This model will take 128 inputs and create a 16D vector to represent an "embedding" of the data 
        self.MetricsModel = networkmodels.FullyConnectedNetwork(128,16,loss_fn=nn.HuberLoss,optimizer_fn=torch.optim.Adam,lr=1e-4,wd=1e-5,architecture=[512,512])

    def text_to_id(self,text,max_length=1024):
        token_ids = self.tokenizer.encode_plus(text,add_special_tokens=True,padding=True,truncation=True,max_length=max_length,return_attention_mask=True,return_tensors="pt")
        return token_ids

    def train_model(self,lr=1e-4,epochs=10):
        #Build the optimizer 
        self.optimizer = AdamW(self.model.parameters(),lr=lr) 

        for epoch in epochs:


            self.optimizer.zero_grad()

    def grab_news(self):
        database.download_today()

    def analyze_news(self):
        pass


if __name__ == "__main__":

    database.DB_PATH = r"S:\data\stockdb"
    tickers = ["AAPL","MMM","AXP","CAT","KO","V"]
    n = Predictor()

    #Add to twitter searches
    for t in tickers:
            if t in database.ALIASES:
                database.DEFAULT_PARAMS['twitter'].append(f"({' OR '.join([t] + database.ALIASES[t])}) {database.DEFAULT_PARAMS['twitter'][0]}")

    print(len(database.DEFAULT_PARAMS['twitter']))
    exit()
    #Grab all news
    t0 = time.time()
    database.download_today(tickers,params=database.DEFAULT_PARAMS)
    print(f"\n\n\n\n\n\ndownloading all data took {(time.time()-t0):.2f}s")

    #ids = n.text_to_id(open(r'D:\data\wiki\pages\enron.txt',"r").read())
    #output = n.model2(ids['input_ids'],attention_mask=ids['attention_mask']) 
    #pprint.pp("success")