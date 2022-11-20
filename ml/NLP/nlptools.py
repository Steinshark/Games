


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


    