import os 
import sys 
sys.path.insert(0,r"D:\code\projects\networking")
from nettools import *
from netErr import *
import nettools
from datetime import datetime
import json 
import time 
## >>>>>>>>>>>>>>>>>>>>>>>>>>>> BEGIN DEFAULT DEFINITIONS <<<<<<<<<<<<<<<<<<<<<<<<<<<< ##

DB_NEWS_PATH = r"S:\Data\market\news"
DB_DATA_PATH = r"S:\Data\market\data"
DOWNLOAD_LINK_LIST  = []

DEFAULT_STOCKS      = ["MMM","AAPL","AXP","CAT","KO","V"]
DEFAULT_PARAMS      = {  "mediafire":
                        [{"date":"2022-11-17,2022-11-20","categories":"business","symbols":"","language":"en","countries":"us","limit":"100","offset":off} for off in [0,100]],
                    "alphavantage":
                        [{'tickers':"","t_from":"20221113T0000","limit":"200"}],
                    "twitter":
                        ["(nyse OR nasdaq OR trade OR buy OR sell OR market OR invest OR company OR business OR finance OR quarter OR report) -has:links -is:retweet lang:en"]
                }

ALIASES             = {     "MMM"   : ["3M","Minnesota Mining and Manufacturing", "Minnesota Mining Manufacturing"],
                            "AAPL"  : ["Apple","Apple_Inc","APL","AppleInc"],
                            "AXP"   : ["American Express","AMEX","AmericanExpress"],
                            "CAT"   : ["Caterpillar",],
                            "KO"    : ["Coca-Cola","CocaCola","Coke"],
                            "V"     : ["Visa"]
                        }

COLLECTION_START    = 1667692800

COLLECTED_TOTAL     = 0
MISSED_TOTAL        = 0
DUPLICATE_TOTAL     = 0
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>> END DEFAULT DEFINITIONS <<<<<<<<<<<<<<<<<<<<<<<<<<<<< ##
##
##
##
##
##
#Saves to the database
def save_media_source(name:str,timestamp:str,source:str,contents:str):
    
    #Globals
    global DUPLICATE_TOTAL

    #Create unique media ID 
    f_name = f"{DB_NEWS_PATH}\{name.__hash__()}_{timestamp.replace(':','{COLON}')}.txt"
    
    #Open file in DB 
    if not os.path.exists(DB_NEWS_PATH):
        #Try to make it 
        print(f"PATH {DB_NEWS_PATH} was not found, attempting to create.")
        os.mkdir(DB_NEWS_PATH)
    
    #Ensure we don't have duplicate files, if so just skip them 
    if os.path.exists(f_name):
        DUPLICATE_TOTAL += 1
        return False
    
    #Open and save contents to file 
    file = open(f_name,"w",encoding='utf-8')
    file.write(f"source:{source}\n")
    file.write(contents)

    #Return 
    return True
##
##
#This method was meant to be run in parallel in separate threads. I currently just do downloading
#within the "download_today" method, but this can be changed with a kwarg
def download_worker():

    #Ensure there is something there
    if DOWNLOAD_LINK_LIST:

        #Grab the next available url and following info 
        name,timestamp,url,source = DOWNLOAD_LINK_LIST.pop(0)
        
        #Get the contents 
        contents = grab_newspage(url)
        if "please enable Javascript and cookies" in contents.lower():
            contents = grab_newspage(url,use_driver=True)

        #Try to write to the database if there's anything to write
        if contents:
            save_media_source(name,timestamp,source,contents)
##
##
def download_today(tickers:list[str],params={},today_only=True,MEDIAFIRE:bool=True,ALPHAVANTAGE:bool=True,TWITTER:bool=True,ROBINHOOD:bool=True):

    #Globals 
    global MISSED_TOTAL, DUPLICATE_TOTAL, COLLECTED_TOTAL

    #Download Mediafire
    for t in tickers:

        #Ensure not skipped
        if not MEDIAFIRE:
            break

        #Make API requests
        for param_set in params['mediafire']:
            param_set["symbols"] = t
            fetch = grab_mediastack(param_set)

            for result in fetch["data"]:
                title = result['title']
                url = result['url']
                pub_date = result['published_at']
                DOWNLOAD_LINK_LIST.append((title,pub_date,url,"MediaFire"))

                #Attempt to grab the content
                content = grab_newspage(url)
                if "please enable Javascript and cookies" in content.lower():
                    content = grab_newspage(url,use_driver=True)
                
                #Clean the content
                content = clean_raw_html(content)
                
                if content:
                    save_media_source(title,pub_date,"MediaFire",content)
                    COLLECTED_TOTAL += 1
                else:
                    MISSED_TOTAL    += 1

            #Ensure we don't cycle over the same source        
            if fetch["pagination"]['total'] < int(param_set['limit']): 
                break
        
    #Download alphavantage
    for t in tickers:

        #Ensure Collecting 
        if not ALPHAVANTAGE:
            break 

        for param_set in params["alphavantage"]:
            param_set["tickers"] = t
            fetch = grab_alphavantage(param_set)

            for result in fetch["feed"]:
                
                #Grab fields
                title = result['title']
                url = result['url']
                pub_date = result['time_published']
                DOWNLOAD_LINK_LIST.append((title,pub_date,url,"alphavantage"))

                #Attempt to grab the content
                content = grab_newspage(url)
                if "please enable Javascript and cookies" in content.lower():
                    content = grab_newspage(url,use_driver=True)
                
                #Clean the content
                content = clean_raw_html(content)

                #Only save if it exists
                if content:
                    save_media_source(title,pub_date,"AlphaVantage",content)
                    COLLECTED_TOTAL += 1 
                else:
                    MISSED_TOTAL += 1

    #Download Twitter
    for twitter_lookup in params["twitter"]:

        #Ensure not skipped 
        if not TWITTER:
            break 

        twitter_lookup = f"{t} {twitter_lookup}"
        tweets = grab_twitter(twitter_lookup)
        
        #Ensure we did got some tweets back 
        if not tweets:
            continue 
        else:
            for tweet in tweets:
                #Grab fields
                date_posted = tweet["created_at"]
                content     = tweet["text"] 
                id          = tweet["id"]

                #Save tweet 
                try:
                    save_media_source(id,date_posted,"twitter",content)
                except FileNotFoundError as FNFE:
                    if "[WinError 54]" in FNFE.__str__():
                        print("It appears that you are trying to connect to a DB that does not exist:")
                        print(f"\t{FNFE}")
                        return
                    else:
                        print(FNFE)

    #Download Robinhood 
    for t in tickers:

        #Ensure not skipped 
        if not ROBINHOOD:
            break
        data = grab_robinhood(t)
        
        #Parse data 
        for d in data:
            title       = d['title'] 
            date_posted = d['date']
            url         = d['url']

            DOWNLOAD_LINK_LIST.append((title,date_posted,url,"robinhood")) 

            #Attempt to grab the content
            content = grab_newspage(url)
            if "please enable Javascript and cookies" in content.lower():
                content = grab_newspage(url,use_driver=True)
            
            #Clean the content
            content = clean_raw_html(content)

            #Only save if it exists
            if content:
                COLLECTED_TOTAL += int(save_media_source(title,date_posted,"robinhood",content))
            else:
                MISSED_TOTAL += 1
##
##
#Take the wikipedia XML file and iteratively save each article to disk
def carve_wiki():
    STOPTITLES = ["demographics","river","lake","diego","nick","geography","henry","holiday","house","howard","hudson","hunt","ian","index of ","list of","interstate","isabel","isaac","politics of","robert","roger","sam","samue","san","thomas","university"]
    file_raw = open("D:\data\wiki\wiki.xml","r",encoding='utf-8')

    #Get to first page
    while file_raw:
        line = file_raw.__next__()
        if "<page>" in line:
            break 

    title = ""
    new_file = ""
    i = 0
    while file_raw:

        #PARSE ONE PAGE OF WIKI
        new_file = ""
        financial = 0
        reset = False
        line = file_raw.__next__().strip()
        title = line.replace("<title>","").replace("</title>","").replace("?","")

        if "<page>" in title:
            line = file_raw.__next__().strip()
            title = line.replace("<title>","").replace("</title>","").replace("?","").replace(":","").replace("/","").replace("\\","").replace("*","-")
        title = title.lower()
        i += 1 
        if not (i % 10000):
            print(f"filtered {i}/2.7mil")
        for w in STOPTITLES:
            if w in title:
                while not "</page>" in line:
                    line = file_raw.__next__().strip()
                reset = True
                break  

        #print(f"On page {title}",end="")
        redirect_flag = False 
        
        #Read until end of page
        while not "</page>" in line and not reset:

            #Get the next line 
            line = file_raw.__next__().strip()

            #Check if it means were redirecting 
            if "redirect title" in line:
                line = line.replace('<redirect title="',"").replace('" />',"")
                if line:
                    #print(f"-was a redirect to {line}")
                    redirect_flag = True
                    while not "</page>" in line:
                        line = file_raw.__next__().strip()
                    break
            #Check to skip past references
            elif "== See also ==" in line or "=== Citations ===" in line:
                    while not "</page>" in line:
                        line = file_raw.__next__().strip()
                    break   
            
            #Otherwise keep appending to the file 
            if "[[" in line:
                line = line.replace("[[","")
                if "|" in line:
                    line = line.split("|")[0]
            line = line.replace("]]","")
            line = line.replace("ref&gt;"," ").replace("&lt;"," ").replace("%quot"," ")
            new_file += line

            if financial < 4:
                for word in ["market","stock","econo","financ","business","nyse","nasdaq","exchange"]:
                    if word in line:
                        financial += 1 
                        break
            
        #Save file 
        if not redirect_flag and financial >= 4:
            try:
                #print(f"-saving")
                with open(fr"D:\data\wiki\financehits\{title}.txt","w",encoding="utf-8") as file:
                    file.write(new_file)
            except FileNotFoundError as FNFE:
                print(FNFE)
        else:
            pass
##
##
#This method is meant to grab data and append to a weekly,stock specific db file ONLY if it doenst 
#already exist in there
#All data is added in 5 min format
def robinhood_data_download(tickers:list[str]):
    
    for ticker in tickers:
        
        #Download data 
        data = grab_robinhood_data(ticker)

        for item in data:
            #Figure out what weekly DB file it belongs in 
            week = (item['time']- COLLECTION_START) // 604800
            f_name = os.path.join(DB_DATA_PATH, f"{ticker.upper()}_{week}.sdbf")    #"StockDataBaseFile"

            #File contents will be a line of the json encoded dict 
            contents = json.dumps(item)
            contents += "\n"

            #Create the file if it doesnt exist
            if not os.path.exists(f_name):
                file = open(f_name,"w",encoding='utf-8')
                #Add to file and leave 
                file.write(contents)
                file.close()
            #Ensure its not in here already
            else:
                file = open(f_name,"r")
                
                #if it exists, skip 
                if contents in file.read():
                    file.close()
                #Otherwise, write the data to the file    
                else:
                    file.close()
                    file = open(f_name,"a")
                    #Add to file and leave 
                    file.write(contents)
                    file.close()
##
##
##     
if __name__ == "__main__":
    t0 = time.time()
    robinhood_data_download(DEFAULT_STOCKS)
    download_today(DEFAULT_STOCKS,params=DEFAULT_PARAMS)
    total_attempts = COLLECTED_TOTAL + MISSED_TOTAL
    print(f"Attempted to fetch {total_attempts} items in {(time.time()-t0):.1f}s\n\t{(100*COLLECTED_TOTAL/(total_attempts)):.2f}%\tnew\n\t{(100*DUPLICATE_TOTAL/total_attempts):.2f}%\texisted\n\t{(100*(MISSED_TOTAL-DUPLICATE_TOTAL)/total_attempts):.2f}%\tfailed")

