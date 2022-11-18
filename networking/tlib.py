import requests 
import json 
from netErr import *

def grab_tweets(QUERY,max_results=100):
    #Grab API keys 
    twitter_keys = json.loads(open("twitterkeys.txt").read())
    
    #Make request 
    base_url = "https://api.twitter.com/2/tweets/search/recent/"
    query = f"{base_url}?query={QUERY}&max_results={max_results}"
    r = requests.get(url=query,headers={"Authorization": f"Bearer {twitter_keys['Bearer_token']}"})

    #Ensure proper code 
    if r.status_code == 200:
        import pprint 
        pprint.pp(r.text)
        return r.text 
    else:
        raise StatusCodeErr(f"grab_tweets recieved response code {r.status_code} on url {query}")

if __name__ == "__main__":
    try:
        import sys
        grab_tweets(sys.argv[1])
    except StatusCodeErr as SCE:
        print(SCE)
        exit()