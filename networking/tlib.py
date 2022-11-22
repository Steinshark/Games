import requests 
import json 
from netErr import *
import pprint 

def grab_tweets(QUERY,fields="created_at",max_results=100):
    #Grab API keys 
    twitter_keys = json.loads(open(r"D:\code\twitterkeys.txt").read())
    
    #Make request 
    base_url = "https://api.twitter.com/2/tweets/search/recent"
    params = {'query':QUERY,"tweet.fields":fields,"max_results":max_results}
    query_params = {'query': '(from:twitterdev -is:retweet) OR #twitterdev','tweet.fields': 'author_id'}

    r = requests.get(url=base_url,params=params,headers={"Authorization": f"Bearer {twitter_keys['Bearer_token']}"})

    #Ensure proper code 
    if r.status_code == 200:
        return json.loads(r.text) 
    else:
        raise StatusCodeErr(f"grab_tweets recieved response code {r.status_code} on url {r.url}",r.status_code,url=r.url)

if __name__ == "__main__":
    try:
        import sys
        pprint.pp(grab_tweets(" ".join(sys.argv[1:]))['data'][:5])
    except StatusCodeErr as SCE:
        print(SCE)
        exit()