
#Handle network spoofing
from fake_useragent import UserAgent 
from fake_headers import Headers
header_gen = Headers(browser="chrome",os="win",headers=True)

#Handle networking and requests
import requests 
from netErr import *
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from newspaper import Article
import tlib
from  robin_stocks import robinhood
from robin_stocks.robinhood import authentication as rauth 
from robin_stocks.robinhood import profiles as rprof 
from robin_stocks.robinhood import stocks as rstocks

from datetime import datetime,timezone
#Instantiate some globals
PARSER = Article("")
OPTIONS = webdriver.chrome.options.Options()
OPTIONS.add_argument("--headless")
OPTIONS.add_argument('--disable-logging') 
OPTIONS.add_experimental_option('excludeSwitches', ['enable-logging'])


#GLOBALS
DEFAULT_DRIVER  = webdriver.Chrome(options=OPTIONS)
LOGIN_TSTAMP    = 0
LOGIN_DURATION  = 24*3600

#Utilities
import time
import random
import json 
import re
#Fill our proxy list with the most recent data
# allows for proxies to be filtered 
def update_proxy_list(country="US",protocol="https"): 
    url = f"https://www.freeproxylists.net/?c={country}&pt=&pr={protocol}"
    headers = {"referer": "https://seekingalpha.com/","user-agent" : UserAgent().random,"authority":"https://seekingalpha.com/"}
    r = requests.get(url,headers=headers)
    
    #Bad response
    if not r.status_code == 200 or "robot" in r.text or "captcha" in r.text:
        print(f"request went bad - {r.status_code}")
        input(r.text)
        return
    
    #Good response
    else:
        page_scrape = r.text
        input(page_scrape)
        page_scrape = page_scrape.split("DataGrid")[1].split("tbody>")[1].split("</tbody>")[0]
##
##
#Unused and unwritten
def build_driver():
    pass
##
##
#Grab a list of urls of news articles on a specific ticker from Yahoo Finance 
def pull_yf_urls(ticker):
  
    base_url = f"https://finance.yahoo.com/quote/{ticker}/news?"
    settings = webdriver.chrome.options.Options()

    #CANNOT BE DONE HEADLESSLY 
    #settings.add_argument("--headless")
    driver = webdriver.Chrome(options=settings)
    driver.get(base_url)
    
    #Make sure page is good return
    assert "No results found." not in driver.page_source
    
    #Get past popup
    try:
        button = driver.find_element(By.CSS_SELECTOR,"[aria-label=Close]")
        button.click()
    except:
        pass 

    #Lengthen page 
    for i in range(10):
        driver.execute_script(f"window.scrollTo(0, {2000*i});")
        time.sleep(1)

    #Grab all URLS
    raw_html = driver.page_source
    url_chunks = raw_html.split("js-content-viewer wafer")[1:]

    if not len(url_chunks) > 1:
        raise UnexpectedReturnErr("Split was less than 1")
    
    return [chunk.split('href="')[1].split('"')[0] for chunk in url_chunks]
##
##
#Grab a list of urls of news articles on a specific ticker from Yahoo Finance 
def pull_fool_urls(ticker,exchange):
    base_url = f"https://www.fool.com/quote/{exchange}/{ticker}/"
    settings = webdriver.chrome.options.Options()
    #CANNOT BE DONE HEADLESSLY 
    #settings.add_argument("--headless")
    driver = webdriver.Chrome(options=settings)
    driver.get(base_url)
    time.sleep(3)
    b = driver.find_element(By.CLASS_NAME,".flex.items-center.load-more-button.foolcom-btn-white.mt-24px.md:mb-80.mb-32px")
    b.click()
##
##
#Downoads a url 
def grab_newspage(url,use_driver=False):

    if use_driver:
        DEFAULT_DRIVER.get(url)
        return DEFAULT_DRIVER.page_source

    r = requests.get(url=url,headers=header_gen.generate())

    #Check status code 
    if not r.status_code == 200:
        return ""
    else:
        return r.text
##
##
#Grab only article contents off of html
def clean_raw_html(html):
    if not html:
        return " "
    text = ""
    paragraphs = [i.split("</p>")[0] for i in html.split("<p>")]
    new_p = []

    #Scan for text 
    for p in paragraphs:
        if "<" in p or "/" in p or "[" in p and not "</strong>" in p:
            continue
        else:
            new_p.append(p)

    #Ensure we got SOMETHING
    if len(new_p) == 0:
        PARSER.set_html(html)
        PARSER.parse()
        PARSER.text = "LEN WAS 0" + PARSER.text
        for char in PARSER.text:
            text += char.encode("utf-8").decode("utf-8")
    else:
        for char in ". ".join([p.strip() for p in new_p]):
            text += char.encode("utf-8").decode("utf-8")

    #Finish good text
    if not text:
        PARSER.set_html(html)
        PARSER.parse()
    return text.replace("&nbsp;"," ").replace("&#8217;","'").replace("&#8230;","...").replace("&amp;","&").replace("&#8221;","'".replace("&#8220;","'"))
##
##
#Make a Mediastack API request
#Params ican be {"data","categories","symbols","langauge","countries","keywords","limit","offset"}
def grab_mediastack(params:dict):
    base_url = "http://api.mediastack.com/v1/news?"
    access_key = open("D:\code\mediastack.txt","r").read().strip()[1:-1]
    base_url += f"access_key={access_key}"

    #Add all addtl params
    for param in params:
        base_url += f"&{param}={params[param]}" 
    
    
    #Get data 
    r = requests.get(url=base_url)

    if not r.status_code == 200:
        raise StatusCodeErr(f"mediastack gave code {r.status_code} on url:{base_url}",code=r.status_code,url=r.ur)
    else:
        return json.loads(r.text)
##
##
#Make an alphavantage API request
#Params ican be {"data","categories","symbols","langauge","countries","keywords","limit","offset"}
def grab_alphavantage(params:dict):
    api_key = open('D:\code\\alphavantage.txt','r').read().strip()
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={params['tickers']}&apikey={api_key}&time_from={params['t_from']}"

    r = requests.get(url=url)

    if not r.status_code == 200:
        raise StatusCodeErr(f"alphavantage gave code {r.status_code} on url:{r.url}",code=r.status_code,url=r.ur)
    data = r.json()
    return data
##
##
#Make twitter API call
def grab_twitter(twitter_lookup,fields=None):
    try:
        if fields is None: fields = "created_at"
        response = tlib.grab_tweets(twitter_lookup,fields=fields)
        if 'data' in response:
            return response['data']
        else:
            return None
    except StatusCodeErr as SCE:
        print(f"twitter lookup failed code {SCE.code} for {SCE.url}")
        return None 
##
##    
#Make a robinhood api call for news 
def grab_robinhood(ticker):

    #Login
    try: 
        rprof.load_account_profile(info=None)
        logged_in = True
    except Exception:
        logged_in = False
    if not logged_in:
        user,passw = tuple(json.loads(open(r"D:\code\robinhood.txt","r").read()))
        rauth.login(username=user,password=passw,expiresIn=24*3600)

    #Grab the data for all stocks listed
    responses = rstocks.get_news(ticker)
    if not responses:
        return False 
    
    data = [{"title":resp['title'],"url":resp['url'],"date":resp['published_at']} for resp in responses]
    return data
##
##
#Make a request for market data for a stock ticker
def grab_robinhood_data(ticker,interval="5minute",span="week"):

    #Login
    try: 
        rprof.load_account_profile(info=None)
        logged_in = True
    except Exception:
        logged_in = False
    if not logged_in:
        user,passw = tuple(json.loads(open(r"D:\code\robinhood.txt","r").read()))
        rauth.login(username=user,password=passw,expiresIn=24*3600)

    response = rstocks.get_stock_historicals(ticker,interval=interval,span=span)
    return [{'time':(datetime.now(timezone.utc)-datetime.strptime(r['begins_at'],"%Y-%m-%dT%H:%M:%S%z")).seconds,'open':r['open_price'],'close':r['close_price'],'high':r['high_price'],'low':r['low_price'],'volume':r['volume']} for r in response]
##  
##
##
if __name__ == "__main__":
    resp = grab_robinhood_data("AAPL")
    print(resp[389])