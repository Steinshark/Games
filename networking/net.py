
#Handle network spoofing
from fake_useragent import UserAgent 
from fake_headers import Headers
header_gen = Headers(browser="chrome",os="win",headers=True)

#Handle net big tasks
import requests 
from netErr import *
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

#Utilities
import time
import random
import json 

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

def build_driver():
    pass
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

#Downoads a url 
def grab_yf_newspage(url):
    r = requests.get(url=url,headers=header_gen.generate())

    #Check status code 
    if not r.status_code == 200:
        raise StatusCodeErr(f"yf newspage download failed on url {url} with code {r.status_code}")
    else:
        return r.text

#Make a Mediastack API request
#Params ican be {"data","categories","symbols","langauge","countries","keywords","limit","offset"}
def grab_mediastack(params:dict):
    base_url = "http://api.mediastack.com/v1/news?"
    access_key = open("D:\code\mediastack.txt","r").read().strip()[1:-1]
    base_url += f"access_key={access_key}"

    #Add all addtl params
    for param in params:
        base_url += f"&{param}={params[param]}" 
    
    
    print(f"making request to {base_url}\n")
    #Get data 
    r = requests.get(url=base_url)

    if not r.status_code == 200:
        raise StatusCodeErr(f"mediastack gave code {r.status_code} on url:{base_url}")
    else:
        return json.loads(r.text)

if __name__ == "__main__":
    from pprint import pp
    res = grab_mediastack({"date":"2022-11-13,2022-11-20","categories":"business","symbols":"aapl","language":"en","countries":"us","limit":"100"})
    while(True):
        exec(input())