import requests 
import random 
from bs4 import BeautifulSoup as bs 
import time 

def grab_proxies():
    url = "https://free-proxy-list.net/"

    gather = bs(requests.get(url).content,"html.parser")
    proxy_list = {}
    for row in gather.find("table", attrs={"id": "proxylisttable"}).find_all("tr")[1:]:
        tds = row.find_all("td")
        try:
            proxy_ip = tds[0].text.strip()
            proxy_port = tds[1].test.strip()
            proxy_list[f"{proxy_ip}:{proxy_port}"] = time.time()
        except IndexError:
            pass 
    return proxy_list

# This function builds a one time use session to grab a url  
# a proxy can be optionally supplied 
def grab_webpage(url,proxy=None,t_out=1):
    
    #Build session
    session = requests.Session()
    session.proxies = {"http" : proxy,"https":proxy} if proxy else {}
    page_html = session.get(url,timeout=t_out)

    if not page_html.status_code == 200:
        return None 
    else:
        return page_html.text.strip()




if __name__== "__main__":
    proxies = grab_proxies()
    grab_webpage(proxy="92.154.53.1:80")

