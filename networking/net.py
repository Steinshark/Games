import requests 
from fake_useragent import UserAgent 

__PROXIES = {}



def get_data():
    
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


if __name__ == "__main__":
    update_proxy_list()