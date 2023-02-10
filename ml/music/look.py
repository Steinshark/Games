import requests 
import time 
import random 

names = open("data.csv","r").readlines()
names = [n.rstrip() for n in names]
names = [n.split(",") for n in names]

print(names[100:110])

for name in names[100:]:
    b_id = name[0]

    url = f"https://cis.scc.virginia.gov/EntitySearch/BusinessInformation?businessId={b_id}&source=FromEntityResult&isSeries%20=%20false"

    r = requests.get(url,timeout=2)
    time.sleep(1+random.random()*2) 
    if r.status_code == 200:
        html = r.text
        f = open(f"pages/{b_id}.html","w")
        f.write(html)
        f.close()
        print(f"saved to {f.name}")
    else:
        print("bad")
