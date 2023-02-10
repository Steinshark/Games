import requests
import random
import time

links = {
9:"https://www.howtostudykorean.com/?page_id=502",
10:"https://www.howtostudykorean.com/?page_id=693",
11:"https://www.howtostudykorean.com/unit1/unit-1-lessons-9-16/lesson-11/",
12:"https://www.howtostudykorean.com/?page_id=943",
13:"https://www.howtostudykorean.com/?page_id=965",
14:"https://www.howtostudykorean.com/?page_id=1073",
15:"https://www.howtostudykorean.com/?page_id=1166",
16:"https://www.howtostudykorean.com/?page_id=1237",
}

fname = "korean_vocab"


def parse_grammar(page_url):
    splits = page_url.split('.mp3">')[1:]

    vocab = {}
    for s in splits:
        try:
            korean  = s.split("</a")[0]
            english = s.split("</a")[1].split(">")[2].split("<")[0]

            if not "\n" in english and not len(korean) > 10:
                vocab[korean] = english
        except IndexError:
            continue

    import pprint 
    #pprint.pp(vocab)
    return vocab

def save_progess(vocab):
    with open(fname,"w",encoding="utf-16") as file:
        writer = ""
        for k_word in vocab:
            writer += f"{k_word},{vocab[k_word]}\n"
        file.write(writer)
    return


vocab = {}
last_stopped = 0 
fails = 0
for unit in [1,2,3,4,5]:
    for lesson in list(range(last_stopped,100)):
        request_url = f"https://www.howtostudykorean.com/unit-{unit}/unit-{unit}-lesson-{lesson}"
        if unit == 1 and lesson > 8:
            if lesson > 16: 
                last_stopped = 16 
                break
            request_url = links[lesson]
        if unit > 1:
            request_url = f"https://www.howtostudykorean.com/unit-{unit}/lesson-{lesson}/"
        r   = requests.get(request_url,timeout=.5)

        if r.status_code == 200:
            print(f"Unit-{unit} Lesson-{lesson}\tsuccess")
            #input(r.text)

            new_vocab = parse_grammar(r.text)
            for word in new_vocab:
                if word in vocab:
                    vocab[word] = f"{vocab[word]} / {new_vocab[word]}"
                else:
                    vocab[word] = new_vocab[word]
            
        else:
            print(f"Unit-{unit} Lesson-{lesson}\tfailed with {r.status_code}")
            print(request_url)
            fails += 1 
            if fails > 2:
                last_stopped = lesson-3
                break
        time.sleep(random.random()+2)
    save_progess(vocab)
