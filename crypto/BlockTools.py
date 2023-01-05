import requests
import json 
import datetime
import bitcoin_tools
import binascii 
import pprint 
def grab_block(block_num):
    req_url = f"https://api.blockchair.com/bitcoin/raw/block/{str(block_num)}"

    res = requests.get(req_url) 
    if res.status_code == 200:
        try:
            block_dict      = json.loads(res.text)['data'].values().__iter__().__next__()['decoded_raw_block']
            return block_dict
        except AttributeError:
            block_dict      = json.loads(res.text)['data']
            input(block_dict)
    else:
        print(res.status_code)


def decode_txn(txn_hash):
    print(txn_hash)
    req_url = f"https://api.blockchair.com/bitcoin/raw/transaction/{txn_hash}"
    res = requests.get(req_url) 
    if res.status_code == 200:
        txn      = json.loads(res.text)['data'].values().__iter__().__next__()['decoded_raw_transaction']
        pprint.pp(txn)
        return txn
    else:
        print(res.status_code)


if __name__ == "__main__":
    block_dict = grab_block(700112)
    decode_txn(block_dict['tx'][6])


    