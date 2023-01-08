from datetime import datetime
from time import sleep
import urllib.request
from multiprocessing import Pool
from random import uniform
import os
import json

IMG_PATH = 'input/images/search_images/'

def get_img_from_dict(i_dict):
    for idx, linn in enumerate(i_dict[1]):
        try:
            img_name = IMG_PATH + 'recipe-' + str(i_dict[0]) + '-' + str(idx+1) + '.jpg'
            urllib.request.urlretrieve(linn, img_name)
        except:
            print('Error:', i_dict[0])
    sleep_time = uniform(2., 3.) # schone den server mit anfragen
    sleep(sleep_time)

start_time = datetime.now()
print(start_time)

if __name__ == '__main__':

    with open('input/test/matches.txt','r') as f:
        matches = json.load(f)
    
    with Pool(20) as p:
        p.map(get_img_from_dict, matches.items())

    print("--- %s seconds ---" % (datetime.now() - start_time))