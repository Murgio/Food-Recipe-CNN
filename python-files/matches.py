# AUTO-GENERATED FROM JUPYTER NOTEBOOKS
# coding: utf-8

# In[1]:


get_ipython().system("echo 'tqdm\\nPyDrive\\npython-levenshtein' > requirements.txt")
get_ipython().system('pip install -r requirements.txt')


# In[ ]:


import random
import numpy as np
import pandas as pd
import _pickle as pickle
from pprint import pprint

import matplotlib.pyplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')

from tqdm import tqdm

import os
from os import listdir
from os.path import isfile, join
import shutil
import stat
import collections
from collections import defaultdict

from google.colab import files
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import psutil
from tensorflow.python.client import device_lib


# In[4]:


uploaded = files.upload()

with open("cookies.txt", 'wb') as f:
    f.write(uploaded[list(uploaded.keys())[0]])


# In[5]:


get_ipython().system('curl -b cookies.txt https://kantiolten-my.sharepoint.com/personal/muriz_serifovic_kantiolten_ch/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fmuriz%5Fserifovic%5Fkantiolten%5Fch%2FDocuments%2FMachine%20Learning%2Fmeta%2Fchefkoch%2Ecsv > chefkoch.csv')
get_ipython().system('curl -b cookies.txt https://kantiolten-my.sharepoint.com/personal/muriz_serifovic_kantiolten_ch/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fmuriz%5Fserifovic%5Fkantiolten%5Fch%2FDocuments%2FMachine%20Learning%2Fmeta%2Flinks%2Ecsv > links.csv')


# In[4]:


import pandas as pd
pic_list = pd.read_csv('links.csv', header=None)
pic_list.head()


# In[ ]:


pic_list = pic_list.drop(pic_list.index[105484])
pic_list = pic_list.drop(pic_list.index[140099])


# In[23]:


links = []
for index, row in pic_list.iterrows():
    link = row[1].split(',')
    links.append(link)

        
print(len(links))
count=0
for ll in links:
    for l in ll:
        count+=1
print(count)


# In[ ]:


import Levenshtein

def similar(a, b):
  return Levenshtein.ratio(a, b)
  
def remove_values_from_list(the_list, val):
  return [value for value in the_list if value[0] != val]


# In[ ]:


links = remove_values_from_list(links, 'error')


# In[26]:


error = []
partial_ids = []
for link in links[:]:
    try:
        partial_ids.append(link[0].split('/')[6]) # extrakt partial id
    except (IndexError):
        error.append(link)
partial_ids[:11]


# In[55]:


title_ids = []
for link in links:
    try:
        title_ids.append(link[0].split('fix-')[-1].split('.')[0])
    except (IndexError):
        error.append(link)
title_ids[:5]


# In[56]:


import csv
def get_list_of_recipes_id():
    recipe_links = []
    chef_file = 'chefkoch.csv'
    with open(chef_file, 'r') as f:
        chefkoch = csv.reader(f)
        for row in chefkoch:
            try:
                recipe_links.append(row[-4])
            except:
                print('ERROR')
                continue 
    return(recipe_links)

all_ids = get_list_of_recipes_id()
all_ids_clean = []
for id in all_ids[1:]: # erste spalte überspringen, es ist der spalten name
    all_ids_clean.append(id[32:].lower()) # 'recipe-' extrahieren
del all_ids[:]
all_ids_clean[:10]


# In[59]:


print(all_ids_clean[:10])
print(title_ids[:10])
print(partial_ids[:10])


# In[69]:


import json
matches = {} # key: recipe_id , value: pics_list
for i, n in enumerate(partial_ids):
  match = [ii for ii in all_ids_clean if ii.startswith(n)]
  highest_ratio = 0.0
  favs = []
  for x in match:
    name = x[:-5].split('/')[1]
    ratio = similar(title_ids[i], x)
    if ratio > highest_ratio:
      highest_ratio = ratio
      favs.append(x)
  try:
    r_id = favs[-1].split('/')[0]
    matches[r_id] = links[i]
  except:
    print(match, favs, i, n)
  
  if i % 1000 == 0:
    print(i)
  
  
with open('matches_v3.txt', 'w') as file:
    file.write(json.dumps(matches))


# **match, favs, i, n**
# 
# ['1712011280048532/fix-kirschgruetze-mit-eiskaffee-sahne.html', '171201074240691/limettenschnitten.html'] [] 26215 171201
# 
# ['925441197878752/fix-torte.html'] [] 87383 92544
# 
# ['2065361333924671/fix-und-feierabend-pfanne-mit-gnocchi.html'] [] 130520 206536

# In[ ]:


# Authenticate and create the PyDrive client.
# This only needs to be done once per notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

textfile = drive.CreateFile()
textfile.SetContentFile('matches_v3.txt')
textfile.Upload()

