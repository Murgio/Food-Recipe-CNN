# AUTO-GENERATED FROM JUPYTER NOTEBOOKS
# coding: utf-8

# # [Chefkoch.de](http://www.chefkoch.de/)
# ------
# 
# ## Ziel: 
# ### Scraping der Hauptrezeptesammlung von über 300'000 verschiedenen Rezepten (1.Teil)

# In[44]:


import os
import time
import datetime as dt
from datetime import datetime
from multiprocessing import Pool

from time import sleep, time
from random import randint, choice
import requests
import urllib.request
from bs4 import BeautifulSoup
import csv


# Daten wie Rezeptname, Bewertung, Datum vom Upload des Rezeptes, etc. werden in eine csv Datei gespeichert.
# Falls das Rezept ein Bild hat, wird das Thumbnail im Ordner **search_thumbnails** abgelegt.

# In[45]:


# OS
NOW           = dt.datetime.now()
FILE_NAME     = 'chefkoch_rezepte_' + NOW.strftime('%d-%m-%Y') + '.csv'
DATASET_FOLDER = 'input/csv_files/'
IMGS_FOLDER  = 'input/images/search_thumbnails/'

# Chefkoch.de Seite
CHEFKOCH_URL  = 'http://www.chefkoch.de'
START_URL     = 'http://www.chefkoch.de/rs/s'
CATEGORY      = '/Rezepte.html'


# Alle 300k Rezepte sortiert nach Datum: http://www.chefkoch.de/rs/s30o3/Rezepte.html

# Wenn man Website Scrapping durchführt, ist es wichtig die robots.txt Datei zu respektieren. Manche Administratoren möchten nicht das bestimmte Directories von Bots besucht werden. https://www.chefkoch.de/robots.txt liefert:
# 
# - User-agent: *  # directed to all spiders, not just Scooter
# - Disallow: /cgi-bin
# - Disallow: /stats
# - Disallow: /pictures/fotoalben/
# - Disallow: /forumuploads/
# - Disallow: /pictures/user/
# - Disallow: /user/
# - Disallow: /avatar/
# - Disallow: /cms/
# - Disallow: /produkte/
# - Disallow: /how2videos/
# 
# Aufgeführt sind Directories die uns gar nicht interessieren, weshalb man getrost weiter machen kann. Nichtsdestotrotz  sind Massnahmen, wie zufällige Headers und genügend grosse Pausen zwischen den einzelnen Requests empfehlenswert, um einen möglichen Ban der Website zu vermeiden.

# In[46]:


desktop_agents = ['Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
                 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
                 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
                 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/602.2.14 (KHTML, like Gecko) Version/10.0.1 Safari/602.2.14',
                 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
                 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
                 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
                 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
                 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
                 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0']

def random_headers():
    return {'User-Agent': choice(desktop_agents),'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}


# In[47]:


category_url = START_URL + '0o3' + CATEGORY

def _get_html(url):
    page = ''
    while page == '':
        try:
            page = requests.get(url, headers=random_headers())
        except:
            print('Connection refused')
            time.sleep(10)
            continue
    return page.text

def _get_total_pages(html):
    soup = BeautifulSoup(html, 'lxml')
    total_pages = soup.find('div', class_='ck-pagination qa-pagination').find('a', class_='qa-pagination-pagelink-last').text
    return int(total_pages)

html_text_total_pages = _get_html(category_url)
total_pages = _get_total_pages(html_text_total_pages)
print('Total pages: ', total_pages)


# Liste von allen einzelnen Rezepteurls bei Chefkoch im folgenden Format:
# 1. Seite: http&#58;//www.chefkoch.de/rs/s**0**o3/Rezepte.html
# 2. Seite: http&#58;//www.chefkoch.de/rs/s**30**o3/Rezepte.html
# 3. Seite: http&#58;//www.chefkoch.de/rs/s**60**o3/Rezepte.html
# 4. Seite: ...
# 
# Auf einer Seite erhält man 30 Rezepte. Um jede Seite aufrufen zu können, muss man nur die Zahl zwischen **s** und **o3** um 30 erhöhen.

# In[48]:


url_list = []

for i in range(0, total_pages + 1):
    url_to_scrap = START_URL + str(i * 30) + 'o3' + CATEGORY
    url_list.append(url_to_scrap)


# In[49]:


from pprint import pprint
# Die ersten 30 Seiten:
pprint(url_list[:30])


# In[53]:


def _write_to_recipes(data):
    path = DATASET_FOLDER + FILE_NAME
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow((data['recipe_id'],
                        data['recipe_name'],
                        data['average_rating'],
                        data['stars_shown'],
                        data['votes'],
                        data['difficulty'],
                        data['preparation_time'],
                        data['date'],
                        data['link'],
                        data['has_picture']))

def _get_picture_link(item):
    item_class = item.find('picture').find('img').get('class')
    if item_class == ['lazyload']:
        img_link = item.find('picture').find('img').get('data-srcset')
    else: 
        img_link = item.find('picture').find('source').get('srcset')
    return(img_link)

def _get_front_page(html):
    soup = BeautifulSoup(html, 'lxml')
    lis = soup.find_all('li', class_="search-list-item")
    
    for index, li in enumerate(lis):

        # get rezept ID
        try:
            id_r = li.get('id')
        except:
            id_r = ''

        # bild speichern falls eins verügbar
        try: 
            if li.find('picture') is not None:
                img_link = _get_picture_link(li)
                img_name = IMGS_FOLDER + str(id_r) + '.jpg'
                urllib.request.urlretrieve(img_link, img_name)
                has_pic = 'yes'
            else: 
                has_pic = 'no'
        except:
            has_pic = ''

        # link
        try:
            link = CHEFKOCH_URL + li.find('a').get('href')
        except:
            link = ''

        # name des rezeptes
        try:
            name = li.find('div', class_='search-list-item-title').text.strip()
        except:
            name = ''

        # durchschnitts bewertung von nutzern
        try:
            stars = li.find('span', class_='search-list-item-uservotes-stars').get('title')
        except:
            stars = ''

        # anzahl sterne
        try:
            stars_shown = li.find('span', class_='search-list-item-uservotes-stars').get('class')[1]
        except:
            stars_shown = ''

        # anzahl votes
        try:
            votes = li.find('span', class_='search-list-item-uservotes-count').text.strip()
        except:
            votes = ''

        # schwierigkeitsgrad des rezeptes => simpel, normal oder pfiffig
        try:
            difficulty = li.find('span', class_='search-list-item-difficulty').text.strip()

        except:
            difficulty = ''

        # zubereitungs zeit
        try:
            preptime = li.find('span', class_='search-list-item-preptime').text.strip()
        except:
            preptime = ''

        # datum
        try:
            date = li.find('span', class_='search-list-item-activationdate').text.strip()
        except:
            date = ''

        # write dictionary
        data = {'recipe_id' : id_r,
                'recipe_name' : name,
                'average_rating': stars,
                'stars_shown' : stars_shown,
                'votes' : votes,
                'difficulty' : difficulty,
                'preparation_time' : preptime,
                'has_picture' : has_pic,
                'date' : date,
                'link' : link}
        
        # append file
        _write_to_recipes(data)
        
def scrap_main(url):
    print('Current url: ', url)
    html = _get_html(url)
    _get_front_page(html)
    #sleep(randint(1, 2))


# In[54]:


start_time = time()
with Pool(1) as p:
    p.map(scrap_main, url_list[7813:])
print("--- %s seconds ---" % (time() - start_time))


# ## CSV Datei lesen und korrigieren
# 
# Unter input/csv_files/ findet man die erstellte CSV Datei.
# Grösse: 62.1 MB

# In[4]:


get_ipython().system('ls input/csv_files/')


# In[6]:


import pandas as pd
chef_rezepte = pd.read_csv('input/csv_files/chefkoch_rezepte_26-12-2017.csv', header=None)
chef_rezepte.head()


# In[22]:


chef_rezepte[[8]][:10] # erste 10 zeilen der 8. spalte


# Beim Scraping ist ein Fehler unterlaufen. Die Links müssen als https Links und nicht als http Links gespeichert werden.
# ### Korrektur:

# In[23]:


chef_rezepte[8] = chef_rezepte[8].str.replace('http', 'https')


# In[26]:


chef_rezepte[[8]][:10]


# In[28]:


chef_rezepte.columns = ['recipe_id', 'recipe_name', 'average_rating',
                        'stars_shown', 'votes', 'difficulty', 'preparation_time',
                        'date', 'link', 'has_picture']


# In[31]:


chef_rezepte.head()


# In[32]:


chef_rezepte.to_csv('input/csv_files/chefkoch_rezepte_27-12-2017.csv')


# ### Umbennenung der Thumbnails: Hänge ein 0 hintendran -> Thumbnail ist das erste Bild des Rezeptes.
# #### 1, 2, 3, ..., n Bilder kommen im Teil 2 dazu.

# In[ ]:


import os
path = 'input/images/search_thumbnails/'
files = os.listdir(path)
i = 1
for file in files:
    filename, file_extension = os.path.splitext(file)
    os.rename(os.path.join(path, file), os.path.join(path, filename + '-0' + file_extension))
    print('renamed: ', i)
    i = i+1


# # Weiter gehts mit Teil 2: 02_rezepte_details.ipynb
