# AUTO-GENERATED FROM JUPYTER NOTEBOOKS
# coding: utf-8

# # [Chefkoch.de](http://www.chefkoch.de/)
# ------
# 
# ## Ziel: 
# ### Scraping der Hauptrezeptesammlung von über 300'000 verschiedenen Rezepten (2.Teil)

# In[2]:


import datetime as dt
import csv
from bs4 import BeautifulSoup
from lxml import html
from datetime import datetime
from multiprocessing import Pool
from random import uniform, choice
from time import sleep, time
import requests
import os.path
import re
import os


# In[3]:


DATAST_FOLDER = 'input/test/'
IMGS_FOLDER  = 'input/images/search_thumbnails/'
DFILE_NAME    = 'recipe_details_' + dt.datetime.now().strftime('%d-%m-%Y') + '.csv'
PIC_LIST      = 'pic_list_' + dt.datetime.now().strftime('%d-%m-%Y') + '.csv'


# #### Diesmal ist das Ziel die Zutatenliste, die Zubereitung, die Tags und alle Bilder jedes einzelnen Rezeptes zu laden.
# 
# ### Beispiel:

# In[3]:


from IPython.display import Image
PATH = "/Users/Muriz/Desktop/"
Image(filename = PATH + "dPO2Ot8.png", width='100%', height=140)


# Dieses Rezept hat die Tags **Backen** und **Brot oder Brötchen**.
# Ausserdem hat es 31 weitere Bilder die wir noch brauchen.

# In[4]:


def get_list_of_recipes():
    recipe_links = []
    chef_file = DATAST_FOLDER + 'chefkoch_rezepte_27-12-2017.csv'
    with open(chef_file, 'r') as f:
        chefkoch = csv.reader(f)
        for row in chefkoch:
            try:
                recipe_links.append(row[-2])
            except: 
                continue 
    return(recipe_links)

def get_list_of_scraped_recipes():
    recipe_links = []
    recipes_file = DATAST_FOLDER + DFILE_NAME
    if os.path.isfile(recipes_file):
        with open(recipes_file, 'r') as f:
            chefkoch = csv.reader(f)
            for row in chefkoch:
                try:
                    recipe_links.append(row[0])
                except: 
                    continue
    return(recipe_links)

def list_to_scrape():
    print('AUFGERUFEN')
    l_scraped = get_list_of_scraped_recipes()
    l_all     = get_list_of_recipes()
    
    l_to_scrape = list(set(l_all) - set(l_scraped))
    print('Difference of sets:', len(l_to_scrape))
    
    return(l_to_scrape)


# In[5]:


recipe_links = list_to_scrape()


# In[6]:


def write_recipe_details(data):
    dpath = DATAST_FOLDER + DFILE_NAME
    with open(dpath, 'a', newline='') as f:
        writer = csv.writer(f)
        try:
            writer.writerow((data['link'],
                             data['ingredients'],
                             data['zubereitung'],
                             data['tags'],
                             data['gedruckt:'],
                             data['n_pics']
                             #data['reviews'],
                             #data['gespeichert:'],
                             #data['Freischaltung:'],
                             #data['author_registration_date'],
                             #data['author_reviews']
                            ))
        except:
            writer.writerow('')


# ### Die Links zu den Bildern werden in einer Liste gespeichert und seperat runtergeladen.

# In[7]:


def write_picture_list(pics):
    dpath = DATAST_FOLDER + PIC_LIST
    with open(dpath, 'a', newline='') as f:
        writer = csv.writer(f)
        try:
            writer.writerow(pics)
        except:
            writer.writerow('')


# In[8]:


def get_stats(soup):
    stats = {}
    # DROPPED
    # anzahl bewertungen
    reviews = soup.find('div', id="recipe__rating").find('span', class_ = "rating__total-votes m-r-s").text
    #stats['reviews'] = reviews
    
    # andere statistiken
    if reviews != '(0)':
        stats_table = soup.find('div', id="recipe-statistic").find_all('table')[1].find_all('tr')[1:5]
    else:
        stats_table = soup.find('div', id="recipe-statistic").find_all('table')[0].find_all('tr')[1:5]
        
    for tr in stats_table:
        stat_name  = tr.find_all('td')[0].text.strip()
        stat_value = tr.find_all('td')[1].text.strip().split(' ')[0]
        stats[stat_name] = stat_value
    return(stats)

def get_n_pictures(html_text):
    tree = html.fromstring(html_text)
    # alle bilder links
    links_pics = tree.xpath('//div[contains(@id, "slider")]//img[@class="slideshow-image lazyload"]/@src')
    return links_pics

def get_zubereitung(soup):
    zuber = soup.find('div', id="rezept-zubereitung").text.strip()
    zuber = zuber.replace('\n', ' ').replace('\r', '')
    return zuber

def get_ingredients(soup):
    # liste von zubereitungen
    ingredient_list = []
    amounts_ingredients = soup.find('table', class_="incredients").find_all('tr')
    
    for tr in amounts_ingredients:
        am = tr.find_all('td')[0].text.strip() # amount
        td = tr.find_all('td')[1].text.strip().split(',')[0] # ingredient
        td = re.sub('\(.*?\)','', td)
        ingr = am + '@' + td
        ingredient_list.append(ingr)
        
    return(str.join(',', ingredient_list))

def get_tags(soup):
    tags = []
    tag_cloud = soup.find('ul', class_ = 'tagcloud').find_all('li')
    for li in tag_cloud:
        tags.append(li.find('a').text.strip())
        
    return(str.join(',', tags))

# DROPPED
#def get_author_info(soup):
#    author = {}
#    author['author_registration_date'] = soup.find('div', class_="user-details").find('p').find('br').previous_sibling.strip()
#    author['author_reviews'] = soup.find('div', class_="user-details").find('p').find('br').next_sibling.strip()
#    return(author)


# In[9]:


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


# In[10]:


failed_urls = []


# In[11]:


def get_html(url):
    i = 5
    while i > 0:
        try:
            page = requests.get(url, headers=random_headers())
            if page.status_code != requests.codes.ok:
                page.raise_for_status()
            else:
                print(url)
                return page.text        
        except requests.exceptions.RequestException as e:
            print("Could not fetch " + url)
            print(e)
            # sichere url zu einer Liste von fehlgeschlagenen Links
            failed_urls.append(url)
            sleep(5)
            i = i - 1
            continue

    print("Link zum 5. Mal nicht abrufbar " + url + " ,abbrechen")
    return None


# In[ ]:


def get_recipe_info(url):
    sleep_time = uniform(2, 4)
    sleep(sleep_time)
    
    html_text = get_html(url)
    try:
        soup = BeautifulSoup(html_text, 'lxml')
        
        # tags ingredients preparation
        ingredient_list = get_ingredients(soup)
        tags = get_tags(soup)
        prep = get_zubereitung(soup)
        
        # pics
        list_pics = get_n_pictures(html_text)
        n_pics = len(list_pics)
        
        # update dictionary
        stats = get_stats(soup)
        #author_info = get_author_info(soup)
        
        # write dictionary
        data = {'link' : url,
                'ingredients' : ingredient_list,
                'zubereitung' : prep,
                'tags': tags, 
                'n_pics': n_pics}
        data.update(stats)
        #data.update(author_info)
        
    except:
        data = ''
        list_pics = ['error']
    
    # write file 
    write_recipe_details(data)
    write_picture_list(list_pics)


# In[ ]:


start_time = datetime.now()
print(start_time)
with Pool(15) as p:
    p.map(get_recipe_info, recipe_links)
print("--- %s seconds ---" % (datetime.now() - start_time))
failed_urls


# #### 15'360 Rezepte wurden pro Stunde in *recipe_details_27-12-2017.csv* heruntergeladen. Gesamte Laufzeit: 20h 36min
# #### Die cvs Datei enthält nun die sehr wichtigen Daten

# In[100]:


DFILE_NAME    = 'recipe_details_' + '27-12-2017' + '.csv'
PIC_LIST      = 'pic_list_' + '27-12-2017' + '.csv'

import pandas as pd
recipe_details = pd.read_csv(DATAST_FOLDER+DFILE_NAME, header=None)
recipe_details.head() # erste 5 zeilen


# # Herunterladen der Bilder
# 
# Bis jetzt sind nur die Thumbnails heruntergeladen. Die Grösse beträgt 164x140. Zu finden sind die Thumbnails unter input/images/search_thumbnails. Beispiel:

# In[171]:


from IPython.display import Image
PATH = "input/images/search_thumbnails/"
Image(filename = PATH + "recipe-1742451283182013-0.jpg", width=164, height=140)


# Chefkoch erlaubt seinen Nutzern Bilder vom Rezept hochzuladen. Oftmals ist es so, dass es mehr als ein Bild pro Rezept gibt. Um möglichst viele Daten zu haben, ladet man natürlich alle Bilder herunter. Die Links sind in der Datei *pic_list_27-12-2017.csv*

# In[6]:


import pandas as pd
pic_list = pd.read_csv('input/test/fixed.csv', header=None)
pic_list.head()


# Zuerst überprüfen wir ob die Liste leere Einträge hat, welche entfernt werden müssen.

# In[7]:


pic_list[pic_list[1].isnull()] # 2 NaN entries


# In[8]:


pic_list = pic_list.drop(pic_list.index[105484])
pic_list = pic_list.drop(pic_list.index[140099])


# In[9]:


pic_list[pic_list[1].isnull()] # No more NaN entries


# **689'651** Bilder von **107'052** Rezepten müssen noch heruntergeladen werden.
# 
# ### Bisherige Statistik: (Stand: Ende Dezember 2017)
#     Chefkoch.de enthält:
#         - Insgesamt 879'620 Bilder
#         - 316'756 Rezepte
#             - Davon enthalten 189'969 ein oder mehr Bilder
#                 - Davon enthalten 107'052 Rezepte mehr als 2 Bilder
#             - 126'787 enthalten kein Bild

# In[81]:


links = []
for index, row in pic_list.iterrows():
    link = row[1].split(',')
    if len(link) > 1: links.append(link[1:]) # ignorieren falls weniger als ein Link vorhanden

        
print(len(links))
count=0
for ll in links:
    for l in ll:
        count+=1
print(count)


# In[194]:


links[0]


# In[179]:


partial_ids = []
for link in links:
    partial_ids.append(link[0].split('/')[6]) # extrakt partial id
partial_ids[:11]


# In[144]:


title_ids = []
for link in links:
    title_ids.append(link[0].split('fix-')[1].split('.')[0])
title_ids[:5]


# In[148]:


def get_list_of_recipes_id():
    recipe_links = []
    chef_file = DATAST_FOLDER + 'chefkoch_rezepte_27-12-2017.csv'
    with open(chef_file, 'r') as f:
        chefkoch = csv.reader(f)
        for row in chefkoch:
            try:
                recipe_links.append(row[-2])
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


# In[191]:


print(len(all_ids_clean))
print(len(partial_ids))
print(len(title_ids))


# In[ ]:


get_ipython().run_cell_magic('time', '', "import json\nmatches = {} # key: recipe_id , value: pics_list\nlost = [] # sichere jede reciped_id wo es keinen match finden konnte => kein match\nfor i in range(len(title_ids)-1):\n    match = [s for s in all_ids_clean if title_ids[i] in s and s.startswith(partial_ids[i])]\n    if len(match) == 0: # kein match\n        lost.append(i)\n        continue\n    r_id = match[0].split('/')[0]\n    matches[r_id] = links[i]\n\nwith open('input/test/matches.txt', 'w') as file:\n    file.write(json.dumps(matches))\n#print(matches)\nprint('Lost:', len(lost))\nprint(lost)")


# ## Download von 630'770 Bildern

# In[264]:


from datetime import datetime
from time import sleep
import urllib.request
from multiprocessing import Pool
from random import uniform
import os


# In[265]:


# beispiel: recipe-2841831436245733-1.jpg
IMG_PATH = 'input/images/search_images/'


# In[266]:


def get_img_from_dict(i_dict):
    for idx, linn in enumerate(i_dict[1]):
        try:
            img_name = IMG_PATH + 'recipe-' + str(i_dict[0]) + '-' + str(idx+1) + '.jpg'
            urllib.request.urlretrieve(linn, img_name)
        except:
            print('Error:', i_dict[0])
    sleep_time = uniform(2., 3.) # schone den server mit anfragen
    sleep(sleep_time)


# In[ ]:


start_time = datetime.now()
print(start_time)

with Pool(20) as p:
    p.map(get_img_from_dict, matches.items())

print("--- %s seconds ---" % (datetime.now() - start_time))


# Nach fast 5 Stunden sind alle Bilder unter /input/images/search_images/ zu finden. **32.36 GB**
# ### Beispiel:

# In[19]:


import os
from IPython.display import Image, display
PATH = "input/images/search_images/"
images = []
for i in range(1, 10):
    images.append(Image(filename = PATH + "recipe-378801124204434-"+str(i)+".jpg", width=420, height=280))
display(*images)

