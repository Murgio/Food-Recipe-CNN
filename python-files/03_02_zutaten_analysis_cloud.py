# AUTO-GENERATED FROM JUPYTER NOTEBOOKS
# coding: utf-8

# # [Chefkoch.de](http://www.chefkoch.de/) Maturaarbeit 2017/18
# ------
# 
# ## Ziel: 
# ### Kurze Analyse der Zutaten von 316'755 Rezepten mit dem [APRIORI](https://en.wikipedia.org/wiki/Apriori_algorithm) Algorithmus (Teil 3.1)

# In[1]:


get_ipython().system('pip install mlxtend')


# In[2]:


from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import OnehotTransactions
import pandas as pd
import csv
import re


# Die Zutaten sind in der Datei *recipe_details_merged.csv* gespeichert. Als erstens holt man die Zutaten in eine Liste

# In[3]:


def get_list_of_ingredients():
    recipe_links = []
    chef_file = '/input/recipe_details_merged.csv'
    with open(chef_file, 'r', encoding='utf-8') as f:
        chefkoch = csv.reader(f)
        for row in chefkoch:
            try:
                recipe_links.append(row[-2])
            except:
                print('MISSED')
                continue 
    return(recipe_links)


# In[4]:


zutaten = []
for zutat in get_list_of_ingredients()[1:]:
    zzz = zutat.split(',')
    z_liste = []
    for zz in zzz:
        z = zz.split('@')
        z_liste.append(z[-1])
    zutaten.append(z_liste)


# Die Liste *zutaten* hat für jedes Rezept nun eine Liste der Zutaten

# In[5]:


print(zutaten[:4])


# ### Insgesamt teilen sich alle Rezepte 3'248'846 Zutaten

# In[6]:


count = 0
for i in zutaten:
    count += len(i)
print(count)


# In[7]:


seen = set()
uniq = []
duplicates = []
for x in zutaten:
    for zutat in x:
        if zutat not in seen:
            uniq.append(zutat)
            seen.add(zutat)
        else: duplicates.append(zutat)


# ### Von diesen 3'248'846 Zutaten sind 63'588 verschieden

# In[8]:


print(len(uniq))
print(len(duplicates))


# In der Zutatenliste entfernt man alles ausser Buchstaben

# In[9]:


# remove everything but letters
clean_zutaten = []
regex = re.compile(r'[^A-Za-zäöüßéàèêëïùâîûç]+', re.IGNORECASE)

def get_n_statistic(zutat_name):
    count_ = 0
    for zutatliste in zutaten:
        temp = []
        for zutat in zutatliste:
            new_name = regex.sub(' ', zutat)
            if new_name.strip().lower().startswith(zutat_name):
                count_ += 1
                seperat = new_name.split(' und ')
                temp.append(seperat[0])
                temp.append(seperat[1])
                continue
            if len(new_name) > 1: temp.append(new_name)
        clean_zutaten.append(temp)
    print('Count von {} ist {}'.format(zutat_name, count_))


# In[10]:


get_n_statistic('salz und pfeffer')


# In[11]:


len(clean_zutaten)


# Der Apriori Algorithmus braucht schlichtwegs zu viel RAM (>20GB) deshalb nehme ich einen zwölftel von allen Zutaten. EDIT: Das Script wurde in der Cloud neu berechnet mit sarker Hardware und 56GB RAM

# In[12]:


sub_clean_zutaten = clean_zutaten[80000:160000] # 80'000 subset


# ### Entferne leere Einträge

# In[15]:


all_clean_zutaten = sorted(sub_clean_zutaten)[:]


# In[16]:


all_clean_zutaten


# Umwandlung der Zutaten in ein one-hot encoded Pandas DataFrame

# In[17]:


get_ipython().run_cell_magic('time', '', 'oht = OnehotTransactions()\noht_ary = oht.fit(all_clean_zutaten).transform(all_clean_zutaten)\ndf = pd.DataFrame(oht_ary, columns=oht.columns_)')


# In[19]:


df.drop(df.columns[[0, 1]], axis=1, inplace=True) # drop EL and TL
df.head(10)


# Apriori liefert ein DataFrame in welchem man sieht, welche Zutaten in Kombinationen mit anderen Zutaten wie oft insgesamt vorkommen. Zum Beispiel: 
# - 57.8 Prozent in allen Rezepten kommt Salz vor.
# - 39.8 Prozent in allen Rezepten kommt Salz **und** Pfeffer vor.

# In[20]:


get_ipython().run_cell_magic('time', '', "frequent_ingr = apriori(df, min_support=0.04, use_colnames=True)\nfrequent_ingr['length'] = frequent_ingr['itemsets'].apply(lambda x: len(x))")


# In[21]:


frequent_ingr.sort_values(by='support', ascending=False)


# ## Welche Tuples von Zutaten kommen am meisten vor ?

# In[22]:


frequent_ingr[(frequent_ingr['length'] == 2) & (frequent_ingr['support'] >= 0.125)].sort_values(by='support', ascending=False)


# ## Welche Triplets von Zutaten kommen am meisten vor ?

# In[23]:


frequent_ingr[(frequent_ingr['length'] == 3) & (frequent_ingr['support'] >= 0.08)].sort_values(by='support', ascending=False)


# ## Quadruplets ?

# In[24]:


frequent_ingr[(frequent_ingr['length'] == 4) & (frequent_ingr['support'] >= 0.04)].sort_values(by='support', ascending=False)


# ## Quintuplets ??

# In[25]:


frequent_ingr[(frequent_ingr['length'] == 5) & (frequent_ingr['support'] >= 0.04)].sort_values(by='support', ascending=False)

