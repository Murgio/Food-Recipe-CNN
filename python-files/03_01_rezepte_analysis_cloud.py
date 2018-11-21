# AUTO-GENERATED FROM JUPYTER NOTEBOOKS
# coding: utf-8

# # [Chefkoch.de](http://www.chefkoch.de/) Maturaarbeit 2017/18
# ------
# 
# ## Ziel: 
# ### Analyse der Hauptrezeptesammlung von über 300'000 verschiedenen Rezepten (3.Teil)

# In[11]:


get_ipython().system('pip install seaborn')


# In[3]:


# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('fivethirtyeight')
import matplotlib.pylab as pylab
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6


# # Bereinigung
# 
# 1. Die Spalte *average_rating* soll nur noch die letzte Zahl enthalten. "n Bewertungen - Ø" kann gelöscht werden da die Anzahl Bewertungen in der Spalte *votes* gespeichert ist.
# 2. Von der Spalte *stars_shown* kann der substring 'star-' gelöscht werden. Danach in Typ Category umwandeln.
# 3. Die Klammern sind nicht von Gebrauch in *votes*.
# 4. *difficulty* kann in eine Category umgewandelt werden.
# 5. *preparation_time*: substring ' min.' kann gelöscht werden.
# 6. *date* kann in datetime umgewandelt werden.
# 7. *has_picture* kann in eine Category umgewandelt werden.
# 8. Bereinige 'recipe-' von *recipe_id*
# 9. Die Spalte *link* ist nicht von Bedeutung für Data Analysis und wird entfernt.

# In[ ]:


# 1
rezepte['average_rating'] = rezepte['average_rating'].apply(lambda x: x.replace(" Bewertungen - Ø ", ":"))
rezepte['average_rating'] = rezepte['average_rating'].apply(lambda x: x.replace(" Bewertung - Ø ", ":"))
rezepte['average_rating'].replace(to_replace='(.*?):', value='',inplace=True,regex=True)
rezepte['average_rating'] = rezepte['average_rating'].astype('float64')

# 2
rezepte['stars_shown'] = rezepte['stars_shown'].apply(lambda x: x.replace("star-", ""))
rezepte['stars_shown'] = rezepte['stars_shown'].astype('category')

# 3
rezepte['votes'] = rezepte['votes'].str.extract('(\d+)')
rezepte['votes'] = rezepte['votes'].astype('float64')

# 4
rezepte['difficulty'] = rezepte['difficulty'].astype('category')

# 5
rezepte['preparation_time'] = rezepte['preparation_time'].apply(lambda x: x.replace(" min.", "")).astype('float64')

# 6
rezepte['date'] = pd.to_datetime(rezepte['date'])

# 7
rezepte['has_picture'] = rezepte['has_picture'].astype('category')

# 8
rezepte['recipe_id'] = rezepte['recipe_id'].apply(lambda x: x.replace("recipe-", ""))

# 9
rezepte.drop(rezepte.columns[[-2]], axis=1, inplace=True)


# In[4]:


rezepte = pd.read_csv('/input/chefkoch_rezepte_analysis.csv')


# In[7]:


rezepte.head()


# In[9]:


rezepte.info()


# In[5]:


rezepte.drop(rezepte.columns[[0]], axis=1, inplace=True)
rezepte.drop_duplicates(subset=['recipe_id'], keep='first', inplace=True)
rezepte.head()


# In[109]:


rezepte.recipe_id.value_counts()[:20]


# In[7]:


rezepte['date'] = pd.to_datetime(rezepte['date'])
rezepte['year'] = rezepte['date'].map(lambda x: x.year)


# In[8]:


rezepte.head(2)


# In[9]:


from pylab import rcParams
rcParams['figure.figsize'] = 16, 8


# In[10]:


#!pip install statsmodels
get_ipython().system('pip install ggplot')


# In[105]:


import pandas as pd
import re
import statsmodels as sm


from ggplot import *
theme_bw()

import matplotlib.pyplot as mpl
mpl.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[28]:


rezepte.groupby('year').count().plot(kind='bar', color='darkgreen', title="Anzahl Rezepte hinzugefügt pro Jahr", legend=None).set_xlim(0)


# In[12]:


rezepte['YearMonth'] = rezepte['date'].map(lambda x: 1000*x.year + x.month)


# In[34]:


rezepte['YearMonth'].head()


# In[45]:


fig = plt.figure()
rezepte.groupby('YearMonth').count().plot(kind='bar', color='darkgreen', title="Anzahl Rezepte hinzugefügt pro Jahr", legend=None).set_xlim(0)
rezepte.groupby('year').count().plot(color='darkgreen', title="Anzahl Rezepte hinzugefügt pro Jahr", legend=None)
#ax1 = plt.axes()
#x_axis = ax1.axes.get_xaxis()
#x_axis.set_visible(False)
#for label in x_axis.get_ticklabels()[::2]:
#    label.set_visible(False)
plt.show()


# In[146]:


rezepte.sort_values(by='year', ascending=True).groupby('year').count().cumsum(0).plot(kind='bar', color='darkgreen', title='Gesamte Anzahl an Rezepten', legend=None).set


# In[91]:


rezepte['new_rating'] = [rezepte['average_rating'][i] if x > 0 else 0 for i, x in enumerate(rezepte['votes'])]


# In[112]:


rezepte.plot(kind='hist', y='new_rating',
        color='darkgreen', alpha=0.7, legend=None,
        title='Distribution der Bewertungen').set_xlim(0)


# In[113]:


rezepte.plot(kind='hist', y='votes', 
        bins=100, logy=True,
        legend=None, color='darkgreen', alpha=0.7).set_xlim(0)


# In[125]:


from itertools import cycle, islice
my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, 10))


# In[153]:


rezepte[['votes','difficulty','year']].groupby(['year','difficulty']).count().unstack().plot(kind='bar',colormap='RdYlGn',stacked=True).set_xlim(0)


# In[130]:


rezepte.head()


# In[134]:


from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")
# Compute the correlation matrix
corr = rezepte.drop(['new_rating', 'recipe_id'], 1).corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

# heatmap with mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[169]:


def plot_correlation_map(df):
    corr = rezepte.drop(['new_rating', 'recipe_id'], 1).corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ))
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 })


# In[170]:


plot_correlation_map(rezepte)


# In[155]:


import seaborn as sns; sns.set(style='ticks', color_codes=True)
g = sns.pairplot(rezepte.drop(['new_rating', 'recipe_id'], 1))


# In[166]:


f = sns.pairplot(rezepte[['average_rating', 'votes']], size=6)


# In[168]:


f = sns.pairplot(rezepte[['preparation_time', 'votes']], size=6)

