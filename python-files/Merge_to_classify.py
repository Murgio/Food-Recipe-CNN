# AUTO-GENERATED FROM JUPYTER NOTEBOOKS
# coding: utf-8

# In[75]:


import pandas as pd


# In[76]:


cr = 'test/chefkoch_rezepte_27-12-2017.csv'
rd = 'test/recipe_details_27-12-2017.csv'


# # Chefkoch Rezepte

# In[77]:


chefkoch_rezepte = pd.read_csv(cr)


# In[86]:


chefkoch_rezepte.drop(chefkoch_rezepte.columns[-1], axis=1, inplace=True) # 0, 2, 2, -1


# In[87]:


chefkoch_rezepte.head()


# In[88]:


chefkoch_rezepte['recipe_id'] = chefkoch_rezepte['recipe_id'].str.split('-').str[1].str.strip()


# In[93]:


chefkoch_rezepte.head()


# In[92]:


chefkoch_rezepte['recipe_id'] = chefkoch_rezepte['recipe_id'].astype('int64')


# In[105]:


chefkoch_rezepte.info()


# In[118]:


chefkoch_rezepte[chefkoch_rezepte.duplicated(['recipe_id'], keep='first')]


# In[121]:


chefkoch_rezepte.drop_duplicates(inplace=True)


# In[137]:


chefkoch_rezepte.head()


# # Chefkoch Details

# In[94]:


chefkoch_details = pd.read_csv(rd, header=None)
chefkoch_details.head()


# In[97]:


chefkoch_details.drop(chefkoch_details.columns[-1], axis=1, inplace=True) # -1, -1


# In[99]:


chefkoch_details[chefkoch_details.columns[0]] = chefkoch_details[chefkoch_details.columns[0]].str.split('/').str[4].str.strip()


# In[138]:


chefkoch_details.head()


# In[100]:


chefkoch_details[chefkoch_details.columns[0]] = chefkoch_details[chefkoch_details.columns[0]].astype('int64')


# In[106]:


chefkoch_details.columns = ['recipe_id', 'ingredients', 'instructions', 'tags']


# # Merging both DataFrames

# In[142]:


chefkoch_details.merge(chefkoch_rezepte, on="recipe_id", how = 'inner').sort_values(by=['recipe_id']).equals(chefkoch_rezepte.merge(chefkoch_details, on="recipe_id", how = 'inner').sort_values(by=['recipe_id']))


# In[148]:


chefkoch_final = chefkoch_rezepte.merge(chefkoch_details, on="recipe_id", how = 'inner')


# In[151]:


chefkoch_final.to_csv('chefkoch_merged_lists_05_may.csv')

