# AUTO-GENERATED FROM JUPYTER NOTEBOOKS
# coding: utf-8

# # Applying dimensionality reduction to 820'676 image features

# In[1]:


from __future__ import division, print_function

import os
# coding=utf-8
import sys
import time
from datetime import datetime
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tables
from IPython.display import HTML, display
from tqdm import tqdm

np.set_printoptions(threshold=np.nan) # prints the whole nparray no matter the shape of it


# ## Load existing image features with pytables

# In[20]:


hdf5_file_original_1 = tables.open_file('models/vgg16_bottleneck_features.hdf5', mode='r')
features_original_1 = hdf5_file_original_1.root.img_features # Sparse data, mostly zeros
images_original_1 = hdf5_file_original_1.root.img_paths

hdf5_file_original_2 = tables.open_file('models/vgg16_bottleneck_features_02.hdf5', mode='r')
features_original_2 = hdf5_file_original_2.root.img_features # Sparse data, mostly zeros
images_original_2 = hdf5_file_original_2.root.img_paths


# #### Path for the new, third table containg merged image features

# In[2]:


hdf5_path_pca = 'models/vgg16_bottleneck_features_PCA.hdf5'


# #### Creating a new tables file which contains merged image paths and merged image features from both preexisting files

# In[32]:


hdf5_file_pca = tables.open_file(hdf5_path_pca, mode='w')

images_original_3 = hdf5_file_pca.create_array(hdf5_file_pca.root,
                                               'img_paths', atom=images_original_1.atom,
                                               shape=(images_original_1.nrows + images_original_2.nrows,))


# #### Combining image paths from first file and second file

# In[33]:


images_original_3[:images_original_1.nrows] = images_original_1[:]
images_original_3[images_original_1.nrows:] = images_original_2[:]


# #### Flushing pending data to disk

# In[34]:


images_original_3.flush()


# ### features_pca will contain all raw image features

# In[44]:


data_shape = (0, 4096)
img_dtype = tables.Float32Atom()

features_pca = hdf5_file_pca.create_earray(hdf5_file_pca.root, 'img_features', img_dtype, shape=data_shape)


# In[45]:


features_pca.append(features_original_1.read())
features_pca.flush()

ft_2_np = features_original_2.read()
features_original_1.append(ft_2_np)
features_pca.flush()


# In[50]:


hdf5_file_pca.close()


# ##### Loading merged data

# In[4]:


hdf5_file_pca = tables.open_file(hdf5_path_pca, mode='r')


# In[5]:


features_pca = hdf5_file_pca.root.img_features
images_pca = hdf5_file_pca.root.img_paths


# In[6]:


features_pca.shape[0]


# ## Principal component analysis
# 
# The image features take up **12.52 GB** (820'676*4'096*32)/(8*1'024*1'024*1'024) which is simply to big to load it completely into ram. Instead, sklearn provides us with an altered pca implementation enabling us to calculate the eigenvalues batchwise.

# In[10]:


from sklearn.decomposition import IncrementalPCA

n = features_pca.shape[0] # how many rows we have in the dataset
chunk_size = 82000 # how many rows we feed to IPCA at a time, the divisor of n
ipca = IncrementalPCA(n_components=512, batch_size=41000)

for i in tqdm(range(0, n//chunk_size)):
    ipca.partial_fit(features_pca[i*chunk_size : (i+1)*chunk_size])


# In[16]:


ipca.partial_fit(features_pca[820000:]) # 820'000 = chunk_size*(n//chunk_size)


# Storing the IncrementalPCA(batch_size=41000, copy=True, n_components=512, whiten=False) object on disk:

# In[17]:


import pickle

pickle.dump(ipca, open('models/sklearn_ipca_object.p', 'wb'))


# Loading it into mem:

# In[2]:


import pickle
pickle_in = open("models/sklearn_ipca_object.p","rb")
ipca = pickle.load(pickle_in)


# ### Creating our final hdf5 file
# #### Path for the new, third table containg reduced image features

# In[30]:


hdf5_path_ipca = 'models/vgg16_bottleneck_features_IPCA.hdf5'


# In[21]:


hdf5_file_ipca = tables.open_file(hdf5_path_ipca, mode='w') # Create new hdf5 file

# Takes up 61.6 MB on disk
hdf5_file_ipca.create_array(hdf5_file_ipca.root, 'img_paths', images_pca.read()) # Create array for image paths

data_shape = (0, 512) # Shape is now 512!
img_dtype = tables.Float32Atom()

# Create enlargeable array for image features
features_ipca = hdf5_file_ipca.create_earray(hdf5_file_ipca.root, 'img_features', img_dtype, shape=data_shape)


# In[22]:


# PCA.transform actually returns float64 rather than float32

n = features_pca.shape[0] # how many rows we have in the dataset
chunk_size = 82000 # how many rows we feed to IPCA at a time, the divisor of n

for i in tqdm(range(0, n//chunk_size)):
    features_ipca.append(ipca.transform(features_pca[i*chunk_size : (i+1)*chunk_size]))
    
features_ipca.append(ipca.transform(features_pca[820000:])) # 820'000 = chunk_size*(n//chunk_size)


# In[23]:


features_ipca.shape


# ### Flushing and closing our pytable to disk

# In[26]:


hdf5_file_ipca.close()


# ## Preparing image features for nmslib

# In[3]:


hdf5_path_ipca = 'models/vgg16_bottleneck_features_IPCA.hdf5'
hdf5_file_ipca = tables.open_file(hdf5_path_ipca, mode='r') # Create new hdf5 file
features_ipca = hdf5_file_ipca.root.img_features
images = hdf5_file_ipca.root.img_paths


# In[4]:


import nmslib

# nmslib default params for now

# Number of neighbors
K = 18
# Set index parameters
# These are the most important ones
M = 15
efC = 100
num_threads = 4
index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post' : 0}
space_name='l2'
efS = 100
query_time_params = {'efSearch': efS}
index_ann = None


# In[5]:


def init_ann_index(bin_PATH='models/image_features_pca_nmslib_index.bin'):
    global index_ann
    # Intitialize the library, specify the space, the type of the vector and add data points 
    index_ann = nmslib.init(method='hnsw', space=space_name, data_type=nmslib.DataType.DENSE_VECTOR)
    # Re-load the index and re-run queries
    index_ann.loadIndex(bin_PATH)
    # Setting query-time parameters and querying
    print('Setting query-time parameters', query_time_params)
    index_ann.setQueryTimeParams(query_time_params)


# In[6]:


init_ann_index()


# In[41]:


def create_ann_index(bin_PATH):
    global index_ann
    # Intitialize the library, specify the space, the type of the vector and add data points 
    index_ann = nmslib.init(method='hnsw', space=space_name, data_type=nmslib.DataType.DENSE_VECTOR) 
    index_ann.addDataPointBatch(features_ipca.read())
    index_ann.createIndex(index_time_params, print_progress=True)
    index_ann.saveIndex(bin_PATH)
    # Setting query-time parameters and querying
    print('Setting query-time parameters', query_time_params)
    index_ann.setQueryTimeParams(query_time_params)


# In[42]:


create_ann_index('models/image_features_pca_nmslib_index.bin')

