# AUTO-GENERATED FROM JUPYTER NOTEBOOKS
# coding: utf-8

# ## Find out which model predicts with highest accuracy.
# 
# #### 28.10.2018

# # Download and prepare models and dataset for testing

# In[8]:


########################### KNOW RAM AND GPU MEMORY ############################
# Thanks to: Stas Bekman
# https://stackoverflow.com/questions/48750199/google-colaboratory-misleading-information-about-its-gpu-only-5-ram-available#

# memory footprint support libraries/code
get_ipython().system('ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi')
get_ipython().system('pip install gputil')
get_ipython().system('pip install psutil')
get_ipython().system('pip install humanize')
import psutil
import humanize
import os
import GPUtil as GPU
GPUs = GPU.getGPUs()
# XXX: only one GPU on Colab and isn’t guaranteed
gpu = GPUs[0]
def printm():
  process = psutil.Process(os.getpid())
  print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " I Proc size: " + humanize.naturalsize( process.memory_info().rss))
  print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
printm()


# In[10]:


get_ipython().system('ps ax | grep python')


# In[ ]:


get_ipython().system('kill -9 8681')


# In[1]:


get_ipython().system('ls')


# In[2]:


get_ipython().system('pip install PyDrive')


# In[ ]:


from google.colab import files
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


# In[4]:


uploaded = files.upload()

with open("models_result_40409_imgs.txt", 'wb') as f:
    f.write(uploaded[list(uploaded.keys())[0]])


# In[ ]:


get_ipython().system('apt-get upgrade plowshare')


# In[ ]:


get_ipython().system('apt-get install libssl-dev gcc autoconf automake pkg-config git')


# In[10]:


get_ipython().system('openssl version -a')


# In[ ]:


get_ipython().system('apt-get install sudo')
get_ipython().system('apt-get install libcurl4-gnutls-dev')


# In[ ]:


get_ipython().system("echo 'wget http://archive.ubuntu.com/ubuntu/pool/universe/m/megatools/megatools_1.10.2.orig.tar.gz && tar -xzvf megatools_1.10.2.orig.tar.gz && cd megatools-1.10.2 && ./configure && make && make install' > run_4.sh")


# In[ ]:


get_ipython().system('sudo apt-get upgrade asciidoctor')


# In[ ]:


get_ipython().system('apt-get --no-install-recommends install asciidoc -y')


# In[ ]:


get_ipython().system('bash run_4.sh')


# In[12]:


get_ipython().system('ls')


# In[13]:


megadl 'https://mega.nz/#!'


# In[14]:


megadl 'https://mega.nz/#!'


# In[15]:


get_ipython().system('ls megatools-1.10.2')


# In[ ]:


get_ipython().system('mkdir test')
get_ipython().system('mkdir model_candidates')


# In[ ]:


get_ipython().system('cd megatools-1.10.2 && unzip -q test.zip -d ../test/')


# In[ ]:


get_ipython().system('cd megatools-1.10.2 && unzip -q model_candidates.zip -d ../model_candidates/')


# In[19]:


get_ipython().system('find test/test -print | wc -l')
get_ipython().system('find test/test -name "*.jpg" -size -1k -delete')


# In[20]:


get_ipython().system('find test/test -print | wc -l')


# # Test every single model against our test dataset 

# In[21]:


import os

# Model candidates
candidates = sorted([x for x in os.listdir('model_candidates/model_candidates') if not x.startswith('.')])
candidates


# In[22]:


from os import listdir
from os.path import isfile, join
import shutil
import stat
import collections
from collections import defaultdict
import random
import numpy as np
import math
import json

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')

#import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.imagenet_utils import preprocess_input as preprocess_input_vgg
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inc

from keras import metrics
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import keras.backend as K

import h5py


# In[ ]:


root_dir = 'test/test/'
rows = 17
cols = 6
fig, ax = plt.subplots(rows, cols, frameon=False, figsize=(15, 25))
fig.suptitle('Random Image from Each Food Class', fontsize=20)
sorted_food_dirs = sorted(os.listdir(root_dir))[1:] # ignore .DS_Store Mac file
for i in range(rows):
    for j in range(cols):
        try:
            food_dir = sorted_food_dirs[i*cols + j]
        except:
            break
        all_files = os.listdir(os.path.join(root_dir, food_dir))
        rand_img = np.random.choice(all_files)
        img = plt.imread(os.path.join(root_dir, food_dir, rand_img))
        ax[i][j].imshow(img)
        ec = (0, .6, .1)
        fc = (0, .7, .2)
        ax[i][j].text(0, -20, food_dir, size=10, rotation=0,
                ha="left", va="top", 
                bbox=dict(boxstyle="round", ec=ec, fc=fc))
plt.setp(ax, xticks=[], yticks=[])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# In[ ]:


get_ipython().system('find test/test -print | wc -l ')


# In[ ]:


get_ipython().system('find test -name "*.jpg" -size -1k -delete')


# In[ ]:


def top_10_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=20)


# In[ ]:


def eval_models(candidates):
    '''
        Evaluate all trained models in the directory model_candidates.
        Returns: dictionary with model name as key and accuracy, top_k_accuracy
            and loss in a list as value.
    '''
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
            'test/test',  # this is the target directory
            target_size=(299, 299),
            batch_size=256,
            #subset='test',
            shuffle = True,
            class_mode='categorical')
    models_result = {}
    dir_candidates = 'model_candidates/model_candidates/'
    first_vgg_switch = True
    for candidate in candidates:
        model_path = dir_candidates+candidate
        # VGG-16
        if candidate.startswith('w') and first_vgg_switch:
            # Create new test generator with correct target size
            test_generator = test_datagen.flow_from_directory(
                'test/test',  # this is the target directory
                target_size=(224, 224),
                batch_size=128,
                subset='training',
                shuffle = True,
                class_mode='categorical')
            first_vgg_switch = False
        print('Testing ', candidate)
        result = eval_model(model_path, test_generator)
        print('Result: ', result)
        models_result[candidate] = result
        with open('models_result.txt', 'w') as file:
          file.write(json.dumps(models_result)) # use `json.loads` to do the reverse
    return models_result


# In[ ]:


def eval_model(MODEL_PATH, test_generator):
    '''
        Returns: [loss, acc, top_k_acc]

    '''
    model = load_model(MODEL_PATH, custom_objects={'top_10_accuracy': top_10_accuracy})
    
    # VGG-16
    try:
      # ignore experimental models with number of output classes other than 230
      if model.get_layer("dense_1").get_config().get("units", "none") == 230:
        model_result = model.evaluate_generator(test_generator, use_multiprocessing=True, verbose=1)
      else:
        model_result = ()
    except:
      model_result = model.evaluate_generator(test_generator, use_multiprocessing=True, verbose=1)
    return list(model_result)


# In[27]:


dict_result = eval_models(candidates)
print(dict_result)


# In[ ]:


model = load_model('model_candidates/model_candidates/inceptionv3_4_new_ohne_dpot_2.97270.hdf5', custom_objects={'top_10_accuracy': top_10_accuracy})


# In[43]:


test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.6)
test_generator = test_datagen.flow_from_directory(
            'test/test',  # this is the target directory
            target_size=(299, 299),
            batch_size=256,
            subset='validation',
            shuffle = True,
            class_mode='categorical')

history = model.evaluate_generator(test_generator, use_multiprocessing=True, verbose=1)


# In[44]:


list(history)


# In[36]:


plt.figure(1)

# summarize history for accuracy

plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


with open('models_result_40409_imgs.txt') as f:
    models_result = json.load(f)


# # Compare which models suits us the most
# ### Return the best model compared from:
# 
# 
# 
# *   Loss
# *   Top 1 accuracy
# *   Top 10 accuracy
# *   Overall
# 

# In[ ]:


def diffr(items):
  '''
    Opposite of the built-in function sum()
  '''
  try:
    e = items[0] - items[1]
    return [e]
  except(IndexError):
    return []

def best_loss(models_dict):
  '''
    Return: list in ascending order of loss
  '''
  return [x[0] for x in sorted(models_dict.items(), key=lambda x: x[1]) if len(x[1]) is not 0]

def best_top_1_acc(models_dict):
  '''
    Return: list in descending order of accuracy
  '''
  return [x[0] for x in sorted(models_dict.items(), key=lambda x: [y[1] for y in x if len(x[1])>0], reverse=True) if len(x[1]) is not 0]
  
def best_top_10_acc(models_dict):
  '''
    Return: list in descending order of top 10 accuracy
  '''
  return [x[0] for x in sorted(models_dict.items(), key=lambda x: [y[2] for y in x if len(x[1])>0 and len(x[1])>2], reverse=True) if len(x[1]) is not 0]

def best_overall(models_dict):
  '''
    Return: list with overall best models calculatet with substracting the accuracy from the loss
  '''
  return [x[0] for x in sorted(models_dict.items(), key=lambda x: diffr([y for y in x[1] if len(x[1])>0])) if len(x[1]) is not 0]


# In[203]:


print("[INFO] Best Loss: {}".format(best_loss(models_result)[0]))
print("[INFO] Best Top 1 accuracy: {}".format(best_top_1_acc(models_result)[0]))
print("[INFO] Best Top 10 accuracy: {}".format(best_top_10_acc(models_result)[0]))
print("\n[INFO] Best Overall: {}".format(best_overall(models_result)[0]))


# ### Best model overall: inceptionv3_4_new_ohne_dpot_2.97270.hdf5
