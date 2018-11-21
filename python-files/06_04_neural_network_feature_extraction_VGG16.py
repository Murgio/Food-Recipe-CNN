# AUTO-GENERATED FROM JUPYTER NOTEBOOKS
# coding: utf-8

# # **Feature Extraction from FC-2 Layer VGG-16**
# 
# ---
# 
# 

# In[1]:


get_ipython().system("echo 'tqdm\\ntables\\nfalconn\\nPyDrive' > requirements.txt")
get_ipython().system('pip install -r requirements.txt')


# In[2]:


import random
import numpy as np
import pandas as pd
import _pickle as pickle
from pprint import pprint

import matplotlib.pyplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')

import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input as preprocess_input_vgg
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inc
from keras.utils.np_utils import to_categorical
from keras.models import Model, load_model
from keras import metrics

#from scipy.spatial import distance
#from sklearn.model_selection import train_test_split
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

import tables
import falconn
#from annoy import AnnoyIndex


# **Upload following files:**
# 
# 
# *   train.txt
# *   test.txt
# *   cookies.txt (session from OneDrive)
# *   client_secrets.json (Upload trained model to Google Drive)
# 
# 

# In[3]:


uploaded = files.upload()

with open("cookies.txt", 'wb') as f:
    f.write(uploaded[list(uploaded.keys())[0]])


# In[ ]:


get_ipython().system('rm cookies\\ \\(1\\).txt')


# **Download following files and unzip them:**
# 
# *   chunk_1.zip
# *   chunk_2.zip
# 
# If existing model wants to be continued learning:
# 
# *   inceptionv3_newest.hdf5

# In[4]:


get_ipython().system('mkdir images')
get_ipython().system('curl -b cookies.txt https://kantiolten-my.sharepoint.com/personal/muriz_serifovic_kantiolten_ch/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fmuriz%5Fserifovic%5Fkantiolten%5Fch%2FDocuments%2FMachine%20Learning%2Fmeta%2Ftest%2Etxt > test.txt')
get_ipython().system('curl -b cookies.txt https://kantiolten-my.sharepoint.com/personal/muriz_serifovic_kantiolten_ch/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fmuriz%5Fserifovic%5Fkantiolten%5Fch%2FDocuments%2FMachine%20Learning%2Fmeta%2Ftrain%2Etxt > train.txt')
get_ipython().system('curl -b cookies.txt https://kantiolten-my.sharepoint.com/personal/muriz_serifovic_kantiolten_ch/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fmuriz%5Fserifovic%5Fkantiolten%5Fch%2FDocuments%2FMachine%20Learning%2Fimages%2Fchunk%5F1%2Ezip > chunk_1.zip')
get_ipython().system('unzip -q chunk_1.zip -d images')
get_ipython().system('rm chunk_1.zip')
get_ipython().system('curl -b cookies.txt https://kantiolten-my.sharepoint.com/personal/muriz_serifovic_kantiolten_ch/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fmuriz%5Fserifovic%5Fkantiolten%5Fch%2FDocuments%2FMachine%20Learning%2Fimages%2Fchunk%5F2%2Ezip > chunk_2.zip')
get_ipython().system('unzip -q chunk_2.zip -d images')
get_ipython().system('rm chunk_2.zip')
get_ipython().system('rm -r images/__MACOSX/')
get_ipython().system('find images -print | wc -l ')
get_ipython().system('find images -name "*.jpg" -size -1k -delete')
get_ipython().system('curl -b cookies.txt https://kantiolten-my.sharepoint.com/personal/muriz_serifovic_kantiolten_ch/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fmuriz%5Fserifovic%5Fkantiolten%5Fch%2FDocuments%2FMachine%20Learning%2Ffeatures%2Fvgg16%5Fbottleneck%5Ffeatures%2Ehdf5 > vgg16_bottleneck_features.hdf5')
get_ipython().system('curl -b cookies.txt https://kantiolten-my.sharepoint.com/personal/muriz_serifovic_kantiolten_ch/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fmuriz%5Fserifovic%5Fkantiolten%5Fch%2FDocuments%2FMachine%20Learning%2Fmodels%2Finceptionv3%5F4%5Fnew%5Fohne%5Fdpot%5F2%2E97270%2Ehdf5 > incep.hdf5')
get_ipython().system('curl -b cookies.txt https://kantiolten-my.sharepoint.com/personal/muriz_serifovic_kantiolten_ch/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fmuriz%5Fserifovic%5Fkantiolten%5Fch%2FDocuments%2FMachine%20Learning%2Fmeta%2Fclasses%5F230%2Etxt > classes_230.txt')
get_ipython().system('curl -b cookies.txt https://kantiolten-my.sharepoint.com/personal/muriz_serifovic_kantiolten_ch/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fmuriz%5Fserifovic%5Fkantiolten%5Fch%2FDocuments%2FMachine%20Learning%2Fmeta%2Fchefkoch%5Fmerged%5Flists%5F05%5Fmay%2Ecsv > chefkoch.csv')


# In[5]:


root_dir = 'images/'
rows = 6
cols = 6
fig, ax = plt.subplots(rows, cols, frameon=False, figsize=(12, 12))
fig.suptitle('Random Image from Each Food Class', fontsize=20)
sorted_food_dirs = sorted(os.listdir(root_dir))
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
        ax[i][j].text(0, 0, food_dir, size=15, rotation=0,
                ha="left", va="top", 
                bbox=dict(boxstyle="round", ec=ec, fc=fc))
plt.setp(ax, xticks=[], yticks=[])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# In[35]:


print(device_lib.list_local_devices())
print(psutil.virtual_memory())


# In[7]:


get_ipython().system('df -h')


# In[8]:


model = keras.applications.VGG16(weights='imagenet', include_top=True)


# In[ ]:


# get_image will return a handle to the image itself, and a numpy array of its pixels to input the network
def get_image_vgg(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input_vgg(x)
    return img, x
  
def get_image_inc(path):
    img = image.load_img(path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input_inc(x)
    return img, x
  
def plot_preds(image, preds, top_n):  
    plt.imshow(image)
    plt.axis('off')
    plt.figure()
    
    order = list(reversed(range(top_n)))
    labels = [categories[x] for i, x in enumerate(np.argsort(-probabilities)[:top_n])]
    bar_preds = [-np.sort(-probabilities)[i] for i, x in enumerate(np.argsort(-probabilities)[:top_n])]
    
    plt.barh(order, bar_preds, alpha=0.8, color='g')
    plt.yticks(order, labels, color='g')
    plt.xlabel('Probability', color='g')
    plt.xlim(0, 1.01)
    plt.tight_layout()
    plt.show()

def show_result_images(final_result):
    rows = 2
    cols = 3
    fig, ax = plt.subplots(rows, cols, frameon=False, figsize=(12, 12))
    fig.suptitle('Result Images from Query', fontsize=20)
    food_dirs = [food_direction[4] for food_direction in final_result]
    for i in range(rows):
      for j in range(cols):
        food_dir = food_dirs[i*cols + j]
        img = plt.imread(food_dir)
        ax[i][j].imshow(img)
        ec = (0, .6, .1)
        fc = (1, 1, 1)
        ax[i][j].text(0, 0, get_corresponding_recipes(final_result).recipe_name[i*cols + j], size=15, rotation=0,
                ha="left", va="top", 
                bbox=dict(boxstyle="round", ec=ec, fc=fc))
    plt.setp(ax, xticks=[], yticks=[])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# In[10]:


feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
img, x = get_image_vgg("images/kuchen/kuchen-23101005746336-39.jpg.jpg")
feat = feat_extractor.predict(x)

matplotlib.pyplot.figure(figsize=(16,4))
matplotlib.pyplot.plot(feat[0])
matplotlib.pyplot.show()


# In[ ]:


images_path = 'images'
max_num_images = 420000

images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
if max_num_images < len(images):
    images = [images[i] for i in sorted(random.sample(range(len(images)), max_num_images))]

print("keeping %d images to analyze" % len(images))


# In[ ]:


all_ids = []
for im in images:
  id_im = im.split('-')[1]
  all_ids.append(id_im)


# In[ ]:


set_all_ids = set(all_ids)
print(len(set_all_ids))


# **We have 90'301 different recipes at the moment for our service.**

# In[ ]:


hdf5_path = 'vgg16_bottleneck_features.hdf5'


# In[ ]:


data_shape = (0, 4096)
img_dtype = tables.Float32Atom()

hdf5_file = tables.open_file(hdf5_path, mode='w')

hdf5_file.create_array(hdf5_file.root, 'img_paths', images)
features = hdf5_file.create_earray(hdf5_file.root, 'img_features', img_dtype, shape=data_shape)

for image_path in tqdm(images):
    img, x = get_image_vgg(image_path);
    feat = feat_extractor.predict(x)[0]
    features.append(feat[None])
    
hdf5_file.close()


# In[ ]:


feat_nparry = np.array(features)
feat_nparry.shape


# In[ ]:


hdf5_file = tables.open_file(hdf5_path, mode='r')
features = hdf5_file.root.img_features
images = hdf5_file.root.img_paths


# In[ ]:


t = AnnoyIndex(4096)

for i, v in enumerate(tqdm(features[:70000])):
  t.add_item(i, v)
  
t.build(32)
t.save('annoy_index_32_trees.ann')


# In[ ]:


t = AnnoyIndex(4096)
t.load('annoy_index_32_trees.ann')


# In[13]:


get_ipython().run_cell_magic('time', '', 'n = 403885\nd = 4096\n\n#number_of_tables = 50\n#params_cp = falconn.LSHConstructionParameters()\n#params_cp.dimension = len(features[0])\n#params_cp.lsh_family = falconn.LSHFamily.Hyperplane\n#params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared\n#params_cp.l = number_of_tables\n\n\np = falconn.get_default_parameters(n, d)\nt = falconn.LSHIndex(p)\nt.setup(features[:])\nq = t.construct_query_object(num_probes=32)')


# In[ ]:


def get_closest_images_bruteforce(query_features, num_results=5):
    # TODO: Too slow -> scipy.spatial.distance.euclidean
    distances = [ distance.euclidean(query_features, feat) for feat in features ]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]
    return idx_closest
  
def get_closest_images_fast(query_features, num_results=15):
    return q.find_k_nearest_neighbors(query_features, num_results)
  
def get_closest_images_fast_annoy(query_features, num_results=6):
    return t.get_nns_by_vector(query_features, num_results)
    
def get_concatenated_images(indexes, thumb_height):
    thumbs = []
    for idx in indexes:
        img = image.load_img(images[idx])
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image


# In[ ]:


def top_10_accuracy(y_true, y_pred):
  return metrics.top_k_categorical_accuracy(y_true, y_pred, k=10)

model_inc = load_model(filepath='incep.hdf5', custom_objects={'top_10_accuracy': top_10_accuracy})

with open('classes_230.txt', 'r') as f:
    categories = f.read().splitlines()


# In[ ]:


chefkoch_rezepte = pd.read_csv('chefkoch.csv', index_col=False)


# In[ ]:


#chefkoch_rezepte.query('recipe_id in @names')
#chefkoch_rezepte.loc[chefkoch_rezepte['recipe_name'].isin([recipe_id_single[2] for recipe_id_single in final_result])]
#idx_pandas = pd.Index(chefkoch_rezepte['recipe_name']).get_indexer([recipe_id_single[2] for recipe_id_single in final_result])
#chefkoch_rezepte.loc[[recipe_id_single[2] for recipe_id_single in final_result]]


# In[ ]:


def get_corresponding_recipes(final_results, chefkoch_rezepte=chefkoch_rezepte):
  chefkoch_rezepte_result = pd.DataFrame()
  for recipe_id_single in final_results:
    chefkoch_rezepte_result = chefkoch_rezepte_result.append(chefkoch_rezepte.query('recipe_id in @recipe_id_single[2]'), ignore_index=True)
  return chefkoch_rezepte_result


# Scheme of final result:
# 
# ```
# 01. ['category', incep_confidence, recipe_id, image_index, image_path]
# 02. ['category', incep_confidence, recipe_id, image_index, image_path]
# 03. ['category', incep_confidence, recipe_id, image_index, image_path]
# 04. ['category', incep_confidence, recipe_id, image_index, image_path]
# 05. ['category', incep_confidence, recipe_id, image_index, image_path]
# ```
# 
# 

# In[16]:


get_ipython().system('curl http://www.homecookingadventure.com/images/recipes/brownies.jpg > pred.jpg')


# In[22]:


query_image, x = get_image_vgg('pred.jpg');
query_features = feat_extractor.predict(x)[0]

# do a query on a random image
idx_closest = get_closest_images_fast(query_features)
#print(idx_closest)

predicted_labels = [str(images[i]).split('/')[1] for i in idx_closest]
predicted_ids = [[str(images[i]).split('-')[1], str(images[i]).split('-')[2].split('.')[0], images[i].decode("utf-8")] for i in idx_closest]
#print(predicted_labels)

results_image = get_concatenated_images(idx_closest, 400)

# display the query image
matplotlib.pyplot.figure(figsize = (5,5))
plt.axis('off')
imshow(query_image)
matplotlib.pyplot.title("Query Image");

# display the resulting images
matplotlib.pyplot.figure(figsize = (16,12))
plt.axis('off')
imshow(results_image)
matplotlib.pyplot.title("result images");

#pprint([images[i] for i in idx_closest])


# In[36]:


img, x = get_image_inc('pred.jpg')
probabilities = model_inc.predict(x)[0]
plot_preds(img, probabilities, 15)
pred_categories = []

for i, x in enumerate(np.argsort(-probabilities)[:15]):
  confidence = -np.sort(-probabilities)[i]
  #print(categories[x], confidence)
  pred_categories.append([categories[x], confidence])
  
predicted_labels_with_weights = []
for iii in predicted_labels:
  for iiii, ii in enumerate(pred_categories):
    no_result = False
    if ii[0] == iii:
      predicted_labels_with_weights.append([iii, ii[1]])
      break
    if iiii == len(pred_categories)-1:
      predicted_labels_with_weights.append([iii, 0])
      
predicted_labels_with_meta = [xi+yi for xi, yi in zip(predicted_labels_with_weights, predicted_ids)]
final_result = sorted(predicted_labels_with_meta, key=lambda predicted_labels_with_meta: predicted_labels_with_meta[1], reverse=True)

show_result_images(final_result[:6])


# In[33]:


from werkzeug.wrappers import Request, Response
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 8000, app)


# In[34]:


get_ipython().system('hostname')


# #Validate the feature extractor

# In[ ]:


images_path = 'images'
max_num_images = 500

images_test = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
if max_num_images < len(images_test):
    images_test = [images_test[i] for i in sorted(random.sample(range(len(images_test)), max_num_images))]

print("keeping %d images to analyze" % len(images_test))


# In[ ]:


images_test[:10]


# scipy.spatial.distance.euclidean: **200s** with **5 queries**
# 
# falconn.find_k_nearest_neighbors: **0.692s** with **5 queries**

# 1.   10 -> 4.55s
# 2.   100 -> 9.41s
# 3.   500 -> 17.4s
# 
# Accuracy: Size -> 200'000
# 
# **0.356075**
# 
# *   Top1: 23'318
# *   Top5: 47'897
# 
# 
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'top_1_accuracy, top_5_accuracy, totally_wrong = 0, 0, 0\nfor image_test in tqdm(images_test):\n  query_image, x = get_image_vgg(image_test);\n  query_features_test = feat_extractor.predict(x)[0]\n  correct_label = image_test.split(\'/\')[1]\n  #print(correct_label)\n  #matplotlib.pyplot.figure(figsize = (5,5))\n  #imshow(query_image)\n  #matplotlib.pyplot.title("Query Image")\n  idx_closest_test = get_closest_images_fast(query_features_test)\n  #print(idx_closest_test)\n  predicted_labels = [str(images[i]).split(\'/\')[1] for i in idx_closest_test]\n  # Skip first label, falconn gets this right every time\n  #print(predicted_labels[1:])\n  #results_images_test = get_concatenated_images(idx_closest_test, 400)\n  #matplotlib.pyplot.figure(figsize = (16,12))\n  #imshow(results_images_test)\n  #matplotlib.pyplot.title("result images")\n  #plt.axis(\'off\')\n  if correct_label == predicted_labels[1]:\n    top_1_accuracy += 1\n  elif correct_label in predicted_labels[1:]:\n    top_5_accuracy += 1\n  else: totally_wrong += 1\n    \nprint(top_1_accuracy)\nprint(top_5_accuracy)\nprint(totally_wrong)')


# In[ ]:


# Authenticate and create the PyDrive client.
# This only needs to be done once per notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

textfile = drive.CreateFile()
textfile.SetContentFile('vgg16_bottleneck_features.hdf5')
textfile.Upload()

