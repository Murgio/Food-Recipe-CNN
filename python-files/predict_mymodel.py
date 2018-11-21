# AUTO-GENERATED FROM JUPYTER NOTEBOOKS
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os

import random
import numpy as np
import keras

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from PIL import Image

from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model


# In[4]:


from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# In[ ]:


model_new = load_model('test/inceptionv3.hdf5')
model_new.summary()


# In[5]:


model_2 = load_model('test/inceptionv3_3.hdf5')


# In[6]:


with open('test/classes_230.txt', 'r') as f:
    categories = f.read().splitlines()


# In[7]:


target_size = (299, 299) #fixed size for InceptionV3 architecture

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


# In[8]:


# helper function to load image and return it and input vector
def get_image(path):
    img = image.load_img(path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


# In[71]:


get_ipython().system('curl https://static.chefkoch-cdn.de/ck.de/rezepte/90/90634/280470-960x720-pasta-mit-knoblauch-tomaten-shrimps.jpg > ../Desktop/flasktest/simple-keras-rest-api/pred.jpg')


# In[72]:


img, x = get_image('../Desktop/flasktest/simple-keras-rest-api/pred.jpg')
probabilities = model_2.predict(x)[0]
#print(probabilities)
plot_preds(img, probabilities, 5)
[print(categories[x], (-np.sort(-probabilities)[i]*100)) for i, x in enumerate(np.argsort(-probabilities)[:5])];


# In[10]:


def center_crop(x, center_crop_size, **kwargs):
    centerw, centerh = x.shape[0]//2, x.shape[1]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return x[centerw-halfw:centerw+halfw+1,centerh-halfh:centerh+halfh+1, :]


# In[11]:


def predict_10_crop(img, ix, top_n=1, plot=False, preprocess=True, debug=False):
    flipped_X = np.fliplr(img)
    crops = [
        img[:299,:299, :], # Upper Left
        img[:299, img.shape[1]-299:, :], # Upper Right
        img[img.shape[0]-299:, :299, :], # Lower Left
        img[img.shape[0]-299:, img.shape[1]-299:, :], # Lower Right
        center_crop(img, (299, 299)),
        
        flipped_X[:299,:299, :],
        flipped_X[:299, flipped_X.shape[1]-299:, :],
        flipped_X[flipped_X.shape[0]-299:, :299, :],
        flipped_X[flipped_X.shape[0]-299:, flipped_X.shape[1]-299:, :],
        center_crop(flipped_X, (299, 299))
    ]
    if preprocess:
        crops = [preprocess_input(x.astype('float32')) for x in crops]

    if plot:
        fig, ax = plt.subplots(2, 5, figsize=(10, 4))
        ax[0][0].imshow(crops[0])
        ax[0][1].imshow(crops[1])
        ax[0][2].imshow(crops[2])
        ax[0][3].imshow(crops[3])
        ax[0][4].imshow(crops[4])
        ax[1][0].imshow(crops[5])
        ax[1][1].imshow(crops[6])
        ax[1][2].imshow(crops[7])
        ax[1][3].imshow(crops[8])
        ax[1][4].imshow(crops[9])
    
    y_pred = model_2.predict(np.array(crops))
    preds = np.argmax(y_pred, axis=1)
    top_n_preds= np.argpartition(y_pred, -top_n)[:,-top_n:]
    if debug:
        print('Top-1 Predicted:', preds)
        print('Top-5 Predicted:', top_n_preds)
        print('True Label:', y_test[ix])
    return preds, top_n_preds


# In[36]:


import matplotlib.image as img
import collections
from collections import defaultdict
pic_path = '../Desktop/flasktest/simple-keras-rest-api/pred.jpg'
pic = img.imread(pic_path)
preds = predict_10_crop(np.array(pic), 0)[1]
best_pred = [collections.Counter(pred).most_common(1)[0][0] for pred in preds]
[print(categories[i]) for i in best_pred]
plt.imshow(pic)


# In[13]:


import pandas as pd
recipes_csv = pd.read_csv('test/recipes_classify.csv')


# In[14]:


recipes_csv.head()


# In[15]:


labels_all = [categories[x] for i, x in enumerate(np.argsort(-probabilities)[:5])]
bar_preds = [-np.sort(-probabilities)[i] for i, x in enumerate(np.argsort(-probabilities)[:5])]


# In[24]:


import random
results = []
for label_x in labels_all:
    result_0 = recipes_csv.ix[recipes_csv['category']==label_x.replace('_', ' ')]
    result_0=result_0.take(np.random.permutation(len(result_0))[:1])
    result_0.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'recipe_id', 'tags', 'category'], 1, inplace=True)
    results.append(result_0.to_json(force_ascii=False))


# Schokokuchen

# In[ ]:


pred_1 = ['Schokokuchen', '4 Ei(er)\n250 g Zucker\n200 ml Öl\n200 ml Orangensaft\n300 g Mehl\n1 Pck. Backpulver\n3 EL Kakaopulver\n200 g Kuvertüre (Zartbitter)\n', 'Den Ofen auf 200° vorheizen. Eier mit Zucker dick-cremig schlagen. Öl und Saft zugeben. Mehl mit Backpulver und Kakao rasch unterrühren. Teig in eine gefettete Kastenform füllen, im Ofen bei 180° (Umluft) 40-45 Minuten backen und abkühlen lassen. Den Kuchen mit der geschmolzenen Zartbitterkuvertüre bestreichen.']


# In[ ]:


pred_2 = ['Pasta mit Spinat und Shrimps', '500 g Nudeln, am besten Spiralnudeln\n1 Pck. Spinat, TK, aufgetaut\n1 Becher Schmand\n1 Zitrone(n)\n2 Knoblauchzehe(n)\n200 g Shrimps\nSalz und Pfeffer\nMuskat\nButter', 'Die Nudeln in Salzwasser al dente kochen. In der Zwischenzeit die Shrimps in einem Topf mit Butter und Knoblauch anbraten. Den Spinat und den Schmand dazu geben. Die Zitrone auspressen und den Saft auch dazu geben. Mit Salz, Pfeffer und Muskat abschmecken. Alles mit den Nudeln vermischen und sofort servieren.']


# In[ ]:


pred_3 = ['Mojito', '6 cl Rum (Havana Club, 3 Jahre oder Anejo Reserva))\n1/2 TL Rohrzucker, weiß, sehr fein\n1/2 Limette(n), bei wenig Saftgehalt auch eine ganze\nMineralwasser\nMinze, Zweige\nEis in Stücken', 'In ein Cocktailglas Zucker und Sodawasser geben. Limette viertel, Saft über dem Glas ausdrücken, Limettenstücke dazugeben. Mit einem Holzstößel die Limette im Glas nochmals ausdrücken. Gut verrühren. Einige Minzezweige dazugeben, mit dem Holzlöffel die Stiele zerquetschen, dabei nicht die Blätter beschädigen. Das Glas mit grob zerschlagenen Eiswürfeln füllen. Havanna und etwas Sodawasser dazugeben und gut umrühren. Mit Trinkhalm servieren.']


# In[70]:


recipes_csv.loc[[23809]].to_json(force_ascii=False)


# In[25]:


from IPython.core.display import display, HTML

display(HTML(results[0]))


# In[26]:


import uuid
from IPython.display import display_javascript, display_html, display
import json

class RenderJSON(object):
    def __init__(self, json_data):
        if isinstance(json_data, dict):
            self.json_str = json.dumps(json_data)
        else:
            self.json_str = json_data
        self.uuid = str(uuid.uuid4())

    def _ipython_display_(self):
        display_html('<div id="{}" style="height: 600px; width:100%;"></div>'.format(self.uuid), raw=True)
        display_javascript("""
        require(["https://rawgit.com/caldwell/renderjson/master/renderjson.js"], function() {
        document.getElementById('%s').appendChild(renderjson(%s))
        });
        """ % (self.uuid, self.json_str), raw=True)


# In[62]:


results[0]


# In[28]:


RenderJSON(results[0])


# In[30]:


print (json.dumps(results[0], indent=2))


# In[31]:


def pp_json(json_thing, sort=True, indents=4):
    if type(json_thing) is str:
        print(json.dumps(json.loads(json_thing), sort_keys=sort, indent=indents))
    else:
        print(json.dumps(json_thing, sort_keys=sort, indent=indents))
    return None


# In[33]:


pp_json(results[0])


# In[ ]:


# THIS IS A TRY WITH EUCLIDIAN DISTANCE


# In[11]:


import _pickle as pickle
images_all, pca_features_all = pickle.load(open('test/features_300000_recipes.p', 'rb'), encoding='latin1')


# In[12]:


add_string = '../Desktop/Extracting-food-preferences-master/notebooks/input/images'
for index, s in enumerate(images_all):
    final_string = add_string+s[6:]
    images_all[index] = final_string


# In[36]:


def get_closest_images(num_results=6):
    distances = [ distance.euclidean(pca_features[10], feat) for feat in pca_features_all ]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]
    return idx_closest

def get_concatenated_images(indexes, thumb_height):
    thumbs = []
    for idx in indexes:
        img = image.load_img(images_all[idx])
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image


# In[34]:


import random
from scipy.spatial import distance
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot
from matplotlib.pyplot import imshow


# In[35]:


# do a query on a random image
#query_image_idx = int(len(images) * random.random())
#print(images[query_image_idx])
idx_closest = get_closest_images()
#[print(xx) for xx in idx_closest]
#query_image = get_concatenated_images([query_image_idx], 300)
results_image = get_concatenated_images(idx_closest, 200)

# display the query image
#matplotlib.pyplot.figure(figsize = (5,5))
#imshow(query_image)
#matplotlib.pyplot.title("query image (%d)" % query_image_idx)

# display the resulting images
matplotlib.pyplot.figure(figsize = (16,12))
imshow(results_image)
matplotlib.pyplot.title("result images")


# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import random
import _pickle as pickle
import numpy as np
import matplotlib.pyplot
from matplotlib.pyplot import imshow
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA
from scipy.spatial import distance
from tqdm import tqdm


# In[14]:


model_vgg_original = keras.applications.VGG16(weights='imagenet', include_top=True)
feat_extractor = Model(inputs=model_vgg_original.input, outputs=model_vgg_original.get_layer("fc2").output)


# In[15]:


# get_image will return a handle to the image itself, and a numpy array of its pixels to input the network
def get_image_vgg(path):
    img = image.load_img(path, target_size=model_vgg_original.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


# In[16]:


#features = []
#img, x = get_image_vgg('../Desktop/flasktest/simple-keras-rest-api/pred.jpg');
#feat = feat_extractor.predict(x)[0]
#features.append(feat)

features = []
img, x = get_image_vgg('../Desktop/flasktest/simple-keras-rest-api/pred.jpg');
feat = feat_extractor.predict(x)[0]
for _ in range(1000):
    features.append(feat)

features = np.array(features)
pca = PCA(n_components=300)
pca.fit(features)
pca_features = pca.transform(features)


# In[27]:


pca_features[0]

