# AUTO-GENERATED FROM JUPYTER NOTEBOOKS
# coding: utf-8

# # **Inception V3 Food Classification 230 categories**
# 
# ---
# 
# 

# In[ ]:


import random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')

import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.np_utils import to_categorical
from keras.applications.inception_v3 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model, load_model

import os
from os import listdir
from os.path import isfile, join
import shutil
import stat
import collections
from collections import defaultdict

import h5py
from sklearn.model_selection import train_test_split


# **Upload following files:**
# 
# 
# *   train.txt
# *   test.txt
# *   cookies.txt (session from OneDrive)
# *   client_secrets.json (Upload trained model to Google Drive)
# 
# 

# In[17]:


from google.colab import files
uploaded = files.upload()

with open("cookies.txt", 'wb') as f:
    f.write(uploaded[list(uploaded.keys())[0]])


# **Download following files and unzip them:**
# 
# *   chunk_1.zip
# *   chunk_2.zip
# 
# If existing model wants to be continued learning:
# 
# *   inceptionv3_newest.hdf5

# In[ ]:


get_ipython().system('mkdir images')


# In[18]:


get_ipython().system('curl -b cookies.txt https://kantiolten-my.sharepoint.com/personal/muriz_serifovic_kantiolten_ch/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fmuriz%5Fserifovic%5Fkantiolten%5Fch%2FDocuments%2FMachine%20Learning%2Fimages%2Fchunk%5F1%2Ezip > chunk_1.zip')


# In[19]:


get_ipython().system('unzip -q chunk_1.zip -d images')


# In[ ]:


get_ipython().system('rm chunk_1.zip')


# In[21]:


get_ipython().system('ls')


# In[22]:


get_ipython().system('curl -b cookies.txt https://kantiolten-my.sharepoint.com/personal/muriz_serifovic_kantiolten_ch/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fmuriz%5Fserifovic%5Fkantiolten%5Fch%2FDocuments%2FMachine%20Learning%2Fimages%2Fchunk%5F2%2Ezip > chunk_2.zip')


# In[23]:


get_ipython().system('unzip -q chunk_2.zip -d images')


# In[ ]:


get_ipython().system('rm chunk_2.zip')


# In[ ]:


get_ipython().system('rm -r images/__MACOSX/')


# In[26]:


get_ipython().system('find images -print | wc -l ')


# In[ ]:


get_ipython().system('find images -name "*.jpg" -size -1k -delete')


# In[34]:


get_ipython().system('curl -b cookies.txt https://kantiolten-my.sharepoint.com/personal/muriz_serifovic_kantiolten_ch/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fmuriz%5Fserifovic%5Fkantiolten%5Fch%2FDocuments%2FMachine%20Learning%2Fmodels%2Finceptionv3%5F4%5Fnew%5F3%2E04903%2Ehdf5 > incep.hdf5')


# In[28]:


root_dir = 'images/'
rows = 17
cols = 6
fig, ax = plt.subplots(rows, cols, frameon=False, figsize=(15, 25))
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
        ax[i][j].text(0, -20, food_dir, size=10, rotation=0,
                ha="left", va="top", 
                bbox=dict(boxstyle="round", ec=ec, fc=fc))
plt.setp(ax, xticks=[], yticks=[])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# In[ ]:


# Only split files if haven't already
if not os.path.isdir('test') and not os.path.isdir('train'):

    def copytree(src, dst, symlinks = False, ignore = None):
        if not os.path.exists(dst):
            os.makedirs(dst)
            shutil.copystat(src, dst)
        lst = os.listdir(src)
        if ignore:
            excl = ignore(src, lst)
            lst = [x for x in lst if x not in excl]
        for item in lst:
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if symlinks and os.path.islink(s):
                if os.path.lexists(d):
                    os.remove(d)
                os.symlink(os.readlink(s), d)
                try:
                    st = os.lstat(s)
                    mode = stat.S_IMODE(st.st_mode)
                    os.lchmod(d, mode)
                except:
                    pass # lchmod not available
            elif os.path.isdir(s):
                copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)

    def generate_dir_file_map(path):
        dir_files = defaultdict(list)
        with open(path, 'r') as txt:
            files = [l.strip() for l in txt.readlines()]
            for f in files:
                dir_name, id = f.split('/')
                dir_files[dir_name].append(id + '.jpg')
        return dir_files

    train_dir_files = generate_dir_file_map('train.txt')
    test_dir_files = generate_dir_file_map('test.txt')


    def ignore_train(d, filenames):
        print(d)
        subdir = d.split('/')[-1]
        to_ignore = train_dir_files[subdir]
        return to_ignore

    def ignore_test(d, filenames):
        print(d)
        subdir = d.split('/')[-1]
        to_ignore = test_dir_files[subdir]
        return to_ignore

    copytree('images', 'test', ignore=ignore_train)
    copytree('images', 'train', ignore=ignore_test)
    
else:
    print('Train/Test files already copied into separate folders.')


# In[30]:


from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# In[31]:


get_ipython().system('df -h')


# In[32]:


import psutil
psutil.virtual_memory()


# In[ ]:


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD, Adadelta
from keras.regularizers import l2
import keras.backend as K
import math

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.layers import Input


# In[29]:


inc = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))


# In[ ]:


x = inc.output
x = AveragePooling2D(pool_size=(8, 8))(x)
x = Dropout(.2)(x) # Dropout slows training down
x = Flatten()(x)
predictions = Dense(230, kernel_initializer='glorot_uniform', kernel_regularizer=l2(.0005), activation='softmax')(x)

model = Model(inputs=inc.input, outputs=predictions)

#model = load_model(filepath='inceptionv3_3.hdf5')

#opt = SGD(lr=0.01, momentum=.9)
model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='inceptionv3_4.hdf5', verbose=1, save_best_only=True)

batch_size = 64

train_datagen = ImageDataGenerator(rescale=1./255,
                   rotation_range=10,
                   width_shift_range=0.05,
                   height_shift_range=0.05,
                   zoom_range=0.2,  # 0.75,
                   channel_shift_range=10,
                   shear_range=0.05,
                   horizontal_flip=True,
                   fill_mode="constant")

test_datagen = ImageDataGenerator(rescale=1./255,
                   rotation_range=10,
                   width_shift_range=0.05,
                   height_shift_range=0.05,
                   zoom_range=0.2,  # 0.75,
                   channel_shift_range=10,
                   shear_range=0.05,
                   horizontal_flip=True,
                   fill_mode="constant")

train_generator = train_datagen.flow_from_directory(
        'train',  # this is the target directory
        target_size=(299, 299),
        batch_size=batch_size,
        shuffle = True,
        class_mode='categorical')

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'test',
        target_size=(299, 299),
        batch_size=batch_size,
        shuffle = True,
       class_mode='categorical')


history = model.fit_generator(train_generator,
                    validation_data=validation_generator,
                    epochs=3,
                    verbose=1,
                    callbacks= [checkpointer])


# In[ ]:


opt = SGD(lr=0.001, momentum=.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='inceptionv3_3_1.hdf5', verbose=1, save_best_only=True)
model.fit_generator(train_generator,
                    validation_data=validation_generator,
                    epochs=1,
                    verbose=1,
                    callbacks= [checkpointer])


# In[ ]:


model = load_model(filepath='incep.hdf5')


# In[ ]:


for layer in model_new.layers[:172]:
  layer.trainable = False
for layer in model_new.layers[172:]:
  layer.trainable = True


# In[86]:


for layer in model_new.layers[-4:]:
  print(layer.output_shape)


# In[61]:


# pop the last 4 layers
model.layers.pop()
model.layers.pop()
model.layers.pop()
model.layers.pop()


# In[ ]:


x = model.layers[-1].output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer
predictions = Dense(230, activation='softmax')(x)

model_new = Model(inputs=model.input, outputs=predictions)


# In[ ]:


model_new.compile(
    optimizer='adadelta',
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy'])
model_new.summary()


# In[ ]:


for layer in model_new.layers[:172]:
  layer.trainable = False
for layer in model_new.layers[172:]:
  layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model_new.compile(
    optimizer=SGD(lr=0.001, momentum=0.3, nesterov=True),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy'])


# In[88]:


#model = load_model(filepath='incep.hdf5')

#for layer in model.layers[:172]:
#  layer.trainable = False
#for layer in model.layers[172:]:
#  layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
#model.compile(
#    optimizer=SGD(lr=0.0001, momentum=0.9),
#    loss='categorical_crossentropy',
#    metrics=['accuracy', 'top_k_categorical_accuracy'])


#checkpointer = ModelCheckpoint(filepath='inceptionv3_4_new_{epoch:02d}-{val_loss:.3f}.hdf5', verbose=1, save_best_only=False)
checkpointer = ModelCheckpoint(filepath='inceptionv3_4_new_17_april.hdf5', verbose=1, save_best_only=True)

batch_size = 64

train_datagen = ImageDataGenerator(rescale=1./255,
                   rotation_range=10,
                   width_shift_range=0.05,
                   height_shift_range=0.05,
                   zoom_range=0.2,  # 0.75,
                   channel_shift_range=10,
                   shear_range=0.05,
                   horizontal_flip=True,
                   fill_mode="constant")

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'train',  # this is the target directory
        target_size=(299, 299),
        batch_size=batch_size,
        shuffle = True,
        class_mode='categorical')

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'test',
        target_size=(299, 299),
        batch_size=batch_size,
        shuffle = True,
       class_mode='categorical')


history = model_new.fit_generator(train_generator,
                    validation_data=validation_generator,
                    epochs=2,
                    verbose=1,
                    callbacks= [checkpointer])


# In[90]:


get_ipython().system('curl https://s3.amazonaws.com/stratospark/food-101/model4b.10-0.68.hdf5 > model4b.hdf5')


# In[ ]:


model4b = load_model(filepath='model4b.hdf5')


# In[94]:


history = model_new.fit_generator(train_generator,
                    validation_data=validation_generator,
                    epochs=1,
                    verbose=1,
                    callbacks= [checkpointer])


# In[89]:


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


from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

#1st authentification
gauth = GoogleAuth()
gauth.LocalWebserverAuth() # Creates local webserver and auto handles 
#authentication.
drive = GoogleDrive(gauth)

file1 = drive.CreateFile()
file1.SetContentFile('test.txt')
file1.Upload()


# In[ ]:


files.download('model_inception.zip')


# In[ ]:


get_ipython().system('zip -r model_inception.zip model')


# In[ ]:


from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
# Try to load saved client credentials
gauth.LoadCredentialsFile()
if gauth.credentials is None:
    # Authenticate if they're not there
    gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
    # Refresh them if expired
    gauth.Refresh()
else:
    # Initialize the saved creds
    gauth.Authorize()
# Save the current credentials to a file
gauth.SaveCredentialsFile("mycreds.txt")

drive = GoogleDrive(gauth)

textfile = drive.CreateFile()
textfile.SetContentFile('eng.txt')
textfile.Upload()


# In[ ]:


# Install the PyDrive wrapper & import libraries.
# This only needs to be done once per notebook.
get_ipython().system('pip install -U -q PyDrive')
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
# This only needs to be done once per notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

textfile = drive.CreateFile()
textfile.SetContentFile('inceptionv3_4_new_17_april.hdf5')
textfile.Upload()

