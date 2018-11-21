# AUTO-GENERATED FROM JUPYTER NOTEBOOKS
# coding: utf-8

# In[1]:


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[2]:


import csv


# In[3]:


def get_preperation():
    merged_list = []
    skip_first = False # col name
    chef_file = '/input/recipe_details_27-12-2017.csv'
    with open(chef_file, 'r') as f:
        chefkoch = csv.reader(f)
        for row in chefkoch:
            if skip_first:
                skip_first = False
                continue
            try:
                merged_list.append(row[2])
            except: 
                continue
    text = ' '.join(merged_list[:6000])
    return(text)


# In[4]:


text = get_preperation()


# ### Lnge des gesamten Zubereitungstextes von allen Rezepten

# In[5]:


print("{:,} Zeichen fuer 20'000 Rezepte".format(len(text)))


# In[6]:


import os
os.environ["KERAS_BACKEND"] = "theano"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
import keras; import keras.backend
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
if keras.backend.backend() != 'theano':
    raise BaseException("This script uses other backend")
else:
    keras.backend.set_image_dim_ordering('th')
    print("Backend ok")


# In[9]:


import random
import numpy as np
from glob import glob

chars = list(set(text))

# set a fixed vector size
max_len = 20


# In[10]:


model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(max_len, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


# In[11]:


model.summary()


# In[12]:


step = 3
inputs = []
outputs = []
for i in range(0, len(text) - max_len, step):
    inputs.append(text[i:i+max_len])
    outputs.append(text[i+max_len])


# In[11]:


get_ipython().system(' pip install psutil')


# In[13]:


import psutil
psutil.virtual_memory()


# In[14]:


char_labels = {ch:i for i, ch in enumerate(chars)}
labels_char = {i:ch for i, ch in enumerate(chars)}

# using bool to reduce memory usage
X = np.zeros((len(inputs), max_len, len(chars)), dtype=np.bool)
y = np.zeros((len(inputs), len(chars)), dtype=np.bool)

# one-hot vector
for i, example in enumerate(inputs):
    for t, char in enumerate(example):
        X[i, t, char_labels[char]] = 1
    y[i, char_labels[outputs[i]]] = 1


# In[15]:


def generate(temperature=0.35, seed=None, num_chars=150):
    predicate=lambda x: len(x) < num_chars
    
    if seed is not None and len(seed) < max_len:
        raise Exception('{} chars long'.format(max_len))

    else:
        start_idx = random.randint(0, len(text) - max_len - 1)
        seed = text[start_idx:start_idx + max_len]

    sentence = seed
    generated = sentence

    while predicate(generated):
        # generate input tensor
        x = np.zeros((1, max_len, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_labels[char]] = 1.

        probs = model.predict(x, verbose=0)[0]
        next_idx = sample(probs, temperature)
        next_char = labels_char[next_idx]

        generated += next_char
        sentence = sentence[1:] + next_char
    return generated

def sample(probs, temperature):
    a = np.log(probs)/temperature
    dist = np.exp(a)/np.sum(np.exp(a))
    choices = range(len(probs))
    return np.random.choice(choices, p=dist)


# In[16]:


epochs = 100
nb_epoch_num = 0
for i in range(epochs):
    print('epoch %d'%i)

    model.fit(X, y, batch_size=128, epochs=1)
    nb_epoch_num += 1
    model.save_weights('/output/RNN_checkpoint_{}_epoch.hdf5'.format(nb_epoch_num))

    # preview
    for temp in [0.2, 0.5, 1., 1.2]:
        print('temperature: %0.2f'%temp)
        print('%s'%generate(temperature=temp))


# In[21]:


print('%s' % generate(temperature=0.4,
                      seed='Ich salze meine Nudeln mit Salz',
                      num_chars=2000))


# In[22]:


print('%s' % generate(temperature=1.0,
                      seed='Pfanne reinigen, kein Wasser dazu giessen',
                      num_chars=2000))


# In[23]:


print('%s' % generate(temperature=0.2,
seed='Alles dazugeben und in einen Kochtopf auf mittlerer Hitze kurz aufkochen lassen.',
                      num_chars=5000))


# In[24]:


print('%s' % generate(temperature=0.8,
seed='Alles dazugeben und in einen Kochtopf auf mittlerer Hitze kurz aufkochen lassen.',
                      num_chars=5000))

