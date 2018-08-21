from __future__ import division, print_function
# coding=utf-8
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from datetime import datetime

import matplotlib.pyplot
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.applications import VGG16
from keras.applications.imagenet_utils import preprocess_input as preprocess_input_vgg
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inc
from keras.utils.np_utils import to_categorical
from keras.models import Model, load_model
from keras import metrics

import tables
import falconn

from flask import Flask, redirect, url_for, request, render_template, stream_with_context, Response, send_from_directory
from werkzeug.utils import secure_filename

from chefkochParser import food_list_html
import logger
#from gevent.wsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
#app.config['MEDIA_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '/Extracting-food-preferences-master/notebooks/input')

MODEL_PATH = 'models/inceptionv3_4_new_ohne_dpot_2.97270.hdf5'
MODEL_VGG_PATH = 'models/weights.bestVGG16try9.hdf5'
hdf5_PATH = 'models/vgg16_bottleneck_features.hdf5'
CHEFKOCH_PATH = 'https://www.chefkoch.de/rezepte/'
model_inc = None
feat_extractor = None

hdf5_file = tables.open_file(hdf5_PATH, mode='r')
features = hdf5_file.root.img_features
images = hdf5_file.root.img_paths

n = 403885
d = 4096

CONSIDER_N_IMAGES = 15

p = falconn.get_default_parameters(n, d)
t = falconn.LSHIndex(p)
print('STARTING SETUP')
t.setup(features[:])
print('DONE SETUP')
q = t.construct_query_object(num_probes=32)

with open('meta/classes_230.txt', 'r') as textfile:
	categories = textfile.read().splitlines()

def top_10_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=10)

# get_image will return a handle to the image itself, and a numpy array of its pixels to input the network
def get_image_vgg(path):
    img = image.load_img(path, target_size=feat_extractor.input_shape[1:3])
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

def get_closest_images_fast(query_features, num_results=CONSIDER_N_IMAGES):
    return q.find_k_nearest_neighbors(query_features, num_results)
    
def get_concatenated_images(indexes, thumb_height):
    thumbs = []
    for idx in indexes:
        img = image.load_img(images[idx])
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image

def load_models(MODEL_PATH):
    global model_inc, feat_extractor
    model_inc = load_model(MODEL_PATH, custom_objects={'top_10_accuracy': top_10_accuracy})
    model_inc._make_predict_function()
    model_vgg = VGG16(weights='imagenet', include_top=True)
    #model_vgg = load_model(MODEL_VGG_PATH)
    feat_extractor = Model(inputs=model_vgg.input, outputs=model_vgg.get_layer("fc2").output)
    img, x = get_image_vgg("meta/kuchen.jpg")
    feat = feat_extractor.predict(x)
    print('Models loaded. Start serving...')

def model_predict(img_path):

    query_image, x = get_image_vgg(img_path)
    query_features = feat_extractor.predict(x)[0]
    # do a query on a random image
    idx_closest = get_closest_images_fast(query_features)
    #print(idx_closest)
    predicted_labels = [str(images[i]).split('/')[1] for i in idx_closest]
    predicted_ids = [[str(images[i]).split('-')[1], str(images[i]).split('-')[2].split('.')[0], images[i].decode("utf-8")] for i in idx_closest]
    #print(predicted_ids)

    # result_links = []
    # for id_food in predicted_ids:
    #     result_links.append(CHEFKOCH_PATH+id_food[0])
    # result = food_list_html(result_links)
    # result2=[]
    # for image_id in predicted_ids:
    #     result2.append(image_id[0])
    #     result2.append(image_id[1])
    # result = ' '.join(result2)


    img, x = get_image_inc(img_path)
    probabilities = model_inc.predict(x)[0]
    #plot_preds(img, probabilities, 15)
    pred_categories = []

    for i, x in enumerate(np.argsort(-probabilities)[:CONSIDER_N_IMAGES]):
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

    #show_result_images(final_result[:6])
    return final_result

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

#@app.route('/images/<path:filename>')
#def custom_static(filename):
#    print(filename)
#    return send_from_directory(app.config['MEDIA_FOLDER'], filename)

@app.route('/predict', methods=['GET', 'POST'])
def streamed_response():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # [['category', incep_confidence, recipe_id, image_index, image_path], [], ...]
        start = datetime.now()
        result_list = model_predict(file_path)
        end = datetime.now()
        print('FOUND THE FOOD')
        food_time = end-start
        # max 1 second
        print('TOOK ME: ', food_time.microseconds*(1/1000000))
        logger.log(result_list, food_time)
        #result = ' '.join([str(i) for i in result])
        ids = [food_id[2] for food_id in result_list]
        food = food_list_html(result_list=result_list[:5], online=False)
        return ''.join(map(str, food))
    #def upload():
    #        for i, id in enumerate(ids):
    #            yield food_list_html(id_food=id, i=i)
    #return Response(stream_with_context(upload()))
    return None

if __name__ == '__main__':
    load_models(MODEL_PATH)
    app.run(host='0.0.0.0', port=5000, debug=False)

    # Serve the app with gevent
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
