# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA

from keras.preprocessing import image
from keras.applications import VGG16
from keras.applications.imagenet_utils import preprocess_input as preprocess_input_vgg
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inc
from keras.models import Model, load_model
from keras import metrics

import tables
import nmslib
import pickle

class PredictFood:
	"""Calculate the prediction result list from the neural networks.

    Attributes:
        k: Size of the result and number of neighbours from ANN
    """
	def __init__(self, k):
		self.k = k
		# Neural networks
		self.model_inc, self.feat_extractor = self._load_models('models/inceptionv3_4_new_ohne_dpot_2.97270.hdf5')

		# Pytable containing image paths
		self.hdf5_file = tables.open_file('models/vgg16_bottleneck_features_IPCA.hdf5', mode='r')
		self.images = self.hdf5_file.root.img_paths

		# ICPA object from sklearn
		self.ipca = self._load_ipca()

		# nmslib Index for ANN
		self.index_ann = None
		self._set_ann_index('models/image_features_pca_nmslib_index.bin')

		# 230 categories for IncetionV3
		self.categories = self._set_categories('meta/classes_230.txt')
		# CSV file containing scraped data
		self.chefkoch_rezepte = self._load_chefkoch_rezepte('meta/chefkoch.csv')

	def _set_categories(self, path):
		"""Return the 230 categories."""
		with open(path, 'r') as textfile:
			return textfile.read().splitlines()

	def _set_ann_index(self, bin_path):
		"""Initialize the nmslib index."""

		# Intitialize the library, specify the space, the type of the vector and add data points 
		self.index_ann = nmslib.init(method='hnsw', space='l2', data_type=nmslib.DataType.DENSE_VECTOR)
		# Re-load the index and re-run queries
		self.index_ann.loadIndex(bin_path)
		# Setting query-time parameters and querying
		self.index_ann.setQueryTimeParams({'efSearch': 100})

	def _load_ipca(self):
		"""Return the Incremental PCA object from disk."""
		pickle_in = open("models/sklearn_ipca_object.p","rb")
		return pickle.load(pickle_in)

	def _load_models(self, model_PATH):
		"""Return the trained InceptionV3 model and a new VGG-16 model."""
		model_inc = load_model(model_PATH, custom_objects={'top_10_accuracy': self.top_10_accuracy})
		model_inc._make_predict_function()
		model_vgg = VGG16(weights='imagenet', include_top=True)
		feat_extractor = Model(inputs=model_vgg.input, outputs=model_vgg.get_layer("fc2").output)
		img, x = self.get_image_vgg("meta/kuchen.jpg")
		feat = feat_extractor.predict(x)
		return model_inc, feat_extractor

	def _load_chefkoch_rezepte(self, path):
		"""Return the scraped data from disk."""
		return pd.read_csv(path, index_col=False)

	def top_10_accuracy(self, y_true, y_pred):
		"""Custom metric for top K predicition."""
		return metrics.top_k_categorical_accuracy(y_true, y_pred, k=10)

	def get_image_vgg(self, path):
		"""Return a handle to the image itself, and a numpy array of its pixels to input the network."""
		img = image.load_img(path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input_vgg(x)
		return img, x
	  
	def get_image_inc(self, path):
		img = image.load_img(path, target_size=(299, 299))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input_inc(x)
		return img, x
	  
	def get_closest_images_nmslib(self, query_features, num_results):
		"""Return the nearest neighbors to the query image."""
		return self.index_ann.knnQuery(query_features, k = num_results)

	def weighting_neural_net_inputs(self, query_features, probabilities):
		"""Combine outputs from Inceptionv3 and VGG-16 to a result list.
			Argument:
			query_features: query image fingerprint from VGG-16
			probabilities: Inception's category probabilities

			Return: final list containing category, inception's confidence,
					recipe id, image index and image path.
		"""
		# do a query on image
		idx_closest, distances = self.get_closest_images_nmslib(query_features, self.k)

		# Don't forget to adjust string slicing for second hdf5
		# Labels only from ANN
		predicted_labels = [str(self.images[i]).split('/')[1] for i in idx_closest]

		# Results only from ANN
		predicted_ids = [[str(self.images[i]).split('-')[1],
						str(self.images[i]).split('-')[2].split('.')[0], 
						self.images[i].decode("utf-8")] for i in idx_closest]

		# Results only from Inception
		pred_categories = []

		for i, x in enumerate(np.argsort(-probabilities)[:self.k]):
			confidence = -np.sort(-probabilities)[i]
			#print(self.categories[x], confidence)
			pred_categories.append([self.categories[x], confidence])

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

		final_result_ann = [] # food result without weighting inceptions output, just ANN

		for i in predicted_ids:
			i.insert(0, 0)
			i.insert(0, 0)
			final_result_ann.append(i)

		return final_result, pred_categories, idx_closest, distances

	def model_predict(self, query_img_path):
		"""Returns the final food result."""
		query_image, x = self.get_image_vgg(query_img_path)
		query_features = self.feat_extractor.predict(x)
		# project it into pca space
		pca_query_features = self.ipca.transform(query_features)[0]

		img, x = self.get_image_inc(query_img_path) # Preprocess query image for Inception
		probabilities = self.model_inc.predict(x)[0] # Get Inception's category probabilities not sorted

		final_result, inc_result, idx_closest, distances = self.weighting_neural_net_inputs(pca_query_features, probabilities) # Get final food result
		ann_result = {}
		ann_result['ann_ids'] = idx_closest.tolist()
		ann_result['distances'] = distances.tolist()
		return final_result, inc_result, ann_result

