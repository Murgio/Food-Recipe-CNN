import os.path
import json
from datetime import datetime

class LogFood:
	"""Logs useful information for data analysis.

	# Arguments
		log_record: Dictionary which contains the information. Clears out after every request.
		path: Path to the json file.
	"""
	def __init__(self, log_record={}, path=''):
		self.log_record = log_record
		self.path = path

	def new_request(self):
		"""Create new dictionary and time stamp."""
		self.log_record['request'] = {}
		self.log_record['request']['time_stamp'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')

	def new_calc_time(self, food_time):
		"""Time needed for calculating the result food list."""
		self.log_record['request']['calc_time_seconds'] = food_time

	def new_food_ids(self, food_ids):
		"""Result food ids."""
		self.log_record['request']['result_food_ids'] = food_ids

	def new_image_indexes(self, image_indexes):
		"""Result food image indexes."""
		self.log_record['request']['image_indexes'] = image_indexes

	def new_inc_result(self, inc_result):
		"""Predicted categories with probability."""
		self.log_record['request']['inc_result'] = inc_result

	def new_ann_result(self, ann_result):
		"""Closest neighbor indexes and distance."""
		self.log_record['request']['ann_result'] = ann_result

	def get_log_record(self):
		return self.log_record

	def flush(self):
		"""
			https://stackoverflow.com/questions/18087397
			This opens the file for both reading and writing. Then, it goes to the end
			of the file (zero bytes from the end) to find out the file end's 
			position (relatively to the beginning of the file) and goes 
			last one byte back, which in a json file is expected to represent 
			character ]. In the end, it appends a new dictionary to the structure, 
			overriding the last character of the file and keeping it to be valid json. 
			It does not read the file into the memory.
		"""
		with open(self.path, 'r+') as json_file:
			json_file.seek(0,2)
			position = json_file.tell() - 2
			json_file.seek(position)
			json_file.write( ",{}]}}".format(json.dumps(self.log_record, sort_keys=True, indent=4)))
			self.log_record = {}