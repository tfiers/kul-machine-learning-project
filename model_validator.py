
import url_predictor as pred
import json
import os
from csv import *
from numpy import average,median

def train_and_validate(csv_file_training,csv_file_validate):
	"""
	"""
	# Train the prediction model with the data from the training set
	train_model(csv_file_training)
	# Preprocess the data in the test set
	pages = preprocess(csv_file_validate)
	# Validate the model on the test data
	# Returns a dictionary with the number of primary, secondary and tertiary 
	#	predictions that were correct
	validation = validate(pages)

	return validation

def validate(pages):
	"""
	Expects a list of web pages representing a web session
	Calculates the number of correct predictions
	Returns a dictionary with the number of primary, secondary and tertiary 
		predictions that were correct
	"""
	print(len(pages))
	# A dictionary for recording the number of correct predictions
	validation = 0
	distances = []

	for i in range(len(pages)-1):
		page = pages[i]

		print("PAGE: " + "i:" + str(i) + " " + page)
		predictions = pred.get_guesses(page)

		j = 0
		maxJ = len(predictions)
		found = False
		while (not found) and (j < maxJ):
			web_page = predictions[j]
			for k in range(i+1,len(pages)-1):
				if web_page == pages[k]:
					validation += 1
					distances.append(k-i)
					print("\t WEB PAGE: " + str(j+1) + ": " + web_page + "\n")
					found = True
					break
			j += 1

	return validation, own_average(distances)

def own_average(lst):
	if len(lst) > 0:
		return average(lst)
	else:
		return 0

def validate_directory(directory="data"):
	"""
	"""
	prediction_list = []
	distance_list = []

	users = {}
	for file in os.listdir("./"+directory):
		if file.endswith(".csv"):
			s = file.split('_')
			if s[0] not in users:
				users[s[0]] = ["./"+directory+"/"+file]
			else:
				users[s[0]].append("./"+directory+"/"+file)

	print(str(users))
	for user in users:
		result = validate_user(users[user])
		prediction_list.append(result[0])
		distance_list.append(result[1])

	return users

def validate_user(user_files):
	"""
	"""
	pred.clear_model()

	prediction_list = []
	distance_list = []


	page_counter = 0
	for file in user_files:
		openfile = open(file).readlines()
		page_counter += len(openfile)

	# The test data set is the last 30% of the total data set
	test_data_start = int(round(0.7 * page_counter))

	page_counter = 0
	for file in user_files:
		openfile = open(file).readlines()
		fraction = (test_data_start - page_counter) / len(openfile)
		print(fraction)
		if fraction >= 1:
			pred.learn_from(open(file),1)
		elif fraction > 0:
			pred.learn_from(open(file),fraction)
			pages = preprocess(file,fraction)
			result = validate(pages)
			#print("RESULT: " + str(result))
			prediction_list.append(result[0])
			distance_list.append(result[1])
		else:
			pages = preprocess(file,fraction)
			result = validate(pages)
			prediction_list.append(result[0])
			distance_list.append(result[1])

	#print("PREDICTION LIST:" + str(prediction_list))
	#print("DISTANCE LIST:" + str(distance_list))
	return int(round(average(prediction_list))), int(round(average(distance_list)))



def train_model(filename):
	"""
	"""
	pred.clear_model()
	pred.learn_from(open(filename))


def preprocess(filename,fraction=0):
	"""
	"""
	# Read the csv file as a list of strings.
	lines = open(filename, 'r').readlines()
	# Make a list of 'event' dictionaries.
	events = pred.parse(lines)
	page_visits = pred.make_page_visits(events)

	n = int(round(fraction*len(page_visits)))

	pages = [page_visit['url'] for page_visit in page_visits[n:]]

	return pages
	
def parse(lines):
	"""
	Expects a list of string with comma separated data.
	Returns a list of url strings representing the sequence of web pages
	"""

	# List of web pages we'll return
	pages = []
	for line in lines:
		# Parse the line as JSON.
		# Add brackets to line so it is valid JSON.
		data = json.loads('[{}]'.format(line))
		# Add the url to the list of pages
		pages.append(data[2])

	return pages
