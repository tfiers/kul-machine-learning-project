
import url_predictor as pred
import json
from csv import *
from numpy import average

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
	validation = { '1' : 0, '2' : 0, '3': 0}
	distances = []

	for i in range(len(pages)-1):
		page = pages[i]

		predictions = pred.get_guesses(page)

		j = 0
		maxJ = len(predictions)
		found = False
		while (not found) and (j < maxJ):
			print("j: " + str(j))
			web_page = predictions[j]
			for k in range(i+1,len(pages)-1):
				if web_page == pages[k]:
					validation[str(j+1)] += 1
					distances.append(k-i)
					print(str(i) + ": " + page + "\n")
					print("\t" + str(j+1) + ": " + web_page + "\n")
					print("\t" + str(k) + ": " + pages[k] + "\n")
					print("\t" + str(k-i) +  " pages in between" + "\n")
					found = True
					break
			j += 1

	return validation, average(distances)


    

def train_model(filename):
	"""
	"""
	pred.learn_from(open(filename))


def preprocess(filename):
	"""
	"""
	# Read the csv file as a list of strings.
	lines = open(filename, 'r').readlines()
	# Make a list of 'event' dictionaries.
	events = pred.parse(lines)
	page_visits = pred.make_page_visits(events)
	pages = [page_visit['url'] for page_visit in page_visits]

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

	
 



