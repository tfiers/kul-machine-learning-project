
import url_predictor as pred
import json
from csv import *

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


def validate(pages):
	"""
	Expects a list of web pages representing a web session
	Calculates the number of correct predictions
	Returns a dictionary with the number of primary, secondary and tertiary 
		predictions that were correct
	"""
	# A dictionary for recording the web pages that are predicted
	page_prediction_history = {}
	# A dictionary for recording the number of correct predictions
	validation = { '1' : 0, '2' : 0, '3': 0}
	for page in pages:
		# Check if page is a predicted page
		# Loop over all primary, secondary and tertiary predictions
		for key in page_prediction_history:
			if page == key:
				

		# Get the predictions for this page
		# This returns a list of lists
		predictions = pred.get_guesses(page)
		#Store predictions in 


    

def train_model(filename):
	"""
	"""
	with open(filename, 'r') as csv_file:
		pred.handle_csv(csv_file)

	print("test")


def preprocess(filename):
	"""
	"""
	# Read the csv file as a list of strings.
	lines = open(filename, 'r').readlines()
	# Make a list of 'event' dictionaries.
	pages = parse(lines)

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

	
 



