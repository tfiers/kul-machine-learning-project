
import url_predictor as pred
from csv import *

def train_and_validate(csv_file_training,csv_file_validate):
	"""
	"""
	train_model(csv_file_training)

    

def train_model(filename):
	f = open(filename, 'r')

	pred.handle_csv(f)

	print "test"




def preprocess(filename):
	print "test"

	f = open(filename, 'r')

	lines = f.readlines()

	events = pred.parse(lines)

	return events
	

	
 



