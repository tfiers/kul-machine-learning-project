
import url_predictor as pred
import json
import os
from csv import *
from numpy import average,median
import matplotlib.pyplot as plt

def train_and_validate(csv_file_training,csv_file_validate,beta=1.1,max_len=12,alpha=0):
	"""
	"""
	# Train the prediction model with the data from the training set
	train_model(csv_file_training)
	# Preprocess the data in the test set
	pages = preprocess(csv_file_validate)
	# Validate the model on the test data
	# Returns a dictionary with the number of primary, secondary and tertiary 
	#	predictions that were correct
	validation = validate(pages,beta,max_len,alpha)

	return validation

def validate(pages,beta=1.1,max_len=12,alpha=0):
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
		predictions = pred.get_guesses(page,beta,max_len,alpha)

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

	return [validation, own_average(distances), len(pages)]


def validate_directory(directory="data"):
	"""
	"""
	prediction_dict = {}

	users = {}
	for file in os.listdir("./"+directory):
		if file.endswith(".csv"):
			s = file.split('_')
			if s[0] not in users:
				users[s[0]] = ["./"+directory+"/"+file]
			else:
				users[s[0]].append("./"+directory+"/"+file)

	for user in users:
		print(user)
		result = validate_user(users[user])
		prediction_dict[user] = [result[0],result[1],result[2]]

	return prediction_dict

def validate_user(user_files):
	"""
	"""
	valid_user_files = []
	for file in user_files:
		try:
			pred.parse(open(file).readlines())
			valid_user_files.append(file)
		except:
			pass
	pred.clear_model()

	prediction_list = []
	distance_list = []


	nb_pages = 0
	for file in valid_user_files:
		openfile = open(file).readlines()
		nb_pages += len(openfile)

	# The test data set is the last 30% of the total data set
	test_data_start = int(round(0.7 * nb_pages))

	nb_test_lines = 0
	page_counter = 0
	for file in valid_user_files:
		print("VALID FILE: " + file)
		openfile = open(file).readlines()
		if len(openfile) > 0:
			fraction = (test_data_start - page_counter) / len(openfile)
			page_counter += len(openfile)
			if fraction >= 1:
				print("1111111")
				pred.learn_from(open(file),1)
			elif fraction > 0:
				print("2222222")
				pred.learn_from(open(file),fraction)
				pages = preprocess(file,fraction)
				nb_test_lines += len(pages)
				result = validate(pages)
				prediction_list.append(result[0])
				distance_list.append(result[1])
			else:
				print("3333333")
				pages = preprocess(file,0)
				nb_test_lines += len(pages)
				result = validate(pages)
				prediction_list.append(result[0])
				distance_list.append(result[1])

	#print("PREDICTION LIST:" + str(prediction_list))
	#print("DISTANCE LIST:" + str(distance_list))
	return [sum(prediction_list), own_average(distance_list), nb_test_lines]

#### EXPERIMENTS ####

def experiment_beta(csv_file_training,csv_file_validate,minI,maxI):
	lst = []
	i = minI
	while i < maxI:
		result = train_and_validate(csv_file_training,csv_file_validate,i)
		lst.append([i,result[0],result[1],result[2]])
		i += 0.1

	# matplotlib
	xlist = []
	ylist1 = []
	ylist2 = []
	for l in lst:
		xlist.append(l[0])
		ylist1.append(int(round(l[1]/l[3]*100)))
		ylist2.append(l[2])

	print(xlist)
	print(ylist1)
	accuracy = plt.plot(xlist,ylist1,'r',label='Accuracy (%)')
	distance = plt.plot(xlist,ylist2,'b',label='Distance (nb clicks)')
	plt.axis([0,10,0,28])
	plt.legend()
	plt.show()

	return lst

def experiment_alpha(csv_file_training,csv_file_validate,minI,maxI):
	lst = []
	i = minI
	while i < maxI:
		result = train_and_validate(csv_file_training,csv_file_validate,1.1,12,i)
		lst.append([i,result[0],result[1],result[2]])
		i += 0.1

	# matplotlib
	xlist = []
	ylist1 = []
	ylist2 = []
	for l in lst:
		xlist.append(l[0]-1)
		ylist1.append(int(round(l[1]/l[3]*100)))
		ylist2.append(l[2])

	print(xlist)
	print(ylist1)
	accuracy = plt.plot(xlist,ylist1,'r',label='Accuracy (%)')
	distance = plt.plot(xlist,ylist2,'b',label='Distance (nb clicks)')
	plt.axis([-1,9,0,25])
	plt.legend()
	plt.show()

	return lst


#### UTILITY FUNCTIONS ####


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


def own_average(lst):
	if len(lst) > 0:
		return average(lst)
	else:
		return 0
