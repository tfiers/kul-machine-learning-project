# Predictive Web Browsing
University of Leuven – Machine Learning Project

## Setup instructions
Make sure you are running **Python 3**.

Run `pip install -r requirements.txt`.  

Add and enable the user script `urlStreamHandler.user.js` in your browser. (Using Greasemonkey in Firefox, for example).  

Run `python urlStreamHandler.py`. 

Now after a while, the app will start suggesting pages you might want to go to when you are on a page that you have already visited.  

If you want to pre-train the app with historical web usage data, call for example:  
`python urlStreamHandler.py --csv log1.csv log2.csv`.


## File descriptions
The preprocessing and prediction code can be found in `url_predictor.py`.  
The code used to validate the model can be found in `model_validator.py`.

The hours we worked on this project can be found [here](https://docs.google.com/spreadsheets/d/1L8jPud9az4ppOUDhL3XCcTqYnGwVR8CMIRk6lBs5M20/edit?usp=sharing).
