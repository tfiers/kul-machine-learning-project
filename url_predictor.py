# Activate anaconda environment 'kul-ml'.

import json
from dateutil.parser import parse as parse_datetime

model = {}


def get_guesses(url):
    """ Given a url, returns a list of the most probable destination
    urls. Each entry is of the form [url, probability_score].
    The entries are sorted from highest to lowest probabibility.

    Example:
    [['https://dtai.cs.kuleuven.be/events/leuveninc', 0.9],
     ['https://google.com', 0.5]]
    """
    if url not in model:
        return []
    else:
        results = []
        target_url_counts = model[url]
        for url, count in target_url_counts.items():
            results.append([url, count])
        # Sort from highest to lowest count.
        results.sort(key=lambda e: e[1], reverse=True)
        return results


def handle_csv(csv_file_handle):
    """ Trains a model on a csv file.
    (The csv file should be supplied as a file handle.
    The model is stored in the global variable 'model').
    """
    preprocessed_data = preprocess(csv_file_handle)
    learn(preprocessed_data)


def parse(lines):
    """ Expects a list of strings with comma separated data.
    Returns a list of dictionaries.
    """
    # The list we'll return.
    events = []
    for line in lines:
        # Parse the line as JSON.
        # Add brackets to line so it is valid JSON.
        data = json.loads('[{}]'.format(line))
        # Make a new dictionary and add it to the events to be 
        # returned.
        events.append({
            't': parse_datetime(data[0]),
            'event_type': data[1], 
            'url': data[2],
            'target': data[3],
        })
    return events


def make_page_visits(events):
    """ Constructs page visit objects (as dictionaries).
    See the "Data" section in the report and the accompanying figure.
    """
    # The list we'll fill with page visit objects.
    page_visits = []
    # Iterate over all events, stopping at 'load' or 'polling' ones.
    # These events define the start of a page visit.
    start_events = ['load', 'polling']
    for i in range(len(events)):
        curent_event = events[i]
        if curent_event['event_type'] in start_events:
            next_event = events[i+1]
            # Determine the amount of time the user stayed on this 
            # page. The result is a 'timedelta' object.
            dt = next_event['ts'] - curent_event['ts']
            # Determine how this page was left.
            if next_event['event_type'] == 'click':
                exit_type = 'click'
            else:
                exit_type = 'cut'
            # Create a new page visit object.
            page_visits.append({
                'url': curent_event['url'],
                't': curent_event['t'], # = time of event
                'dt' : dt, # dt = delta-t = duration of stay
                'exit_type': exit_type,
            })
    return page_visits


def preprocess(csv_file_handle):
    """ 
    """
    # Read the csv file as a list of strings.
    lines = csv_file.readlines()
    # Make a list of 'event' dictionaries.
    events = parse(csv_file_handle)
    # Make a list of 'page visit' dictionaries.
    page_visits = make_page_visits(events)


def learn(entries):
    """ Trains a model that predicts the most likely destination page
    for each starting url.
    """
    pass
