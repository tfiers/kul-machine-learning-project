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


def handle_csv(csv_file):
    lines = csv_file.readlines()
    entries = parse(lines)
    cleaned_entries = clean(entries)
    learn(cleaned_entries)


def parse(lines):
    """ Expects a list of strings with comma separated data and 
    returns a list of dictionaries.
    """
    # The list we'll return.
    entries = []
    for line in lines:
        # Parse the line as JSON.
        # Add brackets to line so it is valid JSON.
        data = json.loads('[{}]'.format(line))
        # Make a new dictionary and add it to the entries to be 
        # returned.
        entries.append({
            'ts': parse_datetime(data[0]), # timestamp
            'action': data[1], 
            'url': data[2],
            'target': data[3],
        })
    return entries


def clean(entries):
    """ Only retains the click entries for simplicity.
    """
    return [entry for entry in entries if entry['action'] == 'click']


def learn(entries):
    """ Trains a model that predicts the most likely destination page
    for each starting url.
    """
    # We create a map from starting urls to a list of {url: count} dicts.
    # 'count' is the number of times the 'url' followed the starting url.
    for entry in entries:
        starting_url = entry['url']
        target_url = entry['target']
        # If the model does not yet contain the starting url..
        if starting_url not in model:
            # .. add an empty dictionary for this url.
            model[starting_url] = {}
        # If the dictionary for this starting url does not yet have an
        # entry for the target url..
        if target_url not in model[starting_url]:
            # .. add a new entry initiliased to zero for this target url.
            model[starting_url][target_url] = 0
        # Increment the number of times the target url followed the 
        # starting url.
        model[starting_url][target_url] += 1
