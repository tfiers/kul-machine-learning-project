# Activate anaconda environment 'kul-ml'.
# (On Python 2): from __future__ import division
# For all divisions. Otherwise this could should work on Py2.

import json
from dateutil.parser import parse as parse_datetime
from datetime import datetime
from urllib.parse import urlsplit
from publicsuffix import PublicSuffixList
from collections import defaultdict
from time import time
from numpy import average, median
from math import tanh


# Our graph and model. Every node corresponds with a web page.
nodes = {}
# 
# Structure of a 'nodes' entry, after all calculation for it is done
# (all data has a default value for its data type here):
# 
# nodes = 
# {
#   'url_A': {
#       'time_on_page_data': [],
#       'avg_time_on_page': 0,
#       'num_visits': 0,
#       'domain': '',
#       'direct_links': {
#           'url_B': 0
#       },
#   }
# }


def clear_model():
    """ Clears the graph.
    """
    global previous_url
    nodes.clear()
    previous_url = ''



def learn_from(csv_file_handle, fraction=1):
    """ Trains a model -- that predicts the most likely destination 
    pages for each starting url -- on a csv file.
    (The csv file should be supplied as a file handle.
    The model is stored in the global variable 'nodes').


    Repeated calls will expand the existing model.

    The model is only trained on the first 'fraction' of page visits
    in the data.
    """
    # Read the csv file as a list of strings.
    lines = csv_file_handle.readlines()
    # Make a list of 'event' dictionaries.
    events = parse(lines)
    # Make a list of 'page visit' dictionaries.
    page_visits = make_page_visits(events)
    # Calculate the number of page visits that should be trained on.
    n = int(round(fraction*len(page_visits)))
    # Make a graph (as a dictionary of nodes), discarding absolute
    # temporal information and higher-order sequential information.
    # (We modify the global object 'nodes' here).
    add_page_visits_to_graph(page_visits[:n])
    # For debugging in shell.
    return nodes


def parse(lines):
    """ Expects a list of strings with comma separated data.
    Returns a list of dictionaries.
    """
    # The list we'll return.
    events = []
    # Loop over all non-empty lines.
    for line in filter(len, lines):
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
    """ Constructs page visit objects (as dictionaries) from a list
    of events.
    See the "Data" section in the report and its accompanying figure.
    """
    # The list we'll fill with page visit objects.
    page_visits = []
    # Iterate over all events, stopping at 'load' or 'polling' ones.
    # These events define the start of a page visit.
    start_events = ['load', 'polling']
    for i in range(len(events)):
        curent_event = events[i]
        if curent_event['event_type'] in start_events \
           and i < len(events)-1:
            next_event = events[i+1]
            # Determine the amount of time the user stayed on this 
            # page.
            duration = seconds_diff(next_event['t'], curent_event['t'])
            # Create a new page visit object.
            page_visits.append({
                'url': curent_event['url'],
                't': curent_event['t'], # = time of page entry
                'duration' : duration,
            })
    return page_visits


def add_page_visits_to_graph(page_visits):
    """ Constructs a directed graph from the page visits list.
    An existing graph will be extended.
    """
    # The graph is a dictionary of nodes. The keys are page urls,
    # the values contain urls of pages following the page of the key 
    # url, and various metadata (most of which is added later).
    # (This is called an 'adjacency list' representation.)
    for i in range(len(page_visits)):
        # Get the url of the currently visited page, and
        # the duration of the page visit.
        url = page_visits[i]['url']
        time_on_page  = page_visits[i]['duration']
        # Update the graph with this data
        add_visit_to_node(url, time_on_page)
        # Get a reference to the urls linked from the current node.
        linked_urls = nodes[url]['direct_links']
        # If this is not the last page visit of the browsing session..
        if i < len(page_visits)-1: 
            # .. get the url of the page visited next.
            next_url = page_visits[i+1]['url']
            # Add or update a link from the current url to the next.
            add_link_between_nodes(url, next_url)


previous_url = ''
previous_t = None
def learn_from_current_visit(url):
    """ For incremental learning. Update the model with the knowledge
    that the given url is currently being visited.
    """
    # We want some persistency between calls.
    global previous_url, previous_t
    # Get the current timestamp.
    t = datetime.now()
    # Update the graph. We don't know the duration of the page visit yet.
    add_visit_to_node(url, duration=None)
    # If we already visited a page:
    if previous_url:
        # We now know the duration of the previously visited page.
        duration = seconds_diff(t, previous_t)
        # Add this data to its corresponding node.
        nodes[previous_url]['time_on_page_data'].append(duration)
        # We also now have a new link from the previous url to the
        # current one.
        add_link_between_nodes(previous_url, url)
    # Update data for the next run.
    previous_url = url
    previous_t = t


def add_visit_to_node(url, duration=None):
    """ Update the node of the given url, or make a new one.
    """
    # If the visited page is not yet a node in the graph..
    if url not in nodes:
        # .. add a new node.
        nodes[url] = {
            'num_visits': 0,
            'time_on_page_data': [],
            'domain': get_domain(url),
            'direct_links': {},
        }
    # Increase the number of times it was visited.
    nodes[url]['num_visits'] += 1
    # If a page visit duration is provided, add this data to the node.
    if duration is not None:
        nodes[url]['time_on_page_data'].append(duration)


def add_link_between_nodes(url_1, url_2):
    """ Updates the link from url_1 to url_2 in the graph, or make
    a new one.
    """
    # Get a reference to the urls linked from url_1.
    linked_urls = nodes[url_1]['direct_links']
    # If there is already a link from the first url to the 
    # other url.self.
    if url_2 in linked_urls:
        # .. increase the number of times the page at 
        # 'url_2' followed the current page.
        linked_urls[url_2] += 1
    else:
        # If not, add a new link. (Currently followed once).
        linked_urls[url_2] = 1


def seconds_diff(t1, t0):
    """ Returns the total amount of seconds between the given
    timestamps.
    """
    timedelta = t1 - t0
    return timedelta.total_seconds()


psl = PublicSuffixList()
def get_domain(url):
    """ Extracts the domain name part of the given URL.
    """
    # 'netloc' is the subdomain(s) plus the domain name.
    netloc = urlsplit(url).netloc
    # Extract only the domain name itself ('public suffix').
    # See http://stackoverflow.com/a/15460894/2611913
    domain_name = psl.get_public_suffix(netloc)
    return domain_name


# --------------------------------------------------------------------

def get_guesses(url_1, beta=1.1, max_len=10, alpha = 0):
    """ Given a starting url 'url_1', returns a list of most likely
    destination pages.

    A higher 'beta' (> 1) gives a higher weight to destination pages
    further away.

    Only desination pages that are at least within 'max_len' steps 
    of 'url_1' will be considered.
    """

    # Incremental learning.
    learn_from_current_visit(url_1)

    # Predict nothing if we haven't seen the given url before.
    if url_1 not in nodes:
        return []

    # Get the metadata of the page at url_1.
    P1 = nodes[url_1]

    # All node metadata.
    Ps = nodes.values()

    # Calculate the average number of page visits per page
    # and the total number of page visits in the analysed log(s).
    num_visits_data = [P['num_visits'] for P in Ps]
    avg_visits = average(num_visits_data)
    tot_visits = sum(num_visits_data)

    # Calculate average time on page, averaged over all pages.
    # and total time spent on each page, summed over all pages.
    # (We flatten a list of lists on the next line.)
    # We continue with squashed times. See the docstring for 'squash'.
    time_on_page_data = [dt for P in Ps for dt in P['time_on_page_data']]
    squashed_durations = list(map(squash, time_on_page_data))
    global_avg_time_on_page = average(squashed_durations)
    tot_time = sum(squashed_durations)

    # A dictionary indexed by possible destination page.
    # The values contain data about the probability of this page
    # being the page that the user wants to go to, given he is at 
    # 'url_1'.
    destinations = {}

    # Generate all paths up to a certain depth starting from 'url_1'.
    paths = generate_paths(start_url=url_1, max_len=max_len)
    # Aggregate them by destination page.
    for k in range(1, max_len+1):
        for url_2 in paths[k]:
            new_paths = paths[k][url_2]
            if url_2 not in destinations:
                destinations[url_2] = {
                    'paths': new_paths
                }
            else:
                destinations[url_2]['paths'].extend(new_paths)

    # Remove the current url from the possible destinations.
    # (We don't want to predict the currrent url).
    destinations.pop(url_1, None)
    
    # Calculate and store features and score(s) for each possible
    # destination url.
    for url_2 in destinations:
        # Get the node metadata of the page at url_2.
        P2 = nodes[url_2]

        # --- Calculate features ---
        # 
        # For each path length k, calculate the probability that 
        # a random traveller gets from url_1 to url_2 in k steps.
        probabilities = [0]*max_len
        for path in destinations[url_2]['paths']:
            # A path from A to B, represented as ('A', 'B'), has 
            # length 1, hence the "- 1". We use a 0-indexed list 
            # to store the probabilities for convenience, hence
            # another "-1".
            k = len(path) - 1
            probabilities[k-1] += probability(path)
        # Average these probabilities over all considered path lengths.
        p_travel = average(probabilities)

        # Calculate the relative amount of times P2 was visited 
        # in the analysed log(s).
        N_rel = P2['num_visits'] / avg_visits
        # Calculate the proportion of times P2 was visited
        N_prop = P2['num_visits'] / tot_visits

        # Calculate the relative average duration of stay on P2.
        squashed_durations = list(map(squash, P2['time_on_page_data']))
        avg_time_on_page = average(squashed_durations)
        dt_rel = avg_time_on_page / global_avg_time_on_page
        # Calculate the proportion of time the user was on P2.
        tot_time_on_page = sum(squashed_durations)
        dt_prop = tot_time_on_page / tot_time

        # Check if url_1 and url_2 are on the same domain.
        same_domain = (P1['domain'] == P2['domain'])
        if same_domain:
            same_domain = 1 + alpha
        else:
            same_domain = alpha

        # Calculate the length of the shortest path between P1 and P2.
        minlen = min(len(path)-1 \
                     for path in destinations[url_2]['paths'])
        

        # Save features in dictionary for later lookup.
        destinations[url_2]['N_rel']        = N_rel
        destinations[url_2]['N_prop']       = N_prop
        destinations[url_2]['dt_rel']       = dt_rel
        destinations[url_2]['dt_prop']      = dt_prop
        destinations[url_2]['same_domain']  = same_domain
        destinations[url_2]['p_travel']     = p_travel
        destinations[url_2]['minlen']       = minlen

        p_P1 =   sum(P1['time_on_page_data']) / tot_time \
               * P1['num_visits'] / tot_visits

        # --- Calculate score(s) ---
        # 
        # Bayes probability score.
        # This is p(D=P2 | S=P1)
        destinations[url_2]['Bayes_p'] = \
            p_travel * same_domain * N_prop * dt_prop   #/ p_P1
        #   p(S=P1 | D=P2)           * p(D=P2)             / p(S=P1)
        # (The denominator is omitted as this is independent of P2).

        # Length weighted Bayes probability score.
        # Modify the Bayes probability score by giving a higher 
        # weight to pages further away (as predicting these is more
        # useful to the user.)
        destinations[url_2]['len_weighted'] = \
            destinations[url_2]['Bayes_p'] * beta**minlen

    # Make a list of candidate guesses.
    candidates = [url_2 for url_2 in destinations \
                if destinations[url_2]['Bayes_p'] > 0]
    candidates = sorted(candidates, reverse=True, key=lambda url_2: \
                        destinations[url_2]['len_weighted'])

    # Take the x highest scoring candidates.
    x = min(3, len(candidates))
    guesses = candidates[:x]
    # Print info about them.
    print_info(guesses, destinations)
    # Return them.
    return guesses


def print_info(guesses, destinations):
    """ Shows information about the given guesses in the console:
    what are the factors contributing to their scores?
    """
    print()
    for url_2 in guesses:
        md = destinations[url_2] # destination metadata
        print(url_2)
        print()
        print('   N_rel  dt_rel  N_prop dt_prop  same_d  p_trav  minlen')
        #      -------|-------|-------|-------|-------|-------|-------|
        print(''.join(['{:>8.3f}']*7).format(md['N_rel'],
                                             md['dt_rel'],
                                             md['N_prop'],
                                             md['dt_prop'],
                                             md['same_domain'],
                                             md['p_travel'],
                                             md['minlen']))
        print()
        print('Bayes p:      {:.3E}'.format(md['Bayes_p']))
        print('Len-weighted: {:.3E}'.format(md['len_weighted']))
        print()
        # print('Avg duration: {:.2f}'.format(average(
        #                         nodes[url_2]['time_on_page_data'])))
        # print('Squashed:     {:.2f}'.format(squash(average(
        #                         nodes[url_2]['time_on_page_data']))))
        print()


def squash(t):
    """ Applies a sigmoid function to the input value.
    For positive input values, the output will be between 0 and 1.

    Used to squash duration values: a page visit duration of 4 seconds
    should be given a clearly higher weight than a duration of 2
    seconds, BUT a duration of 40 seconds should yield only a slightly
    higher weight than a duration of 10 seconds.

    The function is calibrated so an input value of 3.475 yields 0.5.
    (3.475 is the median page visit time in one of the biggest data
    sets.)
    """
    # 0.5499 ~= inv_tanh(0.5)
    return tanh(t / 3.475 * 0.5499)


def generate_paths(start_url, max_len):
    """ Returns all paths of length up to the given maximum length 
    through the graph defined in 'nodes' starting in the given url.
    Paths may contain loops.
    """
    # The paths we'll return, indexed first by length k,
    # then by ending url.
    paths = {
        0: {
            start_url: [(start_url,)]
        }
    }

    for k in range(max_len):
        paths[k+1] = defaultdict(list)
        for last_url in paths[k]:
            for path in paths[k][last_url]:
                for next_url in nodes[last_url]['direct_links']:
                    paths[k+1][next_url].append(path + (next_url,))
    return paths


def probability(path):
    """ Calculates the probability that a random walker takes
    the given path trough the graph defined in 'nodes'.
    """
    p = 1.0
    for i in range(len(path)-1):
        current_url = path[i]
        next_url = path[i+1]
        total_visits = nodes[current_url]['num_visits']
        hops_to_next = nodes[current_url]['direct_links'][next_url]
        transition_p = hops_to_next / total_visits
        # print('{:.6f}  {}/{}  {}'.format(p, hops_to_next, 
        #                                  total_visits, current_url))
        p *= transition_p
    # print('{:.6f}       {}'.format(p, next_url))
    return p


def format_destination(url, score):
    """ Formats a destination url nicely for the UI, with some
    additional info for peeking inside the workings of the model.
    """
    return '{}  {}'.format(shorten_url(url), score)


def shorten_url(url, max_len=50):
    """ Leaves urls shorter than 'max_len' untouched.
    Longer urls get their middle part replaced by '...' so they
    are max_len long.
    
    max_len should be 5 or greater.
    """
    if len(url) <= max_len:
        return url
    else:
        n, o = divmod(max_len, 2)
        return '{}...{}'.format(url[:n-1], url[-n+2-o:])


def time_f(f, arg):
    t0 = time()
    f(*arg)
    print(time()-t0)


def test():
    learn_from(open('u23_1.csv'))
    # for url in nodes:
    #     get_guesses(url)
    get_guesses('http://www.standaard.be/')

    # lines = open('u23_1.csv').readlines()
    # events = url_predictor.parse(lines)
    # page_visits = url_predictor.make_page_visits(events)
    # for pv in page_visits:
    #    print(url_predictor.shorten_url(pv['url'], 80))

if __name__ == '__main__':
    test()
