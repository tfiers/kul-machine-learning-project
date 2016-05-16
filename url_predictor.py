# Activate anaconda environment 'kul-ml'.
# (On Python 2): from __future__ import division
# For all divisions. Otherwise this could should work on Py2.

import json
from dateutil.parser import parse as parse_datetime
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
    nodes = {}


def learn_from(csv_file_handle, fraction=1):
    """ Trains a model -- that predicts the most likely destination 
    pages for each starting url -- on a csv file.
    (The csv file should be supplied as a file handle.
    The model is stored in the global variable 'nodes').


    Repeated calls will expand the existing model.

    The model is only trained on the first 'fraction' page visits
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
    add_to_graph(page_visits[:n])
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
        if curent_event['event_type'] in start_events:
            next_event = events[i+1]
            # Determine the amount of time the user stayed on this 
            # page.
            timedelta = next_event['t'] - curent_event['t']
            duration = timedelta.total_seconds()
            # Create a new page visit object.
            page_visits.append({
                'url': curent_event['url'],
                't': curent_event['t'], # = time of page entry
                'duration' : duration,
            })
    return page_visits


def add_to_graph(page_visits):
    """ Constructs a directed graph from the page visits list.
    An existing graph will be extended.
    """
    # The graph is a dictionary of nodes. The keys are page urls,
    # the values contain urls of pages following the page of the key 
    # url, and various metadata (most of which is added later).
    # (This is called an 'adjacency list' representation.)
    for i in range(len(page_visits)):
        # Get the url of the currently visited page.
        url = page_visits[i]['url']
        time_on_page  = page_visits[i]['duration']
        # If the visited page is not yet a node in the graph..
        if url not in nodes:
            # .. add a new node, with currently one visit.
            nodes[url] = {
                'num_visits': 1,
                'time_on_page_data': [time_on_page],
                'domain': get_domain(url),
                'direct_links': {},
            }
        else:
            # If it is is already a node in the graph,  increase the 
            # number of times it was visited.
            nodes[url]['num_visits'] += 1
            # And add how long the user stayed on this page during 
            # this page visit.
            nodes[url]['time_on_page_data'].append(time_on_page)
        # Get a reference to the urls linked from the current node.
        linked_urls = nodes[url]['direct_links']
        # If this is not the last page visit of the browsing session..
        if i < len(page_visits)-1: 
            # .. get the url of the page visited next.
            next_url = page_visits[i+1]['url']
            # If there is already a link from the current page to the 
            # other url..
            if next_url in linked_urls:
                # .. increase the number of times the page at 
                # 'next_url' followed the current page.
                linked_urls[next_url] += 1
            else:
                # If not, add a new link. (Currently followed once).
                linked_urls[next_url] = 1
    # Return the graph.
    return nodes

def add_page_visit_to_graph():
    pass


previous_url = ''
previous_t = None
def add_one_url_to_model(url):
    """ 
    """
    t = datetime.datetime.now()
    page_visit = {
        'url': curent_event['url'],
        't': curent_event['t'], # = time of page entry
        'duration' : duration,
    }
    previous_url = url
    previous_t = t



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

def get_guesses(url_1, beta=1.1, max_len=12):
    """ Given a starting url 'url_1', returns a list of most likely
    destination pages.

    A higher 'beta' (> 1) gives a higher weight to destination pages
    further away.

    Only desination pages that are at least within 'max_len' steps 
    of 'url_1' will be considered.
    """

    # Incremental learning.
    add_one_url_to_model(url_1)

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
    time_on_page_data = [dt for P in Ps for dt in P['time_on_page_data']]
    global_avg_time_on_page = average(time_on_page_data)
    tot_time = sum(time_on_page_data)

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
            # length 1, hence the "- 1".
            k = len(path) - 1
            probabilities[k] += probability(path)
        # Average these probabilities over all considered path lengths.
        p_travel = average(probabilities)

        # Calculate the relative amount of times P2 was visited 
        # in the analysed log(s).
        N_rel = P2['num_visits'] / avg_visits
        # Calculate the proportion of times P2 was visited
        N_prop = P2['num_visits'] / tot_visits

        # We work with 'squashed' times below. See the doctext of 
        # 'squash'.
        # Calculate the relative average duration of stay on P2.
        avg_time_on_page = average(P2['time_on_page_data'])
        dt_rel = squash(avg_time_on_page) / squash(global_avg_time_on_page)
        # Calculate the proportion of time the user was on P2.
        tot_time_on_page = sum(P2['time_on_page_data'])
        dt_prop = squash(tot_time_on_page) / squash(tot_time)

        # Check if url_1 and url_2 are on the same domain.
        same_domain = (P1['domain'] == P2['domain'])

        # Save features in dictionary for later lookup.
        destinations[url_2]['N_rel']        = N_rel
        destinations[url_2]['N_prop']       = N_prop
        destinations[url_2]['dt_rel']       = dt_rel
        destinations[url_2]['dt_prop']      = dt_prop
        destinations[url_2]['same_domain']  = same_domain
        destinations[url_2]['p_travel']     = p_travel

        p_P1 =   sum(P1['time_on_page_data']) / tot_time \
               * P1['num_visits'] / tot_visits

        # --- Calculate score(s) ---
        # 
        # Bayes probability score.
        # This is p(D=P2 | S=P1)
        destinations[url_2]['Bayes_p'] = \
            p_travel                 * N_prop * dt_prop   #/ p_P1
        #   p(S=P1 | D=P2)           * p(D=P2)             / p(S=P1)
        # (The denominator is omitted as this is independent of P2).

        # Modify the Bayes probability score by giving a higher 
        # weight to pages further away (as predicting these is more
        # useful to the user.)
        len_shortest_path = min(len(path)-1 \
                            for path in destinations[url_2]['paths'])
        destinations[url_2]['len_weighted_Bayes_score'] = \
            destinations[url_2]['Bayes_p'] * beta**len_shortest_path

    # Make a list of candidate guesses.
    candidates = [url_2 for url_2 in destinations \
                if destinations[url_2]['Bayes_p'] > 0]
    candidates = sorted(candidates, reverse=True, key=lambda url_2: \
                        destinations[url_2]['len_weighted_Bayes_score'])

    # Take the x highest scoring candidates.
    x = min(3, len(guesses))
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
    print('   N_rel  dt_rel  N_prop dt_prop  same_d  p_trav')
    #      -------|-------|-------|-------|-------|-------|
    print()
    for url_2 in guesses:
        md = destinations[url_2] # destination metadata
        print(url_2)
        print()
        print(''.join(['{:>8.3f}']*6).format(md['N_rel'],
                                             md['dt_rel'],
                                             md['N_prop'],
                                             md['dt_prop'],
                                             md['same_domain'],
                                             md['p_travel']))
        print('Bayes p: {:.6f}'.format(md['Bayes_p']))
        print()
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
    return tanh(t / 0.5499)


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
    t0 = time()
    # for url in nodes:
    #     get_guesses(url)
    get_guesses('http://www.standaard.be/')
    print(time()-t0)

    # Timing "get_guesses('http://www.standaard.be/')":
    # 14.93 seconds without memoization
    #  0.05 seconds with memoization (298x faster)

    # lines = open('u23_1.csv').readlines()
    # events = url_predictor.parse(lines)
    # page_visits = url_predictor.make_page_visits(events)
    # for pv in page_visits:
    #    print(url_predictor.shorten_url(pv['url'], 80))

if __name__ == '__main__':
    test()
