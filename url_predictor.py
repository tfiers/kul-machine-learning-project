# Activate anaconda environment 'kul-ml'.
# (On Python 2): from __future__ import division
# For all divisions. Otherwise this could should work on Py2.

import json
from dateutil.parser import parse as parse_datetime
from collections import defaultdict
from time import time
from numpy import average, median

from urllib.parse import urlsplit
from publicsuffix import PublicSuffixList
psl = PublicSuffixList()
def get_domain(url):
    """ Extracts the domain name part of the given URL.
    """
    # 'netloc' = subdomain(s) plus domain name.
    netloc = urlsplit(url).netloc
    # Extract only the domain name itself ('public suffix').
    # See http://stackoverflow.com/a/15460894/2611913
    domain_name = psl.get_public_suffix(netloc)
    return domain_name


# Our graph and model. Every node corresponds with a web page.
nodes = {}
# 
# Structure of a 'nodes' entry, after all calculation for it is done
# (all values have a default value for their data type here):
# 
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


# Three possiblities:
#       'paths': {
#           #          p    k    p    k
#           'url_B': [(0.2, 3), (0.1, 6)]
#       }
# 
# OR:
# 
#       'paths': {
#        #  k              p     p          
#           1: { 'url_B': [0.15, 0.3]}
#           2: {}
#       }
# 
# OR:
# 
#       'paths': {
#            'url_B': [('url', 'url'), ('url', 'url')]
#        }


# Two possibilities for filling the 'paths' value of each node:
# 1) Node per node
# 2) k per k
# 
# 1) > generate_paths(travel_history, max_steps)
#       Each function adds entries for it's step = k = len(travel_history)
#       to the first url of the travel history.
#       Initial call: generate_paths((url_1,), k)  (k=10 eg)
# 
#       Optimisation: we can exploit fact that: after a node (P1) is done,
#       it's stored paths are complete. We can use these when calculating
#       paths from another node (P2): Eg. Direct link from P2 to P1
#       -> we can get all paths of length up to k starting with P2->P1 
#       by prepending P2 to all paths of length up to k-1 starting from P1.
#       FOR THIS OPTIMISATION WE NEED THE SECOND DATA STRUCTURE.
#       (The others would be kindof a pain to use).
#       --> Less and less traversals (jumps through memory) when completing
#       the 'paths' value for more nodes.
#       (Maybe: not much point in calculating and storing these paths
#       beforehand. Maybe do it on the fly.
#       Allright: first do it on the fly, then see: if fast enough: 
#       leave it, it's good. If not fast enough, do precalc, and try 
#       optimisation. This one, or '2)' below).
# 
# 2) > k = 1  -> for each node: copy the 'direct_links' to k=1 entry
#    > k = 2  -> for each node:  for each k=1 entry: append with results 
#                of lookup at that entry's k=1. Add all results to k=2 entry here.
#    > k = 3  -> for each node:  for each k=2 entry: append with results 
#                of lookup at that entry's k=1. Add all results to k=3 entry here.
#    > etc.
#    
#    Optimisation.
#    Question: can we exploit the fact that, when we've 
#    calculated all paths up to length k = 4 for each node (eg.),
#    we could, for each node, quickly calculate all paths of length
#    5 (4+1), 6 (4+2), 7 (4+3) and 8 (4+4).
#    Maybe, let's see.
#    > k = 1
#    > k = 2 (1+1)
#    > k = 3 (2+1) and k = 4 (2+2)
#    > k = 5, k = 6, k = 7, k = 8
#    > k = 9, k = 10, k = 11, k = 12, k = 13, k = 14, k = 15, k = 16
#    > etc.
# 
# 
# For lookups (calculating score(url1->url2)), we'd do:
# - For data structure 1 (and 3) above: simple lookup at url_2
# - For data structure 2 above: loop over all k's, lookup url_2 each time.
# Ok, so lookup ease is no big argument to prefer 1.
# --> We go for data structure 2.
# 
# I guess: not optimised (not precomputed) method 1) will be fast enough
# for on the fly calculation: o^k lookups.
# where o is mean branching factor per page. Take 0=2, k=10: 1024 lookups.
# (For k=16 -> 65 500 lookups.  k=20 -> 1M lookups)
# Algo needn't be recursive by the way. That's nice ( :) )
# 
# 
# Conclusion: use non-precomputet method 1), non recursively.
# No need to store paths.


def learn_from(csv_file_handle):
    """ Trains a model -- that predicts the most likely destination 
    pages for each starting url -- on a csv file.
    (The csv file should be supplied as a file handle.
    The model is stored in the global variable 'nodes').
    """
    # PREPROCESSING
    # 
    # Read the csv file as a list of strings.
    lines = csv_file_handle.readlines()
    # Make a list of 'event' dictionaries.
    events = parse(lines)
    # Make a list of 'page visit' dictionaries.
    page_visits = make_page_visits(events)
    # Make a graph (as a dictionary of nodes), discarding absolute
    # temporal information and higher-order sequential information.
    # (We modify the global object 'nodes' here).
    make_graph(page_visits)

    # For debugging in shell.
    return nodes


    # TRAINING MODEL
    # 
    # ... Done on the fly for each starting url, for now.


def parse(lines):
    """ Expects a list of strings with comma separated data.
    Returns a list of dictionaries.
    """
    # The list we'll return.
    events = []
    # Loop over all non-empty lines.
    for line in filter(lines):
        # Parse the line as JSON.
        # Add brackets to line so it is valid JSON.
        data = json.loads('[{}]'.format(line))
        print(line)
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
            # Determine how this page was left.
            if next_event['event_type'] == 'click':
                exit_type = 'click'
            elif next_event['event_type'] == 'polling':
                exit_type = 'forward'
            else:
                exit_type = 'cut'
            # Create a new page visit object.
            page_visits.append({
                'url': curent_event['url'],
                't': curent_event['t'], # = time of page entry
                'duration' : duration,
                'exit_type': exit_type,
            })
    return page_visits


def make_graph(page_visits):
    """ Constructs a directed graph from the page visits list.
    """
    # The graph is a dictionary of nodes. The keys are page urls,
    # the values contain urls of pages following the page of the key 
    # url, and various metadata (most of which is added later).
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




# --------------------------------------------------------------------


def get_guesses(url_1):
    """ Given a starting url 'url_1', returns a list of most likely
    destination pages. """
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
    # being the page the user wants to go, given he is at 'url_1'.
    destinations = {}

    # Generate all paths up to a certain depth starting from 'url_1'.
    max_len = 12
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
        # Calculate the probability of each path.
        # Sum these probabilities, giving a higher weight to longer
        # paths (as these are more useful to the user).
        p_travel = sum(probability(path) * 1.1**len(path) \
                       for path in destinations[url_2]['paths'])
        
        # Calculate the relative amount of times P2 was visited 
        # in the analysed log(s).
        N_rel = P2['num_visits'] / avg_visits
        # Calculate the proportion of times P2 was visited
        N_prop = P2['num_visits'] / tot_visits 

        # Calculate the relative average duration of stay on P2.
        avg_time_on_page = average(P2['time_on_page_data'])
        dt_rel = avg_time_on_page / global_avg_time_on_page
        # Calculate the proportion of time the user was on P2.
        tot_time_on_page = sum(P2['time_on_page_data'])
        dt_prop = tot_time_on_page / tot_time

        # Check if url_1 and url_2 are on the same domain.
        same_domain = (P1['domain'] == P2['domain'])

        # [Calculate more features]

        # Save features in dictionary for later lookup.
        destinations[url_2]['N_rel']        = N_rel
        destinations[url_2]['N_prop']       = N_prop
        destinations[url_2]['dt_rel']       = dt_rel
        destinations[url_2]['dt_prop']      = dt_prop
        destinations[url_2]['same_domain']  = same_domain
        destinations[url_2]['p_travel']     = p_travel

        # [Calculate score(s)]
        # p(D=P2 | S=P1)
        destinations[url_2]['Bayes_p'] = \
            p_travel * same_domain   * N_prop * dt_prop \
        / (sum(P1['time_on_page_data'])/tot_time 
            * P1['num_visits'] / tot_visits)
        #   p(S=P1 | D=P2)           * p(D=P2)             / p(S=P1)
        # (The denominator is omitted as this is independent of P2).

    guesses = [url_2 for url_2 in destinations]
    guesses = sorted(guesses, reverse=True, 
                     key=lambda url_2: destinations[url_2]['Bayes_p'])

    # Return the x highest scoring candidates.
    # Print info about them.
    x = 3

    print()
    print('   N_rel  dt_rel  N_prop dt_prop  same_d  p_trav')
    #      -------|-------|-------|-------|-------|-------|
    print()
    for url_2 in guesses[:x]:
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

    return guesses[:x]


def sigmoid(t):
    """ Returns the 3.475
    """
    pass


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


# Execute on import:
learn_from(open('u23_1.csv'))
