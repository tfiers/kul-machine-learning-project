# Activate anaconda environment 'kul-ml'.
# (On Python2): from __future__ import division

import json
from dateutil.parser import parse as parse_datetime

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
#       'paths: {
#           'url_B': [('')]
#       }
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
    # Annotate each node with the average time spent on that page.
    # TODO: median, average, or max time on page?
    for P in nodes.values():
        durations = [t.total_seconds() for t in P['time_on_page_data']]
        P['avg_time_on_page'] = average(durations)

    # For debugging in shell.
    return nodes


    # TRAINING MODEL
    # 
    # ... Done on the fly for each starting url, for now.


def average(sequence):
    """ Returns the arithmetic mean of the given sequence of numbers.
    """
    return sum(sequence) / len(sequence) # Py3


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
            # page. The result is a 'timedelta' object.
            duration = next_event['t'] - curent_event['t']
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
                'paths': {},
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
    # Count total number of page visits in the data.
    num_visits_data = [P['num_visits'] for P in nodes.values()]
    global_visits = sum(num_visits_data)
    # Average time on page, averaged over all pages.
    avg_time_on_page_data = [P['avg_time_on_page'] for P in nodes.values()]
    avg_avg_time_on_page = average(avg_time_on_page_data)
    # The list of other urls we'll fill, annotated with scores.
    guesses = []
    # Loop over all other urls.
    # (To find the other url that maximises a score.)
    for url_2, P2 in nodes.items():
        # Don't predict the page the user is currently on:
        if url_2 == url_1:
            # just skip this url_2.
            continue
        # Calculate all possible paths without loops from url_1
        # to url_2 through the graph defined in 'nodes'.
        paths = find_paths(url_1, url_2)
        # Calculate the probability of each path.
        # Sum these probabilities, giving a higher weight to longer
        # paths (as these are more useful to the user).
        # (TODO: why exponential in length L?)
        # L         1    2    3    4     5    6     7     8     9  
        # weight
        # 1.1**L    1.1  1.2  1.3  1.4   1.6  1.7   1.9   2.1   2.3
        # 2*L       2    4    6    8    10   12    14    16    18  
        # L**2      1    4    9    16   25   36    49    64    81  
        # 2**L      2    4    8    16   32   64   128   256   512  
        p_rel = sum(probability(path) * 1.1**len(path) \
                    for path in paths)

        # 
        # TODO: marginale verdeling p(Xn=P2) is eigenlijk verdeling
        # over nodes op tijdstip n (we kunnen n->inf nemen).
        # ---Dit hier is p(X0=P2)---   -> Not. We hebben geen initiële 
        # verdeling (?)
        p_abs = P2['num_visits'] / global_visits # Py3

        # Check if url_1 and url_2 are on the same domain.
        same_domain = (P1['domain'] == P2['domain'])
        # Convert this into a number (where False->0 and True->1)
        D = same_domain+0.5

        # Calculate the relative duration of stay on P2.
        T = P2['avg_time_on_page'] / avg_avg_time_on_page # Py3

        # Combine the above features to calculate (estimate) the 
        # total probability that the user on P1 wants to go to P2.
        # "p(P2|P1) ~= p(P2) *  p(P1|P2)"
        # p(D=P2 | S=P1) * p(S=P1) = p(S=P1 | D=P2) * p(D=P2)
        p_tot = D * p_rel * T * p_abs

        # Ok, to do it correctly:
        # random variable S = user is currently on page s
        # random variable D = user want to go to page d
        # p(S=s)
        # p(D=d)
        # p(D=d AND S=s) = p(D=d, S=s) = p(D=d | S=s) * p(S=s)
        #                              = p(S=s | D=d) * p(D=d)
        # 
        # With history:
        # random variable Si = user visited page si i pages ago.
        # p(S0=s0)
        # p(S1=s1, S0=s0)
        # p(Sn=sn, ..., S1=s1, S0=s0)
        # p(Sn=sn, ..., S1=s1, S0=s0, D=d)
        # 
        # p(S1=s1, S0=s0, D=d) = p(D=d | S1=s1, S0=s0) * p(S1=s1, S0=s0)
        # p(D=d | S1=s1, S0=s0) = p(S1=s1, S0=s0 | D=d) / c
        #  -- naive Bayes assumption: --
        #                       = p(S1=s1 | D=d) * p(S0=s0 | D=d)  / c

        # print(url_2)
        # print('p_tot = D     * p_rel * T     * p_abs')
        # print('{:.3f} = {:.3f} * {:.3f} * {:.3f} * {:.3f}'.format(
        #       p_tot,   D,      p_rel,  T,      p_abs))
        # print()

        guesses.append((url_2, p_tot))
        print(url_2)

    # Return the x highest scoring candidates.
    x = 3
    return sorted(guesses, reverse=True, key=lambda g: g[1])[:x]






def generate_paths(start_url, max_len, current_url=None):
    # The
    paths = []
    # Loop over all pages pages directly reachable from the current 
    # page.
    linked_urls = nodes[current_url]['direct_links']
    for next_url in linked_urls:










def find_paths(current_url, end_url, travel_history=()):
    """ Calculates all paths without loops from 'current_url' to 
    'end_url' through the graph defined in 'nodes'.
    None of the returned paths will contain a node in 'travel_history'.
    """
    # The list of paths from the current url to the end url. 
    paths = []
    # Extend our travel history with the page we're currently at.
    # We work with tuples because they are immutable. 
    # (We now make a copy in memory of the travel history.)
    travel_history += (current_url,)
    # Check whether we already calculated paths from the current url 
    # to the destination url.
    # (This is dynamic programming AKA memoization.)
    existing_paths = nodes[current_url]['paths'].get(end_url, None)
    if existing_paths is not None:
        # If we do, simply return those -- but only the ones where
        # none of the nodes have been visited yet in the current 
        # recursive function call stack.
        for path in existing_paths:
            if not any((node in path) for node in travel_history):
                paths.append(path)
    else:
        # Calculate and store new paths.
        # 
        # If we're at the end of the recursion ..
        if current_url == end_url:
            # .. make a list of only one path. 
            # (Note the comma for constructing a tuple).
            paths.append((current_url,))
        else:
            # Loop over all pages pages directly reachable from the 
            # current page.
            linked_urls = nodes[current_url]['direct_links']
            for next_url in linked_urls:
                # If we would go to a page that's already visited
                # in this recursive function call stack ..
                if next_url in travel_history:
                    # .. skip this next page: we don't want loops.
                    continue
                # If not:
                # Ask for a list of paths from this next page to the 
                # destination page, omitting pages we've already
                # visited.
                new_paths = find_paths(next_url, end_url, travel_history)
                # For each such new path from next_url to end_url ..
                for new_path in new_paths:
                    # .. add a new path that goes:
                    # "current_url + path(next_url->end_url)"
                    # to the list of paths from current_url to end_url.
                    paths.append((current_url,)+new_path)
        if len(travel_history) == 1:
            # Add the calculated list of paths to the model.
            nodes[current_url]['paths'][end_url] = paths
    # Return the list of paths.
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
        transition_p = hops_to_next / total_visits # Py3
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


def test():
    from time import time
    learn_from(open('u23_1.csv'))
    t0 = time()
    # for url in nodes:
    #     get_guesses(url)
    get_guesses('http://www.standaard.be/')
    print(time()-t0)

    # Timing "get_guesses('http://www.standaard.be/')":
    # 14.93 seconds without memoization
    #  0.05 seconds with memoization (298x faster)

if __name__ == '__main__':
    test()