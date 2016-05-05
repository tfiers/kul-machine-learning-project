# Activate anaconda environment 'kul-ml'.

import json
from dateutil.parser import parse as parse_datetime

# Keys are urls.
# Values are lists [[url, score], [url, score], ..]
# of all urls that can be reached from the current page.
model = {}


def get_guesses(url):
    """ Given a url, returns a list of the most probable destination
    urls. Each entry is of the form [url, probability_score].
    The entries are sorted from highest to lowest probabibility.

    Example:
    [['https://dtai.cs.kuleuven.be/events/leuveninc', 0.9],
     ['https://google.com', 0.5]]
    """
    return model.get(url, [])


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
            dt = next_event['t'] - curent_event['t']
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
                't': curent_event['t'], # = time of event
                'dt' : dt, # dt = delta-t = duration of stay
                'exit_type': exit_type,
            })
    return page_visits


def make_graph(page_visits):
    """ Constructs a directed graph from the page visits list.
    """
    # The graph is a dictionary of nodes. The keys are page urls,
    # the values contain urls of pages following the page of the key 
    # url, and the number of times the key url page was visited.
    nodes = {}
    for i in range(len(page_visits)):
        # Get the url of the currently visited page.
        url = page_visits[i]['url']
        # If the visited page is already a node in the graph..
        if url in nodes:
            # .. increase the number of times it was visited.
            nodes[url]['num_visits'] += 1
        else:
            # If not, add a new node, with currently one visit.
            nodes[url] = {
                'num_visits': 1,
                'linked_urls': {},
            }
        # Get a reference to the urls linked from the current node.
        linked_urls = nodes[url]['linked_urls']
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


def preprocess(csv_file_handle):
    """ 
    """
    # Read the csv file as a list of strings.
    lines = csv_file.readlines()
    # Make a list of 'event' dictionaries.
    events = parse(lines)
    # Make a list of 'page visit' dictionaries.
    page_visits = make_page_visits(events)
    # Make a graph (as a dictionary of nodes), discarding absolute
    # temporal information and higher-order sequential information.
    nodes = make_graph(page_visits)


def learn(nodes):
    """ Trains a model that predicts the most likely destination pages
    for each starting url.
    """
    pass
    # For each starting url:
    #   - Traverse graph recursively.
    #   - 


def find_paths(nodes, current_url, end_url, travel_history=()):
    """ Recursively calculates all paths without loops through the 
    graph 'nodes' from 'current_url' to 'end_url'.
    'travel_history' is the sequence of urls already "visited" in the
    current traversal.
    """
    # Extend our travel history with the page we're currently at.
    # We work with tuples because they are immutable.
    travel_history += (current_url,)
    # If we're at the end of the recursion..
    if current_url == end_url:
        # .. return a list of only one path.
        return [travel_history]

    # The list of paths from the current node (url / page) to the
    # destination url that we'll return.
    paths = []
    # A dictionary of pages directly reachable from the current page.
    linked_urls = nodes[current_url]['linked_urls']
    # print(linked_urls)
    # For each such page..
    for next_url in linked_urls:
        # .. if it is not already in the travel history 
        # (to avoid loops)..
        if next_url not in travel_history:
            # .. ask for a list of paths from it to the destination 
            # page.
            new_paths = find_paths(nodes, next_url, end_url, travel_history)
            # Add each such new (complete) path to the list of paths
            # we'll return.
            paths += new_paths
    # Return the list of paths.
    return paths

