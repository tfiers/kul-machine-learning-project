from url_predictor import learn_from
import matplotlib.pyplot as plt
import networkx as nx



def visualise(csv_file):
	G = nx.Graph()
	nodes = learn_from(open(csv_file))

	domain_list = []

	for node in nodes:
		G.add_node(node)

		domain = nodes[node]['domain']
		if domain not in domain_list:
			domain_list.append(domain)
		G.node[node]['domain'] = domain

		links = nodes[node]['direct_links']
		for link in links:
			G.add_edge(node,link)

	color_map = get_color_map(domain_list)

	print(str(color_map))

	nx.draw(G, node_color=[color_map[G.node[node]['domain']] for node in G]) #label = [G.node[node]['domain'] for node in G] ) 

	#plt.legend()

	plt.show()


def get_color_map(list):
	color_table = ['#ff0000','#ff8000','#ffff00','#80ff00','#00ffbf','#00bfff','#0040ff','#8000ff','#bf00ff','#ff00ff']
	color_map = dict()
	for i in range(len(list)):
		pos = i % 10
		color_map[list[i]] = color_table[i]

	return color_map

