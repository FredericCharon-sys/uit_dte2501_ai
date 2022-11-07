import json
import random as rnd
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')

'''
opening graph.json and creating a pheromone matrix. This matrix gets entries containing a '0' corresponding to the
    edges in the graph
'''
graph = json.load(open('graph.json'))
pheromoneMatrix = []
for i in range(len(graph)):
    pheromoneMatrix.append([])
    for j in graph[str(i)]:
        pheromoneMatrix[i].append(0)

'''
The function to determine the ant's way from node 0 to node 29 
'''
def ant_travel():
    nodes_visited = ['0']   # all the visited nodes are saved here
    current_node = '0'      # this is the node the ant is currently on
    while current_node != '29':     # the loop ends once node 29 is visited
        next_node = choose_path(current_node)   # choosing the next node
        current_node = next_node            # updating the current node
        nodes_visited.append(next_node)     # we append the new node to our visited nodes list
    return nodes_visited

'''
This function gets a current node, looks at it's neighboring nodes 
    and chooses one of them randomly. Once a node is chosen, we look up the path's weight and add pheromones 
    based on the weight to the pheromone matrix
'''
def choose_path(node):
    random_index = rnd.randint(0, len(graph[node]) - 1)     # choosing one of the indices at random
    next_node = list(graph[node].keys())[random_index]
    weight = graph[node][next_node]['weight']
    pheromoneMatrix[int(node)][random_index] = round(pheromoneMatrix[int(node)][random_index] + 0.01 / weight, 5)
    return next_node

'''
The last function is visualizing the graph. The yellow nodes are showing the shortest path.
Grey edges are showing a low amount of pheromones, orange edges have a higher amount and red edges have a really 
    high amount of pheromones
'''
def visualize_graph():
    G = nx.Graph()
    edges = []

    # making a list of all the edges and add them with the corresponding weights to the graph
    for i in graph:
        for j in graph[str(i)]:
            edges.append([i, j])
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=graph[edge[0]][edge[1]]['weight'])

    # getting the nodes of the shortest path using dijkstra. The colormap is used to colors these nodes yellow
    dijkstra_path = nx.dijkstra_path(G, "0", "29")
    node_color_map = []
    for node in G:
        if node in dijkstra_path:
            node_color_map.append('yellow')
        else:
            node_color_map.append('green')

    # here we add the edges and color them depending on the amount of pheromones
    edge_color_map = []
    for i in range(len(pheromoneMatrix)):
        for j in pheromoneMatrix[i]:
            if j > 500:
                edge_color_map.append("red")
            elif j > 100:
                edge_color_map.append('orange')
            else:
                edge_color_map.append("grey")

    # the graph is drawn and plotted
    nx.draw_networkx(G, node_color=node_color_map, edge_color=edge_color_map)
    matplotlib.pyplot.show()


# letting 1000 ants travel from node 0 to node 29 and print their paths
for i in range(1000):
    print(ant_travel())
# printing the pheromone matrix
for i in range(len(pheromoneMatrix)):
    print(i, pheromoneMatrix[i])
# visualizing graph
#visualize_graph()
