import networkx as nx
import matplotlib.pyplot as plt

def create_Graph():
  n=int(input("Enter no of nodes: "))
  nodes=[]

  G = nx.DiGraph()

  for i in range(n):
    node=input(f'Enter node {i+1}: ')
    nodes.append(node)

  for i in range(n):
    neighbours=int(input(f"Enter number of neighbours of {nodes[i]}: "))
    for j in range(neighbours):
      neighbour=input(f"Enter neighbour {j+1}: ")
      G.add_edge(nodes[i],neighbour)

  return G

G=create_Graph()

nx.draw(G,with_labels=True,node_size=1000,node_color='purple',font_size=10,font_color='white')
plt.title('Graph')
plt.show()

start=input('Enter Starting node: ')
goal=input('Enter Goal node: ')

def dfs_path(graph, start, goal, path=[], visited_nodes=[]):
    path+=[start]
    visited_nodes.append(start)
    if start == goal:
        return path, visited_nodes
    if start not in graph:
        return None, visited_nodes
    shortest_path = None
    for neighbor in graph[start]:
        neighbor_name = neighbor
        if neighbor_name not in path:
            new_path, visited_nodes = dfs_path(graph, neighbor_name, goal, path, visited_nodes)
            if new_path:
                if shortest_path is None or (new_path < shortest_path):
                    shortest_path = new_path
    return shortest_path, visited_nodes


shortest_path, visited_nodes = dfs_path(G, start, goal)

if shortest_path:
    print("DFS path from {} to {}:".format(start, goal))
    print(" -> ".join(shortest_path))
else:
    print("No path found.")

if shortest_path:
    edges_in_dfs_path = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]

pos=nx.spring_layout(G)

node_labels = {node: node for node in G.nodes()}
nx.draw(G,pos,node_size=1000,node_color='violet')
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
nx.draw_networkx_edges(G, pos, edgelist=edges_in_dfs_path, edge_color='orange', width=2)
plt.title("DFS Visualisation")
plt.show()