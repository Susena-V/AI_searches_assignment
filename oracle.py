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
      cost=int(input("Cost: "))
      G.add_edge(nodes[i],neighbour,cost=cost)

  return G

G=create_Graph()

nx.draw(G,with_labels=True,node_size=1000,node_color='purple',font_size=10,font_color='white')
plt.show()

start=input('Enter Starting node: ')
goal=input('Enter Goal node: ')

paths=[]

for path in nx.all_simple_paths(G,start,goal):
  paths.append(path)
  print(path)

paths_cost=[]

for path in paths:
    cost=sum(G[path[i]][path[i+1]]['cost'] for i in range(len(path)-1))
    paths_cost.append((path,cost))

paths_cost.sort(key=lambda x:x[1])

oracle=paths_cost[0][1]
sp=" --> ".join(paths_cost[0][0])
print(sp," ",oracle)
oracle_path=paths_cost[0][0]

for path, cost in paths_cost:
    p = " -> ".join(path)
    print(f"The Path Followed is {p} and the cost of it is {cost}")
    path_graph = G.edge_subgraph([(path[i], path[i+1]) for i in range(len(path)-1)])

    pos = nx.shell_layout(path_graph)
    nx.draw_networkx_nodes(path_graph, pos)
    nx.draw_networkx_edges(path_graph, pos)
    nx.draw_networkx_labels(path_graph, pos)
    plt.title(f"Path: {' -> '.join(path)}\nCost: {cost}")
    plt.show()
