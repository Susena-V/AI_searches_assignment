import networkx as nx
import matplotlib.pyplot as plt
import copy

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

#Finding oracle
paths=[]

for path in nx.all_simple_paths(G,start,goal):
  paths.append(path)

paths_cost=[]

for path in paths:
    cost=sum(G[path[i]][path[i+1]]['cost'] for i in range(len(path)-1))
    paths_cost.append((path,cost))

paths_cost.sort(key=lambda x:x[1])

oracle=paths_cost[0][1]
oracle_path=paths_cost[0][0]
print('-->'.join(oracle_path))

current_node = start
explored_nodes = [current_node]
total_cost = 0
bbg_path = [current_node]

while current_node != goal:
    neighbors = G[current_node]
    min_cost = float('inf')
    min_cost_neighbor = None

    for neighbor, data in neighbors.items():
        neighbor_cost = data['cost']
        if neighbor not in explored_nodes:
            if neighbor_cost < min_cost:
                min_cost = neighbor_cost
                min_cost_neighbor = neighbor

    if min_cost_neighbor is None:
        print("No path to the goal exists.")
        break

    current_node = min_cost_neighbor
    explored_nodes.append(current_node)
    total_cost += min_cost
    bbg_path.append(current_node)

if current_node == goal:
    print("Final Chosen Path:", " -> ".join(bbg_path))
    print("Cost of the Path:", total_cost)

if bbg_path == oracle_path:
    print("Algorithm Matched the oracle path:", bbg_path)
else:
    print("Algorithm did not find the oracle path.")

edges_in_bbg_path=[(bbg_path[i],bbg_path[i+1]) for i in range(len(bbg_path)-1)]

pos=nx.spring_layout(G)

node_labels = {node: node for node in G.nodes()}
nx.draw(G,pos,node_size=1000,node_color=(0.9,0.7,1),font_size=10)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
nx.draw_networkx_edges(G, pos, edgelist = edges_in_bbg_path, edge_color='orange', width=2)
plt.title("Branch and Bound Greedy Visualisation")
plt.show()