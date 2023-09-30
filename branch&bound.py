import networkx as nx
import matplotlib.pyplot as plt
from queue import PriorityQueue
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

priority_queue = PriorityQueue()
priority_queue.put((0, [start]))
explored_nodes = []

best_known_costs = {}

while not priority_queue.empty():
    current_cost, path = priority_queue.get()
    current_node = path[-1]
    explored_nodes.append(current_node)
    current_graph = copy.deepcopy(G)

    min_cost_path = path
    subgraph = current_graph.subgraph(min_cost_path)

    for neighbor in current_graph[current_node]:
        neighbor_cost = current_graph[current_node][neighbor]['cost']
        if neighbor not in path:
            total_cost = current_cost + neighbor_cost
            if neighbor not in best_known_costs or total_cost < best_known_costs[neighbor]:
                best_known_costs[neighbor] = total_cost
                new_path = path + [neighbor]
                priority_queue.put((total_cost, new_path))

if goal in explored_nodes:
    bb_path = path
    bb_cost = current_cost
    print("Final Chosen Path:", " -> ".join(bb_path))
    print("Cost of the Path:", bb_cost)
    if bb_path == oracle_path:
      print("Algorithm matched the oracle path:", " -> ".join(path))
    else:
        print("Algorithm did not match the oracle path.")
else:
    print("No path to the destination exists.")

edges_in_bb_path=[(bb_path[i],bb_path[i+1]) for i in range(len(bb_path)-1)]

pos=nx.spring_layout(G)

node_labels = {node: node for node in G.nodes()}
nx.draw(G,pos,node_size=1000,node_color='turquoise',font_size=10)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
nx.draw_networkx_edges(G, pos, edgelist = edges_in_bb_path, edge_color='orange', width=2)
plt.title("Branch and Bound Visualisation")
plt.show()
