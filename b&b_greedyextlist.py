import networkx as nx
import matplotlib.pyplot as plt
from queue import PriorityQueue
import copy

def create_Graph():
  n=int(input("Enter no of nodes: "))
  nodes=[]
  Heuristic={}

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

  start=input('Enter Starting node: ')
  goal=input('Enter Goal node: ')

  for node in nodes:
    if node==goal:
      Heuristic.update({goal:0})
    else:
      h=int(input(f"Enter Heuristic from {node} to {goal}: "))
      Heuristic.update({node:h})

  return G,start,goal,Heuristic

G,start,goal,Heuristic=create_Graph()

nx.draw(G,with_labels=True,node_size=1000,node_color='purple',font_size=10,font_color='white')
plt.show()

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

def calc_heuristic(node,goal,heuristic):
    return heuristic.get(node, 0)

priority_queue = PriorityQueue()

priority_queue.put((0, [start], calc_heuristic(start, goal, Heuristic)))

explored_nodes = []

best_known_costs = {}
var = 0

while not priority_queue.empty():
    var = var + 1
    current_cost, path, heuristic = priority_queue.get()
    current_node = path[-1]

    explored_nodes.append(current_node)
    current_graph = copy.deepcopy(G)

    min_cost_path = path

    if current_node == goal:
        break

    for neighbour in G.neighbors(current_node):
        neighbour_cost = G[current_node][neighbour]['cost']
        if neighbour not in path:
            total_cost = current_cost + neighbour_cost
            neighbor_heuristic = calc_heuristic(neighbour, goal, Heuristic)
            total_heuristic = neighbor_heuristic
            if neighbour not in best_known_costs or total_cost < best_known_costs[neighbour]:
                best_known_costs[neighbour] = total_cost
                new_path = path + [neighbour]
                priority_queue.put((total_cost, new_path, total_heuristic))

if current_node == goal:
    print("Final Chosen Path by Greedy Branch and Bound  with extended list :", " -> ".join(path))
    print("Cost of the Path:", current_cost)

edges_in_bbgel_path=[(path[i],path[i+1]) for i in range(len(path)-1)]

pos=nx.spring_layout(G)

node_labels = {node: node for node in G.nodes()}
nx.draw(G,pos,node_size=1000,node_color=(0.6,0.5,1),font_size=10)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
nx.draw_networkx_edges(G, pos, edgelist = edges_in_bbgel_path, edge_color='orange', width=2)
plt.title("Branch and Bound Greedy + Extended List Visualisation")
plt.show()