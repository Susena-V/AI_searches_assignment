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

class PathNode:
    def __init__(self, node, path, cost, heuristic):
        self.node = node
        self.path = path
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

def bnb_greedy_heur(graph, start, goal, heuristics):
    priority_queue = PriorityQueue()

    initial_path_node = (0, [start], heuristics[start])
    priority_queue.put(initial_path_node)

    while not priority_queue.empty():
        current_cost, path, current_heuristic = priority_queue.get()
        current_node = path[-1]

        if current_node == goal:
            return path, current_cost

        for neighbor in graph[current_node]:
            neighbor_cost = graph[current_node][neighbor].get('cost', 0)

            total_cost = current_cost + neighbor_cost

            neighbor_heuristic = heuristics.get(neighbor, 0)

            total_heuristic = neighbor_heuristic

            new_path = path + [neighbor]

            priority_queue.put((total_cost, new_path, total_heuristic))

    return [], float('inf')

bbgh_path, bbgh_cost = bnb_greedy_heur(G, start, goal, Heuristic)

if not bbgh_path:
    print("No path to the goal exists.")
else:
    print("Path Chosen by B&B with Hueristics and greedy:", " -> ".join(bbgh_path))
    print("Optimal Cost:", bbgh_cost)

if bbgh_path == oracle_path:
    print("Algorithm matched the oracle path:", "->".join(bbgh_path))
else:
    print("Algorithm did not match the oracle path.")

edges_in_bbgh_path=[(bbgh_path[i],bbgh_path[i+1]) for i in range(len(bbgh_path)-1)]

pos=nx.spring_layout(G)

node_labels = {node: node for node in G.nodes()}
nx.draw(G,pos,node_size=1000,node_color=(1,1,0.65),font_size=10)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
nx.draw_networkx_edges(G, pos, edgelist = edges_in_bbgh_path, edge_color='orange', width=2)
plt.title("Branch and Bound Greedy + Extended List Visualisation")
plt.show()