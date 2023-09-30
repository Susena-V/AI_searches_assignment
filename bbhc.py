import networkx as nx
import matplotlib.pyplot as plt
from queue import PriorityQueue
import math
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

def bnb_heuristics_and_cost(graph, start, goal, heuristics):
    priority_queue = PriorityQueue()

    initial_heuristic = calc_heuristic(start, goal, heuristics)
    initial_path_node = PathNode(start, [start], 0, initial_heuristic)
    priority_queue.put(initial_path_node)

    best_known_costs = {}

    while not priority_queue.empty():
        current_path_node = priority_queue.get()

        current_node = current_path_node.node
        current_path = current_path_node.path
        current_cost = current_path_node.cost

        if current_node == goal:
            return current_path, current_cost

        for neighbour in graph[current_node]:
            neighbor_cost = graph[current_node][neighbour].get('cost', 0)

            total_cost = current_cost + neighbor_cost
            neighbor_heuristic = calc_heuristic(neighbour, goal, heuristics)
            total_heuristic = neighbor_heuristic

            if neighbour not in best_known_costs or total_cost < best_known_costs[neighbour]:
                best_known_costs[neighbour] = total_cost

                new_path = current_path + [neighbour]
                new_path_node = PathNode(neighbour, new_path, total_cost, neighbor_heuristic)
                priority_queue.put(new_path_node)

    return [], math.inf

bbhc_path, optimal_cost = bnb_heuristics_and_cost(G, start, goal, Heuristic)

if not bbhc_path:
    print("No path to the goal exists.")
else:
    print("Path Chosend by Branch and Bound with Hueristics and cost is :", " -> ".join(bbhc_path))
    print("Least Cost that is found :", optimal_cost)

edges_in_bbhc_path=[(bbhc_path[i],bbhc_path[i+1]) for i in range(len(bbhc_path)-1)]

pos=nx.spring_layout(G)

node_labels = {node: node for node in G.nodes()}
nx.draw(G,pos,node_size=1000,node_color=(1,0.5,0.5),font_size=10)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
nx.draw_networkx_edges(G, pos, edgelist = edges_in_bbhc_path, edge_color='orange', width=2)
plt.title("Branch and Bound with Heuristic & Cost Visualisation")
plt.show()