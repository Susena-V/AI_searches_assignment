import heapq
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

def a_star(graph, start, goal, heuristic):
    open_list = []
    closed_set = set()

    heapq.heappush(open_list, (0, [start]))

    while open_list:
        current_cost, path = heapq.heappop(open_list)
        current_node = path[-1]

        if current_node == goal:
            return path, current_cost

        if current_node in closed_set:
            continue

        closed_set.add(current_node)

        for neighbor in graph.neighbors(current_node):
            if neighbor not in closed_set:
                cost = graph[current_node][neighbor].get('cost', 1)
                total_cost = current_cost + cost
                neighbor_heuristic = calc_heuristic(neighbor,goal,Heuristic)

                priority = total_cost + neighbor_heuristic

                new_path = path + [neighbor]

                heapq.heappush(open_list, (total_cost, new_path))

    return None, None

astar_path, optimal_cost = a_star(G, start, goal, Heuristic)

if astar_path:
    print("Optimal Path:", " --> ".join(astar_path))
    print("Optimal Cost:", optimal_cost)
else:
    print("No path to the goal exists.")

if astar_path == oracle_path:
    print("Algorithm matched the oracle path: ", ' --> '.join(oracle_path))
else:
    print("Algorithm did not match the oracle path.")

edges_in_AStar_path=[(astar_path[i],astar_path[i+1]) for i in range(len(astar_path)-1)]

pos=nx.spring_layout(G)

node_labels = {node: node for node in G.nodes()}
nx.draw(G,pos,node_size=1000,node_color=(0.8,0.4,1),font_size=10)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
nx.draw_networkx_edges(G, pos, edgelist = edges_in_AStar_path, edge_color='orange', width=2)
plt.title("A* Visualisation")
plt.show()