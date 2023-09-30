import networkx as nx
import matplotlib.pyplot as plt
from queue import PriorityQueue
import math
import copy
import heapq

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
plt.title('Graph')
plt.show()

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
oracle_path=paths_cost[0][0]
print(' --> '.join(oracle_path)," ",oracle)

for path in paths:
    p = " --> ".join(path)
    path_graph = G.edge_subgraph([(path[i], path[i+1]) for i in range(len(path)-1)])

    pos = nx.shell_layout(path_graph)
    nx.draw_networkx_nodes(path_graph, pos)
    nx.draw_networkx_edges(path_graph, pos)
    nx.draw_networkx_labels(path_graph, pos)
    plt.title(f"Path: {' --> '.join(path)}")
    plt.show()

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
    print(" --> ".join(shortest_path))
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

def bfs(G,start):
  queue=[start]
  bfs_path=[]
  visited=[start]

  while queue:
    node=queue.pop(0)
    bfs_path.append(node)
    for neighbour in G.neighbors(node):
      if neighbour not in visited:
          queue.append(neighbour)
          visited.append(neighbour)

  return bfs_path

print(f"BFS path from {start} to {goal}")
bfs_path=bfs(G,start)
print( ' --> '.join(bfs_path))

edges_in_bfs_path=[(bfs_path[i],bfs_path[i+1]) for i in range(len(bfs_path)-1)]

pos=nx.spring_layout(G)

node_labels = {node: node for node in G.nodes()}
nx.draw(G,pos,node_size=1000,node_color='pink',font_size=10)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
nx.draw_networkx_edges(G, pos, edgelist=edges_in_bfs_path, edge_color='orange', width=2)
plt.title("BFS Visualisation")
plt.show()

def HillClimbing(graph, start, goal, heuristics):
    curr = start
    path = [curr]

    while curr != goal:
        neighbors = list(graph.neighbors(curr))
        min_cost = float('inf')
        next_node = None

        for neighbor in neighbors:
            neighbor_heuristic = heuristics.get(neighbor, 0)
            if neighbor not in path and (neighbor_heuristic) < min_cost:
                min_cost = neighbor_heuristic
                next_node = neighbor

        if next_node is None:
            break

        path.append(next_node)

        curr = next_node

    return path

HC_path=HillClimbing(G,start,goal,Heuristic)
print( ' --> '.join(HC_path))

edges_in_hc_path=[(HC_path[i],HC_path[i+1]) for i in range(len(HC_path)-1)]

pos=nx.spring_layout(G)

node_labels = {node: node for node in G.nodes()}
nx.draw(G,pos,node_size=1000,node_color='cyan',font_size=10)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
nx.draw_networkx_edges(G, pos, edgelist=edges_in_hc_path, edge_color='orange', width=2)
plt.title("Hill Climbing Visualisation")
plt.show()

def beam(graph, start, goal, beam_width,heuristic):
    initial_path = [start]
    beam = [(initial_path, heuristic[start])]
    beam_paths=[]

    while beam:
        candidates = []
        for path, cost in beam:
            current_node = path[-1]
            for neighbour in graph.neighbors(current_node):
                if neighbour not in path:
                    new_path = path + [neighbour]
                    new_cost = heuristic[neighbour]
                    candidates.append((new_path, new_cost))

        candidates.sort(key=lambda x: x[1])
        beam = candidates[:beam_width]

        for path, cost in beam:
            if path[-1] == goal:
              return path

    return None

width=int(input("Enter beam_width: "))

beam_path=beam(G,start,goal,width,Heuristic)
print('Beam path: ','-->'.join(beam_path))

edges_in_beam_path=[(beam_path[i],beam_path[i+1]) for i in range(len(beam_path)-1)]

pos=nx.spring_layout(G)

node_labels = {node: node for node in G.nodes()}
nx.draw(G,pos,node_size=1000,node_color=(1,0.2,0.7),font_size=10)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
nx.draw_networkx_edges(G, pos, edgelist=edges_in_beam_path, edge_color='orange', width=2)
plt.title("Beam Visualisation")
plt.show()

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
    print("Branch and Bound Path:", " --> ".join(bb_path))
    print("Cost of  Path:", bb_cost)
    if bb_path == oracle_path:
      print("Algorithm matched the oracle path:", " --> ".join(path))
    else:
        print("Algorithm did not match the oracle path.")
else:
    print("No path to the goal exists.")

edges_in_bb_path=[(bb_path[i],bb_path[i+1]) for i in range(len(bb_path)-1)]

pos=nx.spring_layout(G)

node_labels = {node: node for node in G.nodes()}
nx.draw(G,pos,node_size=1000,node_color='turquoise',font_size=10)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
nx.draw_networkx_edges(G, pos, edgelist = edges_in_bb_path, edge_color='orange', width=2)
plt.title("Branch and Bound Visualisation")
plt.show()

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
    print("Branch and Bound Greedy Path:", " --> ".join(bbg_path))
    print("Cost of Path:", total_cost)

if bbg_path == oracle_path:
    print("Algorithm matched the oracle path: ", bbg_path)
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
    print("Greedy Branch and Bound  with extended list path: ", " --> ".join(path))
    print("Cost of Path:", current_cost)

edges_in_bbgel_path=[(path[i],path[i+1]) for i in range(len(path)-1)]

pos=nx.spring_layout(G)

node_labels = {node: node for node in G.nodes()}
nx.draw(G,pos,node_size=1000,node_color=(0.6,0.5,1),font_size=10)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
nx.draw_networkx_edges(G, pos, edgelist = edges_in_bbgel_path, edge_color='orange', width=2)
plt.title("Branch and Bound Greedy + Extended List Visualisation")
plt.show()

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
    print("Branch and Bound with Heuristics shortest path: ", " --> ".join(bbhc_path))
    print("Cost of path:", optimal_cost)

edges_in_bbhc_path=[(bbhc_path[i],bbhc_path[i+1]) for i in range(len(bbhc_path)-1)]

pos=nx.spring_layout(G)

node_labels = {node: node for node in G.nodes()}
nx.draw(G,pos,node_size=1000,node_color=(1,0.5,0.5),font_size=10)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
nx.draw_networkx_edges(G, pos, edgelist = edges_in_bbhc_path, edge_color='orange', width=2)
plt.title("Branch and Bound with Heuristic & Cost Visualisation")
plt.show()

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
    print("B&B with Hueristics and greedy path : ", " --> ".join(bbgh_path))
    print("Cost of path:", bbgh_cost)

if bbgh_path == oracle_path:
    print("Algorithm matched the oracle path: ", " --> ".join(bbgh_path))
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
    print("A* Path:", " --> ".join(astar_path))
    print("Cost of path:", optimal_cost)
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