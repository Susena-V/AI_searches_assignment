import networkx as nx
import matplotlib.pyplot as plt

def create_Graph():
  n=int(input("Enter no of nodes: "))
  nodes=[]
  Heuristic={}

  G = nx.DiGraph()

  for i in range(n):
    node=input(f'Enter node {i+1}: ')
    nodes.append(node)

  start=input('Enter Starting node: ')
  goal=input('Enter Goal node: ')

  for i in range(n):
    neighbours=int(input(f"Enter number of neighbours of {nodes[i]}: "))
    for j in range(neighbours):
      neighbour=input(f"Enter neighbour {j+1}: ")
      G.add_edge(nodes[i],neighbour)

  for node in nodes:
    if node==goal:
      Heuristic.update({goal:0})
    else:
      h=int(input(f"Enter Heuristic from {node} to {goal}: "))
      Heuristic.update({node:h})

  return G,start,goal,Heuristic

G,start,goal,Heuristic=create_Graph()

nx.draw(G,with_labels=True,node_size=1000,node_color='purple',font_size=10)
plt.title('Graph')
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
print( '->'.join(HC_path))

edges_in_hc_path=[(HC_path[i],HC_path[i+1]) for i in range(len(HC_path)-1)]

pos=nx.spring_layout(G)

node_labels = {node: node for node in G.nodes()}
nx.draw(G,pos,node_size=1000,node_color='cyan',font_size=10)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
nx.draw_networkx_edges(G, pos, edgelist=edges_in_hc_path, edge_color='orange', width=2)
plt.title("Hill Climbing Visualisation")
plt.show()