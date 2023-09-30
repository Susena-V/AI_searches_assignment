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

nx.draw(G,with_labels=True,node_size=1000,node_color='purple',font_size=10)
plt.title("Graph")
plt.show()

start=input('Enter Starting node: ')

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

bfs_path=bfs(G,start)

print( '->'.join(bfs_path))

edges_in_bfs_path=[(bfs_path[i],bfs_path[i+1]) for i in range(len(bfs_path)-1)]

pos=nx.spring_layout(G)

node_labels = {node: node for node in G.nodes()}
nx.draw(G,pos,node_size=1000,node_color='hotpink',font_size=10)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
nx.draw_networkx_edges(G, pos, edgelist=edges_in_bfs_path, edge_color='orange', width=2)
plt.title("BFS Visualisation")
plt.show()