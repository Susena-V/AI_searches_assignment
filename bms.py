import networkx as nx
import matplotlib.pyplot as plt

def create_Graph():
  n=int(input("Enter no of nodes: "))
  nodes=[]

  G = nx.Graph()

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

start=input('Enter Starting node: ')
end=input('Enter Ending node: ')

for path in nx.all_simple_paths(G,start,end):
  print(path)
  
nx.draw(G,with_labels=True,node_size=1000,node_color='purple',font_size=10,font_color='white')
plt.show()