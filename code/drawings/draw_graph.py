import matplotlib.pyplot as plt
import networkx as nx
import os
import pz

# import pkg_resources
# pkg_resources.get_distribution("networkx").version

# graph_name = "android_fcg_7ab" # This graph has 32635 nodes
# graph_name = "dd_class1_1" # This graph has 327 nodes and 899 edges
# graph_name = "enzymes_class1_201" # This graph has 29 nodes and 53 edges
graph_name = "mutag_class1_1" # This graph has 23 nodes and 27 edges
# graph_name = "nc1_class0_1" # This graph has 21 nodes and 21 edges
# graph_name = "nci109_class0_1"# This graph has 21 nodes and 21 edges

# G = nx.dodecahedral_graph()

G = pz.load(graph_name + ".pz")
print G.number_of_nodes()
print G.number_of_edges()

ax = plt.axes(frameon = True)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

# nx.draw(G)  # networkx draw()


# nc1_class0_1 and nci109_class0_1 (21 nodes and 21 edges) --------------------

# k controls the distance between the nodes and varies between 0 and 1
# iterations is the number of times simulated annealing is run
# default k =0.1 and iterations=50
#pos = nx.spring_layout(G,k=0.1,iterations=10000)
#
#nx.draw_networkx_nodes(G, pos, node_size = 50, node_color = 'r')
#nx.draw_networkx_edges(G, pos, alpha = 0.4)

# -----------------------------------------------------------------------------



# mutag_class1_1 (23 nodes and 27 edges) --------------------------------------
red = '#EE1C25'
orange = '#FF6600'
blue = '#3399FF'
dark_blue = '#3F3D99'

labels={}
for i in xrange(G.number_of_nodes()):
    labels[i] = str(i) + ':' + G.nodes(data=True)[i][1]['label']

pos = nx.spring_layout(G,k=0.2,iterations=10000)

nx.draw_networkx_nodes(G, pos, node_size = 125,
                       node_color = orange)
nx.draw_networkx_edges(G, pos, alpha = 0.4)

nx.draw_networkx_labels(G,pos,labels,font_size=9,font_color=dark_blue,
                        font_weight='bold')
                        
#nx.draw_networkx_labels(G,pos,labels,font_size=9,font_color=dark_blue)

plt.savefig(os.path.join("drawings", graph_name + "_NEW.svg"), dpi = 10000)

# -----------------------------------------------------------------------------



# enzymes_class1_201 (29 nodes and 53 edges) ----------------------------------


#pos = nx.spring_layout(G,k=0.065,iterations=10000)
#
#nx.draw_networkx_nodes(G, pos, node_size = 20, node_color = 'r')
#nx.draw_networkx_edges(G, pos, alpha = 0.4)

# -----------------------------------------------------------------------------



# dd_class1_1 (327 nodes and 899 edges) ---------------------------------------


#pos = nx.spring_layout(G,k=0.015,iterations=500)
#
#nx.draw_networkx_nodes(G, pos, node_size = 5, node_color = 'r')
#nx.draw_networkx_edges(G, pos, alpha = 0.4)

# -----------------------------------------------------------------------------


# plt.savefig(os.path.join("drawings", graph_name + ".svg"), dpi = 10000)