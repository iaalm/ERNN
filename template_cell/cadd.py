from network import network
from layer import *
import pickle

a = network(2)
node0 = [i for i in a.G.nodes() if isinstance(i, inputLayer) and i.layer_id == 0][0]
node1 = [i for i in a.G.nodes() if isinstance(i, inputLayer) and i.layer_id == 1][0]
out0 = [i for i in a.G.nodes() if isinstance(i, outputLayer) and i.layer_id == 0][0]
out1 = [i for i in a.G.nodes() if isinstance(i, outputLayer) and i.layer_id == 1][0]
cadd = caddLayer()
a.G.remove_edge(node0, out0)
a.G.remove_edge(node1, out1)
a.G.add_edge(node0, cadd)
a.G.add_edge(node1, cadd)
a.G.add_edge(cadd, out0)
a.G.add_edge(cadd, out1)

with open('template_cell/cadd/cell.pickle', 'wb') as fd:
    pickle.dump(a, fd)
a.writeLua('template_cell/cadd/cell.lua')
