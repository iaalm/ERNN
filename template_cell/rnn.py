import network
from layer import *
import pickle

a = network.network(2)
node0 = [i for i in a.G.nodes() if isinstance(i, inputLayer) and i.layer_id == 0][0]
node1 = [i for i in a.G.nodes() if isinstance(i, inputLayer) and i.layer_id == 1][0]
out0 = [i for i in a.G.nodes() if isinstance(i, outputLayer) and i.layer_id == 0][0]
out1 = [i for i in a.G.nodes() if isinstance(i, outputLayer) and i.layer_id == 1][0]
l0 = linearLayer()
l1 = linearLayer()
cadd = caddLayer()
act = reluLayer()
a.G.remove_edge(node0, out0)
a.G.remove_edge(node1, out1)
a.G.add_edge(node0, l0)
a.G.add_edge(node1, l1)
a.G.add_edge(l0, cadd)
a.G.add_edge(l1, cadd)
a.G.add_edge(cadd, act)
a.G.add_edge(act, out0)
a.G.add_edge(act, out1)

with open('template_cell/rnn/cell.pickle', 'wb') as fd:
    pickle.dump(a, fd)
a.writeLua('template_cell/rnn/cell.lua')
