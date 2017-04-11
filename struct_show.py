#!/usr/bin/python3

import os
import pickle
import argparse
import networkx as nx
from layer import inputLayer, outputLayer
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("dir", metavar='dir', help="model dirs")
args = parser.parse_args()

with open(os.path.join(args.dir, 'cell.lua'), 'r') as fd:
    score = fd.readline().strip().split()[1]
with open(os.path.join(args.dir, 'cell.pickle'), 'rb') as fd:
    net = pickle.load(fd)

fixed_positions = {}
for node in net.G.nodes():
    if isinstance(node, inputLayer):
        fixed_positions[node] = (0, node.layer_id)
    if isinstance(node, outputLayer):
        fixed_positions[node] = (1, node.layer_id)

fixed_nodes = fixed_positions.keys()
pos = nx.spring_layout(net.G, pos=fixed_positions, fixed=fixed_nodes)
nx.draw(net.G, pos)
plt.text(0, 0, score)

plt.show()
