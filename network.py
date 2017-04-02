import networkx as nx
from networkx.algorithms.dag import topological_sort
from layer import *


class network:
    prefix = '''-- automatic generated lua cell
require 'nn'
require 'nngraph'

local cell = {}
function cell.cell(input_size, output_size, rnn_size)
  local inputs = {}
  local outputs = {}
'''

    postfix = '''
  return nn.gModule(inputs, outputs)
end

return cell
'''

    def __init__(self, n_hidden):
        self.parent_path = []
        self.n_hidden = n_hidden
        self.score = -1
        self.G = nx.DiGraph()
        self.G.add_edge(inputLayer(0), outputLayer(n_hidden))
        for i in range(n_hidden):
            self.G.add_edge(inputLayer(i + 1), outputLayer(i))

    def writeLua(self, filename):
        nodes = topological_sort(self.G)
        node_ix = {node: ix for ix, node in enumerate(nodes)}
        # write to file
        with open(filename, 'w') as fd:
            fd.write(self.prefix)
            for ix, node in enumerate(nodes):
                inputs = [node_ix[i] for i in self.G.predecessors(node)]
                fd.write(node.genLua(ix, inputs))
            fd.write(self.postfix)
