import random
import networkx as nx
from networkx.algorithms.dag import topological_sort
from networkx.algorithms.shortest_paths.generic import has_path
from networkx.algorithms.swap import connected_double_edge_swap
from layer import *

class NotPossibleError(Exception):
    pass

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
        self.score = -0.01
        self.G = nx.DiGraph()
        before = inputLayer(0)
        for i in range(1, n_hidden+1):
            new_node = caddLayer()
            self.G.add_edge(before, new_node)
            self.G.add_edge(inputLayer(i), new_node)
            before = new_node
        for i in range(n_hidden+1):
            self.G.add_edge(before, outputLayer(i))

    def autoRollback(fn):
        '''
        check if the net work is good defined
        if not rollback and return false
        '''
        def wrapper(self, *args, **kwargs):
            G = self.G.copy()
            fn(self, *args, **kwargs)
            ret = self.validate()
            if not ret:
                self.G = G
            return ret
        return wrapper

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

    def getLua(self):
        result = ''
        nodes = topological_sort(self.G)
        node_ix = {node: ix for ix, node in enumerate(nodes)}
        # write to file
        for ix, node in enumerate(nodes):
            inputs = [node_ix[i] for i in self.G.predecessors(node)]
            result = result + "\n" + node.genLua(ix, inputs)
        return self.prefix + result + self.postfix

    def randomNode(self, source_node, withInput=False, withOutput=False):
        '''
        get a random node which can be a predecessor of node
        '''
        node = random.choice(self.G.nodes())
        for _ in range(10):
            node = random.choice(self.G.nodes())
            if node != source_node \
                    and (withInput or not isinstance(node, inputLayer)) \
                    and (withOutput or not isinstance(node, outputLayer)) \
                    and (not source_node or not has_path(self.G, source_node, node)):
                return node
        raise NotPossibleError()


    def fitNode(self, node):
        '''
        remove too many inputs or add too less inputs
        '''
        while node.n_input > len(self.G.predecessors(node)):
            print('fitNode add')
            cand = self.randomNode(node, withInput=True)
            self.G.add_edge(cand, node)
        while node.n_input < len(self.G.predecessors(node)):
            print('fitNode remove')
            cand = random.choice(self.G.predecessors(node))
            self.G.remove_edge(cand, node)

    def addNodeOnEdge(self, node, edge):
        '''
        add a node between two node of edge
        if node has more than one inputs and random inputs is added
        '''
        print('addNodeOnEdge')
        print(node)
        print(edge)
        f = edge[0]
        t = edge[1]
        self.G.add_edge(f, node)
        self.G.add_edge(node, t)
        self.G.remove_edge(f, t)
        self.fitNode(node)

    @autoRollback
    def removeNode(self, node):
        '''
        remove a node from G
        connnect all edge to it to its sucessor
        '''
        print('remove')
        print(node)
        succ = self.G.successors(node)
        pre = self.G.predecessors(node)
        self.G.remove_node(node)
        for i in succ:
            self.G.add_edge(random.choice(pre), i)
            self.fitNode(i)

    @autoRollback
    def changeNodeConnect(self, node):
        '''
        change connection of a node
        '''
        print('change')
        print(node)
        # connected_double_edge_swap(self.G)
        pre = self.G.predecessors(node)
        self.G.remove_edge(random.choice(pre), node)
        self.fitNode(node)

    @autoRollback
    def replaceNode(self, node, nnode):
        '''
        replace a node with a new one
        '''
        print('replace')
        print(node)
        print(nnode)
        succ = self.G.successors(node)
        pre = self.G.predecessors(node)
        self.G.remove_node(node)
        for i in pre:
            self.G.add_edge(i, nnode)
        for i in succ:
            self.G.add_edge(nnode, i)
        self.fitNode(nnode)

    def validate(self):
        while True:
            for node in self.G.nodes():
                 if self.G.out_degree(node) == 0 and not isinstance(node, outputLayer):
                     if isinstance(node, inputLayer):
                         return False
                     #self.G.add_edge(self.G.predecessors(node)[0],
                     #        self.G.successors(node)[0])
                     self.G.remove_node(node)
                     break
            else:
                return True

    def simplify(self):
        '''
        valdate and simplify network
        return False if it is a broken net
        '''
        while True:
            for node in self.G.nodes():
                if isinstance(node, reluLayer) \
                        and isinstance(self.G.predecessors(node)[0], reluLayer) \
                        and self.G.out_degree(self.G.predecessors(node)[0]) == 1 \
                        or isinstance(node, linearLayer) \
                        and isinstance(self.G.predecessors(node)[0], linearLayer) \
                        and self.G.out_degree(self.G.predecessors(node)[0]) == 1 :
                    print('simplify remove')
                    print(node)
                    self.G.add_edge(self.G.predecessors(node)[0], self.G.successors(node)[0])
                    self.G.remove_node(node)
                    break
            else:
                return True

