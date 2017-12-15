class inputLayer:
    n_input = 0
    template = '''  -- inputLayer
  node%d = nn.Identity()():annotate{name='input_%d'}
  inputs[%d] = node%d
'''

    def __init__(self, layer_id):  # layer_id is special for io layers
        self.layer_id = layer_id

    def __str__(self):
        return 'inputLayer-%d' % self.layer_id

    def genLua(self, node_id, inputs):
        assert len(inputs) == 0, 'input layer %d / 0' % len(inputs)
        return self.template % (node_id, self.layer_id + 1, self.layer_id + 1, node_id)


class outputLayer:
    n_input = 1
    template = '''  -- outputLayer
  outputs[%d] = nn.Identity()(node%d):annotate{name='output_%d'}
'''
    template_out = '''  -- outputLayer(final)
  node%d = nn.Linear(rnn_size, output_size)(node%d):annotate{name='output_final'}
  outputs[%d] = nn.LogSoftMax()(node%d)
'''

    def __init__(self, layer_id):  # layer_id is special for io layers
        self.layer_id = layer_id

    def __str__(self):
        return 'outputLayer-%d' % self.layer_id

    def genLua(self, node_id, inputs):
        assert len(inputs) == 1, 'output layer %d / 1' % len(inputs)
        if self.layer_id == 0:
            return self.template_out % \
                    (node_id, inputs[0], self.layer_id + 1,  node_id)
        else:
            return self.template % (self.layer_id + 1,  inputs[0], self.layer_id + 1)


class linearLayer:
    n_input = 1
    template = '''  -- linearLayer
  node%d = nn.Linear(rnn_size, rnn_size)(node%d)
'''

    def __str__(self):
        return 'linearLayer'

    def genLua(self, node_id, inputs):
        assert len(inputs) == 1, 'linear layer %d / 1' % len(inputs)
        return self.template % (node_id, inputs[0])


class batchnormalizationLayer:
    n_input = 1
    template = '''  -- reluLayer
  node%d = nn.BatchNormalization(512)(node%d)
'''

    def __str__(self):
        return 'batchnomalizationLayer'

    def genLua(self, node_id, inputs):
        assert len(inputs) == 1, 'batchnormalization layer %d / 1' % len(inputs)
        return self.template % (node_id, inputs[0])

class dropoutLayer:
    n_input = 1
    template = '''  -- reluLayer
  node%d = nn.Dropout(0.1)(node%d)
'''

    def __str__(self):
        return 'dropoutLayer'

    def genLua(self, node_id, inputs):
        assert len(inputs) == 1, 'dropout layer %d / 1' % len(inputs)
        return self.template % (node_id, inputs[0])


class sigmoidLayer:
    n_input = 1
    template = '''  -- reluLayer
  node%d = nn.Sigmoid()(node%d)
'''

    def __str__(self):
        return 'sigmoidLayer'

    def genLua(self, node_id, inputs):
        assert len(inputs) == 1, 'sigmoid layer %d / 1' % len(inputs)
        return self.template % (node_id, inputs[0])


class tanhLayer:
    n_input = 1
    template = '''  -- reluLayer
  node%d = nn.Tanh()(node%d)
'''

    def __str__(self):
        return 'tanhLayer'

    def genLua(self, node_id, inputs):
        assert len(inputs) == 1, 'tanh layer %d / 1' % len(inputs)
        return self.template % (node_id, inputs[0])


class reluLayer:
    n_input = 1
    template = '''  -- reluLayer
  node%d = nn.ReLU(true)(node%d)
'''

    def __str__(self):
        return 'reluLayer'

    def genLua(self, node_id, inputs):
        assert len(inputs) == 1, 'relu layer %d / 1' % len(inputs)
        return self.template % (node_id, inputs[0])


class add01Layer:
    n_input = 1
    template = '''  -- add01layer
  node%d = nn.AddConstant(0.1, true)(node%d)
'''

    def __str__(self):
        return 'add01layer'

    def genLua(self, node_id, inputs):
        assert len(inputs) == 1, 'add01 layer %d / 1' % len(inputs)
        return self.template % (node_id, inputs[0])


class mul09Layer:
    n_input = 1
    template = '''  -- mul09layer
  node%d = nn.MulConstant(0.9, true)(node%d)
'''

    def __str__(self):
        return 'mul09layer'

    def genLua(self, node_id, inputs):
        assert len(inputs) == 1, 'mul09 layer %d / 1' % len(inputs)
        return self.template % (node_id, inputs[0])


class mul11Layer:
    n_input = 1
    template = '''  -- mul1layer
  node%d = nn.MulConstant(1.1, true)(node%d)
'''

    def __str__(self):
        return 'mul11layer'

    def genLua(self, node_id, inputs):
        assert len(inputs) == 1, 'mul11 layer %d / 1' % len(inputs)
        return self.template % (node_id, inputs[0])


class muln1Layer:
    n_input = 1
    template = '''  -- muln1layer
  node%d = nn.MulConstant(-1, true)(node%d)
'''

    def __str__(self):
        return 'muln1layer'

    def genLua(self, node_id, inputs):
        assert len(inputs) == 1, 'muln1 layer %d / 1' % len(inputs)
        return self.template % (node_id, inputs[0])


class caddLayer:
    n_input = 2
    template = '''  -- caddLayer
  node%d = nn.CAddTable(){%s}
'''

    def __str__(self):
        return 'caddLayer'

    def genLua(self, node_id, inputs):
        assert len(inputs) == 2, 'cadd layer %d / 2' % len(inputs)
        return self.template % (node_id, ','.join(['node%d'%i for i in inputs]))


class cmulLayer:
    n_input = 2
    template = '''  -- cmulLayer
  node%d = nn.CMulTable(){%s}
'''

    def __str__(self):
        return 'cmulLayer'

    def genLua(self, node_id, inputs):
        assert len(inputs) == 2, 'cmul layer %d / 2' % len(inputs)
        return self.template % (node_id, ','.join(['node%d'%i for i in inputs]))
