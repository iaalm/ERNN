class inputLayer:
    template = '''  -- inputLayer
  node%d = nn.Identity()()::annotate('input_%d')
  inputs[%d] = node%d
'''

    def __init__(self, layer_id):  # layer_id is special for io layers
        self.layer_id = layer_id

    def __str__(self):
        return 'inputLayer'

    def genLua(self, node_id, inputs):
        assert len(inputs) == 0, ''
        return self.template % (node_id, self.layer_id + 1, self.layer_id + 1, node_id)


class outputLayer:
    n_input = 0
    template = '''  -- outputLayer
  outputs[%d] = node%d:annotate('output_%d')
'''
    template_out = '''  -- outputLayer(final)
  node%d = nn.Linear(rnn_size, output_size)(node%d):annotate('output_final')
  outputs[%d] = nn.LogSoftMax()(node%d)
'''

    def __init__(self, layer_id):  # layer_id is special for io layers
        self.layer_id = layer_id

    def __str__(self):
        return 'outputLayer'

    def genLua(self, node_id, inputs):
        assert len(inputs) == 1, ''
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
        assert len(inputs) == 1, ''
        return self.template % (node_id, inputs[0])


class reluLayer:
    n_input = 1
    template = '''  -- reluLayer
  node%d = nn.ReLU(true)(node%d)
'''

    def __str__(self):
        return 'reluLayer'

    def genLua(self, node_id, inputs):
        assert len(inputs) == 1, ''
        return self.template % (node_id, inputs[0])


class caddLayer:
    n_input = 2
    template = '''  -- caddLayer
  node%d = nn.CAddTable(){%s}
'''

    def __str__(self):
        return 'caddLayer'

    def genLua(self, node_id, inputs):
        assert len(inputs) >= 2, ''
        return self.template % (node_id, inputs[0])


class cmulLayer:
    n_input = 2
    template = '''  -- cmulLayer
  node%d = nn.CMulTable(){%s}
'''

    def __str__(self):
        return 'cmulLayer'

    def genLua(self, node_id, inputs):
        assert len(inputs) >= 2, ''
        return self.template % (node_id, inputs[0])
