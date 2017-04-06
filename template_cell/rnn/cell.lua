-- automatic generated lua cell
require 'nn'
require 'nngraph'

local cell = {}
function cell.cell(input_size, output_size, rnn_size)
  local inputs = {}
  local outputs = {}
  -- inputLayer
  node0 = nn.Identity()():annotate{name='input_3'}
  inputs[3] = node0
  -- inputLayer
  node1 = nn.Identity()():annotate{name='input_1'}
  inputs[1] = node1
  -- linearLayer
  node2 = nn.Linear(rnn_size, rnn_size)(node1)
  -- outputLayer
  outputs[3] = node0:annotate{name='output_3'}
  -- inputLayer
  node4 = nn.Identity()():annotate{name='input_2'}
  inputs[2] = node4
  -- linearLayer
  node5 = nn.Linear(rnn_size, rnn_size)(node4)
  -- caddLayer
  node6 = nn.CAddTable(){node5,node2}
  -- outputLayer(final)
  node7 = nn.Linear(rnn_size, output_size)(node6):annotate{name='output_final'}
  outputs[1] = nn.LogSoftMax()(node7)
  -- outputLayer
  outputs[2] = node6:annotate{name='output_2'}

  return nn.gModule(inputs, outputs)
end

return cell
