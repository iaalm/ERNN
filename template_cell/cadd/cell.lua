-- automatic generated lua cell
require 'nn'
require 'nngraph'

local cell = {}
function cell.cell(input_size, output_size, rnn_size)
  local inputs = {}
  local outputs = {}
  -- inputLayer
  node0 = nn.Identity()():annotate{name='input_1'}
  inputs[1] = node0
  -- inputLayer
  node1 = nn.Identity()():annotate{name='input_2'}
  inputs[2] = node1
  -- caddLayer
  node2 = nn.CAddTable(){node1,node0}
  -- outputLayer
  outputs[2] = node2:annotate{name='output_2'}
  -- outputLayer(final)
  node4 = nn.Linear(rnn_size, output_size)(node2):annotate{name='output_final'}
  outputs[1] = nn.LogSoftMax()(node4)
  -- inputLayer
  node5 = nn.Identity()():annotate{name='input_3'}
  inputs[3] = node5
  -- outputLayer
  outputs[3] = node5:annotate{name='output_3'}

  return nn.gModule(inputs, outputs)
end

return cell
