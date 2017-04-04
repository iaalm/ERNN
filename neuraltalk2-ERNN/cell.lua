-- automatic generated lua cell
require 'nn'
require 'nngraph'

local cell = {}
function cell.cell(input_size, output_size, rnn_size)
  local inputs = {}
  local outputs = {}
  -- inputLayer
  node0 = nn.Identity()()
  inputs[2] = node0
  -- inputLayer
  node1 = nn.Identity()()
  inputs[3] = node1
  -- outputLayer
  outputs[3] = node1
  -- inputLayer
  node3 = nn.Identity()()
  inputs[1] = node3
  -- reluLayer
  node4 = nn.ReLU(true)(node3)
  -- outputLayer
  outputs[2] = node0
  -- outputLayer(final)
  node6 = nn.Linear(rnn_size, output_size)(node4)
  outputs[1] = nn.LogSoftMax()(node6)

  return nn.gModule(inputs, outputs)
end

return cell
