require 'nn'
require 'nngraph'

local cell = {}
function cell.cell(input_size, output_size, rnn_size)
  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
  table.insert(inputs, nn.Identity()()) -- prev_c[L]
  table.insert(inputs, nn.Identity()()) -- prev_h[L]

  local outputs = {}
  -- c,h from previos timesteps
  local prev_h = inputs[3]
  local prev_c = inputs[2]
  -- evaluate the input sums at once for efficiency
  local i2h = nn.Linear(input_size, 4 * rnn_size)(inputs[1]):annotate{name='i2h'}
  local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h'}
  local all_input_sums = nn.CAddTable()({i2h, h2h})

  local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
  local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
  -- decode the gates
  local in_gate = nn.Sigmoid()(n1)
  local forget_gate = nn.Sigmoid()(n2)
  local out_gate = nn.Sigmoid()(n3)
  -- decode the write inputs
  
  local in_transform = nn.Tanh()(n4)
  -- perform the LSTM update
  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
    })
  -- gated cells form the output
  local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  
  table.insert(outputs, next_c)
  table.insert(outputs, next_h)

  -- set up the decoder
  local top_h = outputs[#outputs]
  top_h = nn.Dropout(0.5)(top_h):annotate{name='drop_final'}
  local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return cell

