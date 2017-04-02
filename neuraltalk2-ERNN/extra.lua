require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'hdf5'
net_utils = require 'misc.net_utils'
require 'misc.LanguageModel' -- just load

cnn = torch.load('/d/whz/slurm/whz-baseline/model_id.t7').protos.cnn
h5 = hdf5.open('/s/coco/cocotalk.h5','r')
data = torch.Tensor(616767,512)
ls = h5:read('/label_start_ix'):all()
le = h5:read('/label_end_ix'):all()
out = hdf5.open('cocosent.h5','w')
out:write('/labels',h5:read('/labels'):all())
out:write('/label_start_ix',h5:read('label_start_ix'):all())
out:write('/label_end_ix',h5:read('label_end_ix'):all())
cnn:evaluate()
for i=1,123287 do 
  s=ls[i];
  e=le[i];
  for j=s,e do 
    print(j);
    data[{{j,j},{1,512}}] = cnn:forward(net_utils.prepro(h5:read('/images'):partial({i,i},{1,3},{1,256},{1,256}):float(),true,true)):float()
  end 
end
out:write('/feats',data)
out:close()
