import torch
import TorchEXTLib

a = torch.ones(100).float().cuda()
b = a.clone()
c = torch.zeros(100).float().cuda()


print(TorchEXTLib.my_add(a, b, c))