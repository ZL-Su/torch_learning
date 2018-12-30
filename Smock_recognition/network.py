import torch.nn as nn
import torch.nn.functional as F

def conv_layer(cin, cout, ksize, imsize):
   return nn.Conv2d(cin, cout, ksize), int((imsize-ksize)+1);

def pool_layer(ksize, imsize):
   return nn.MaxPool2d(ksize, ksize), int((imsize-ksize)/ksize+1);

class Net(nn.Module):
   def __init__(self, w):
      super(Net, self).__init__()
      self.size = w

      self.conv1, self.size = conv_layer(3, 6, 5, self.size)
      self.pool, self.size = pool_layer(2, self.size)
      self.conv2, self.size = conv_layer(6, 16, 5, self.size)
      _, self.size = pool_layer(2, self.size)

      self.fc1 = nn.Linear(16*self.size*self.size, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, 4)

   def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = x.view(-1, 16*self.size*self.size)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x