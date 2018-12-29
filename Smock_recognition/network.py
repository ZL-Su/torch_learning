import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
   def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(3, 128, 5)
      self.pool = nn.MaxPool2d(2, 2)
      self.conv2 = nn.Conv2d(128, 128*2, 5)
      self.fc1 = nn.Linear(128*2*5*5, 1024)
      self.fc2 = nn.Linear(1024, 512)
      self.fc3 = nn.Linear(512, 4)

   def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = x.view(-1, 128*2*5*5)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x