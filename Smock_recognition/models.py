import torch.nn as nn
import torchvision.models as models

class ResNet18(nclasses):
   model = models.resnet18(pretrained=True)
   nfeat = model.fc.in_features
   model.fc = nn.Linear(nfeat, nclasses)
