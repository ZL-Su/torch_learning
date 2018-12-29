import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data

transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5),
                        (0.5, 0.5, 0.5))])

trainset = datasets.ImageFolder(root='./data/train',transform=transform)
trainloader = data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = datasets.ImageFolder(root='./data/test',transform=transform)
testloader = data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('empty', 'smock_a', 'smock_b', 'smock_c')

