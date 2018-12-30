import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import PIL.Image as Image

image_size = 256

transform = transforms.Compose([
   transforms.Resize((image_size,image_size), Image.BICUBIC),
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5),
                        (0.5, 0.5, 0.5))])

batchs = 4

trainset = datasets.ImageFolder(root='./data/train',transform=transform)
trainloader = data.DataLoader(trainset, batch_size=batchs, shuffle=True, num_workers=2)

testset = datasets.ImageFolder(root='./data/test',transform=transform)
testloader = data.DataLoader(testset, batch_size=batchs, shuffle=True, num_workers=2)

classes = ('empty', 'smock_a', 'smock_b', 'smock_c')

nclasses = len(classes)