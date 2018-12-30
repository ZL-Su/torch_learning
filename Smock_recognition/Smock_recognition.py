import matplotlib.pyplot as plt
import numpy as np
import trainer
import torchvision.utils as tvutils
from trainer import dataset

is_train = True

def imshow(image):
   image = image/2 + 0.5
   npimg = image.numpy()
   plt.imshow(np.transpose(npimg, (1, 2, 0)))
   plt.show()

def get_model(is_train=False):
   if is_train == True:
      model = trainer.train(8)
      dataset.torch.save(model, './model/model.pt')
      return model;
   if is_train == False:
      model = dataset.torch.load('./model/model.pt')
      model.eval()
      return model;

if __name__ == "__main__":
   dataitr = iter(dataset.trainloader)
   images, labels = dataitr.next()
   imshow(tvutils.make_grid(images))
   print(' '.join('%5s' % dataset.classes[labels[j]] for j in range(dataset.nclasses)))

   model = get_model(True)

   for N in range(5):
      dataitr = iter(dataset.testloader)
      images, labels = dataitr.next()
      #imshow(tvutils.make_grid(images))
      print('Ground truth: ', ' '.join('%5s' % dataset.classes[labels[j]] for j in range(dataset.nclasses)))

      outputs = model(images)
      _, predicted = dataset.torch.max(outputs, 1)
      print('Predicted: ', ' '.join('%5s' % dataset.classes[predicted[j]] for j in range(dataset.nclasses)))

   print('Finish')