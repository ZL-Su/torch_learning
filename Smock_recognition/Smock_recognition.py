import matplotlib.pyplot as plt
import numpy as np
import trainer
import torchvision.utils as tvutils
from trainer import dataset

def imshow(image):
   image = image/2 + 0.5
   npimg = image.numpy()
   plt.imshow(np.transpose(npimg, (1, 2, 0)))
   plt.show()

if __name__ == "__main__":
   dataitr = iter(dataset.trainloader)
   images, labels = dataitr.next()

   imshow(tvutils.make_grid(images))
   print(' '.join('%5s' % dataset.classes[labels[j]] for j in range(4)))

   trainer.train()