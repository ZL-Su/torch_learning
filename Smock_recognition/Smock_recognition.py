import matplotlib.pyplot as plt
import numpy as np
import trainer
from trainer import dataset

def imshow(image):
   image = image /2 + 0.5
   npimg = image.numpy()
   plt.imshow(np.transpose(npimg, (1, 2, 0)))
   plt.show()

dataitr = iter(dataset.trainloader)
images, labels = dataitr.next()


