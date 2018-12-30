import dataset
import network
import torch.optim as optim

def train(epochs=20):
   net = network.Net(dataset.image_size)

   criterion = network.nn.CrossEntropyLoss()
   optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

   for epoch in range(epochs):
      all_loss = 0.0
      for i, data in enumerate(dataset.trainloader, 0):
         inputs, labels = data
         optimizer.zero_grad()

         outputs = net(inputs)
         loss = criterion(outputs, labels)
         loss.backward()
         optimizer.step()

         all_loss += loss.item()

         if i%dataset.batchs == dataset.batchs-1:
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, all_loss/dataset.batchs))
            all_loss = 0.0

   print('Finished training')

   return net;