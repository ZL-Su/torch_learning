import network
import torch.optim as optim

net = network.Net()

criterion = network.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
   all_loss = 0.0
   for i, data in enumerate(trainloader, 0):
      inputs, labels = data
      optimizer.zero_grad()

      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      all_loss += loss.item()

      if i%2000 == 1999:
         print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, all_loss/2000))
         all_loss = 0.0

print('Finished training')