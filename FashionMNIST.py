import torch
import torch.nn as nn
import torch.nn.functional as F



from torchvision import datasets, transforms
import helper

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])


# Download and load the train data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)




# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)



for images, labels in trainloader:
    images = images.view(images.shape[0],-1)
    
hidden_1,hidden_2 = 256,128

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(784,hidden_1)
        self.fc1 = nn.Linear(hidden_1,hidden_2)
        self.output = nn.Linear(hidden_2,10)
        


    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.fc1(x))
        x = self.output(x)
        x = F.log_softmax(x,dim = 1)
        return x
    
model = Network()
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#Putting model on GPU
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"



device = torch.device(dev)
print('We are using:', device)

#Putting model and data on GPU
model = model.cuda(device)

epochs = 6

for epoch in range(0,epochs):
    #Code for single epoch
    running_loss = 0
    for images, labels in trainloader:
        images = images.cuda(device)
        labels = labels.cuda(device)
        images = images.view(images.shape[0],-1)
        #Zeroing optimizer gradient
        optimizer.zero_grad()
        #Computing model forward pass
        outputs = model.forward(images)
        loss = criterion(outputs,labels)
        #Backpropogating loss
        loss.backward()
        #advancing optimizer
        optimizer.step()
        running_loss += loss.item()
    print(running_loss/len(trainloader))



import helper

# Test out your network!
import matplotlib

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
# Convert 2D image to 1D vector
img = img.resize_(1, 784)

model = model.cpu()

# TODO: Calculate the class probabilities (softmax) for img
ps = torch.exp(model.forward(img))

# Plot the image and probabilities
helper.view_classify(img.resize_(1, 28, 28), ps, version='Fashion')
