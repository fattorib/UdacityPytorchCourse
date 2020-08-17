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



    
hidden_1,hidden_2 = 256,128
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784,hidden_1)
        self.fc2 = nn.Linear(hidden_1,hidden_2)
        self.fc3 = nn.Linear(hidden_2,10)
        
        self.dropout = nn.Dropout(p= 0.2)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x,dim = 1)
        return x
    

model = Network()
model.eval()

#Loading saved model parameters
state_dict = torch.load('checkpoint.pth')


model.load_state_dict(state_dict)


with torch.no_grad():
    accuracy = 0
    for images, labels in testloader:
        images = images.view(images.shape[0],-1)
        log_ps = model(images)
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
            

    print("Test Accuracy:",accuracy.item()/len(testloader))


with torch.no_grad():
    accuracy = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0],-1)
        log_ps = model(images)
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
            

    print("Train Accuracy:",accuracy.item()/len(trainloader))