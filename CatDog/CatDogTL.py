import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
    
import helper
    
    
train_root = 'Cat_Dog_Data/train'
test_root = 'Cat_Dog_Data/test'



train_transforms = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), 
                                                     (0.229, 0.224, 0.225))])


test_transforms = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), 
                                                     (0.229, 0.224, 0.225))])


# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(train_root, transform=train_transforms)
test_data = datasets.ImageFolder(test_root, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

'''
To-do:
    1) Create model with pretrained classifier: Let's use ResNet-152
    2) Freeze all feature detection part of the model, we only want to modify the classifier portion
    3) Train using GPU
'''

model = models.resnet152(pretrained=True)

#Freeze all non-classifier parameters
for param in model.parameters():
    #Turn off gradient tracking for feature layers, as we are not doing any optimization here
    
    param.requires_grad = False
    
    
from collections import OrderedDict

#Redefining classifier 
fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048,512)),
                                ('ReLU', nn.ReLU()),
                                ('fc2', nn.Linear(512,2)),
                                ('output', nn.LogSoftmax(dim = 1))]))



#Updating model with custom classifier
model.fc = fc


criterion = nn.NLLLoss()

optimizer = optim.Adam(model.fc.parameters(),lr = 0.01)


epochs = 5

model.cuda()



for e in range(0,epochs):
    
    running_loss = 0
    for images,labels in trainloader:
        images,labels = images.cuda(),labels.cuda()
        
        optimizer.zero_grad()
        outputs = model.forward(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    print(running_loss/len(trainloader))
    
    if e%2 == 0:
        test_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                
                test_loss += batch_loss.item()
                
                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
        print('Test Accuracy:', accuracy/len(testloader))
        running_loss = 0
        model.train()
    




























