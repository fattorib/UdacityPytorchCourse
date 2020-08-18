import torch
from torchvision import datasets, transforms
import helper

train_root = 'Cat_Dog_Data/train'
test_root = 'Cat_Dog_Data/test'

'''
Define:
    1) Transformation pipeline, final step will be transferring to tensor
    2) Load data using ImageFolder
    3) Load into pytorch with Dataloader

'''

transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])


traindata = datasets.ImageFolder(train_root,transform = transform)

trainloader = torch.utils.data.DataLoader(traindata,batch_size = 32,shuffle = True)


images, labels = next(iter(trainloader))
helper.imshow(images[0], normalize=False)
