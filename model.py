from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
import numpy as np

import os

# defining network class and passing in nn.Module, a package that includes all the neural network functionality


class Net(nn.Module):
    # constructor
    def __init__(self):
        # immediately call the super class
        super(Net, self).__init__()
        # define network layers
        # 2d convolutional layers (input channels, output channels, kernel size)
        self.conv1 = nn.Conv2d(3, 3, 5)
        self.conv2 = nn.Conv2d(3, 5, 5)
        self.conv3 = nn.Conv2d(5, 5, 2)
        # Pooling layer (kernel size, step)
        self.pool = nn.MaxPool2d(2, 2)
        # linear layers (input features, output features)
        self.fc1 = nn.Linear(5 * 62 * 62, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 4)
        # ends with 4 features, one for each type of cancer and one for 'no'

    # forward propagation function
    def forward(self, x):
        # pass through layer, rectified linear function and pool all at once
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # print(x[0].shape)
        # transform into linear form
        x = x.view(-1, 5 * 62 * 62)
        # print(x[0].shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # return x
        return F.softmax(x, dim=1)


def load_model():

    FILE = 'model copy.pth'

    # instantiate network
    net = Net()

    # load the parameters from our saved model
    net.load_state_dict(torch.load(FILE))

    # print(net)

    # set the network to evaluation mode (does not calculate gradients)
    net.eval()

    # return the network object to process
    return net


# variable to set image size
imsize = 512

# function to scale the image and transform into tensor at once
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])


def image_loader(image_name):
    # load image with pillow
    image = Image.open(image_name).convert(mode='RGB')

    # crops to image size and transforms into tensor
    image = loader(image).float()
    image = Variable(image, requires_grad=True)

    # adds extra dimension to tensor so it can be processed
    image = image.unsqueeze(0)

    return image


# premade function to transform image into pytorch tensor
trans = transforms.ToTensor()


# function to take in an input image path and return class
def run_model(image_path):

    # load the image from the imageloader function
    image = image_loader(image_path)

    # load the model
    net = load_model()

    # push the image through the network
    tens = net.forward(image)

    # find the class with the highest probability
    output, scan_class = torch.max(tens, 1)

    scan_class = scan_class.item()

    # select type of cancer
    cancer_type = ''
    if scan_class == 1:
        cancer_type = 'meningioma'
    elif scan_class == 2:
        cancer_type = 'glioma'
    elif scan_class == 3:
        cancer_type = 'pituitary'
    else:
        cancer_type = 'no cancer'

    out_dict = {
        'probability_tensor': tens,
        'highest_prob': output.item(),
        'cancer_class': scan_class,
        'cancer_type': cancer_type
    }
    # return all needed data (can change)
    return out_dict


# testing
# for each image we already have installed
# directory = './static/images'
# for fileName in os.listdir(directory):
#     if fileName.endswith('.png'):
#         # run them through the model and see the output
#         output = run_model(directory + '/' + fileName)
#         print(fileName, output)
