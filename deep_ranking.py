import io
import torch 
from PIL import Image
import requests
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

IMG_URL = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg'
response = requests.get(IMG_URL)
img = Image.open(io.BytesIO(response.content))  # Read bytes and store as an img.
# img.show()
min_img_size = 224  # The min size, as noted in the PyTorch pretrained models doc, is 224 px.
transform_pipeline = transforms.Compose([transforms.Resize(min_img_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
img = transform_pipeline(img)

# PyTorch pretrained models expect the Tensor dims to be (num input imgs, num color channels, height, width).
# Currently however, we have (num color channels, height, width); let's fix this by inserting a new axis.
img = img.unsqueeze(0)  # Insert the new axis at index 0 i.e. in front of the other axes/dims. 
img = Variable(img)

model = models.vgg16(pretrained=True)  # This may take a few minutes.
new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier = new_classifier

# prediction = vgg(img)  # Returns a Tensor of shape (batch, num class labels)
#prediction = prediction.data.numpy().argmax()  # Our prediction will be the index of the class label with the largest value.
#vgg.add_module('fin',module=nn.Linear(20,4000))

# x = model(img)
# x = nn.AvgPool2d(x)
# print(x)
# x = F.relu(x)
# print(x)
# t = F.normalize(x,p=2, dim=1)
# first_conv = nn.Conv2d(in_channels=3,out_channels=96,kernel_size=8,stride=16,padding=0)
# first_max = nn.MaxPool2d(kernel_size=8,stride=4,padding=0)

model = models.vgg16(pretrained=True)  # This may take a few minutes.
new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier = new_classifier
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,96,8, stride=16,padding=1)
        self.maxpool1 = nn.MaxPool2d(3,4,padding=1)
        self.conv2 = nn.Conv2d(3,96,8, stride=32,padding=1)
        self.maxpool2 = nn.MaxPool2d(7,2,padding=1)
        
    def forward(self,x):
        out1 = model(x)
        y = self.conv1(x)
        y = self.maxpool1(y)
        #y = F.normalize(y,dim=1,p=2)
        z = self.conv2(x)
        z = self.maxpool2(z)
        #z = F.normalize(z,dim=1,p=2)
        z = emb(z)
        return out
test = Network()
print(summary(test,input_size=(3,224,224)))
