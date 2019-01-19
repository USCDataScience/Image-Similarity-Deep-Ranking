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


model = models.vgg16(pretrained=True)  
new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier = new_classifier
print(summary(model,input_size=(3,224,224)))
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,96,8, stride=16,padding=1)
        self.maxpool1 = nn.MaxPool2d(3,4,padding=1)
        self.conv2 = nn.Conv2d(3,96,8, stride=32,padding=1)
        self.maxpool2 = nn.MaxPool2d(7,2,padding=3)
        
    def forward(self,x):
        out1 = model(x)
        y = self.conv1(x)
        y = self.maxpool1(y)
        y = y.view(y.size(0), -1)
        y = F.normalize(y,dim=1,p=2)
        z = self.conv2(x)
        z = self.maxpool2(z)
        z = z.view(y.size(0), -1)
        z = F.normalize(z,dim=1,p=2)
        #out = torch.cat((out1,y,z),1)
        out = torch.cat((out1, y, z), 1)
        return out
test = Network()
print(summary(test,input_size=(3,224,224)))
