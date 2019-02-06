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
import torch.optim as optim


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
# print(summary(model,input_size=(3,224,224)))
class Network_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model
        self.conv1 = nn.Conv2d(3,3,1, stride=4,padding=2)
        self.conv2 = nn.Conv2d(3,96,8,stride=4,padding=4)
        self.maxpool1 = nn.MaxPool2d(3,4,padding=0)
        self.conv3 = nn.Conv2d(3,3,1, stride=8,padding=2)
        self.conv4 = nn.Conv2d(3,96,8,stride=4,padding=4)
        self.maxpool2 = nn.MaxPool2d(7,2,padding=3)
    def forward(self,x):
        out1 = self.model(x)
        y = self.conv1(x)        
        y = self.conv2(y)
        y = self.maxpool1(y)
        y = y.view(y.size(0), -1)
        y = F.normalize(y,dim=1,p=2)
        z = self.conv3(x)
        z = self.conv4(z)
        z = self.maxpool2(z)
        z = z.view(y.size(0), -1)
        z = F.normalize(z,dim=1,p=2)
        out = torch.cat((out1, y, z), 1)
        out = F.normalize(out,dim=1,p=2)
        return out
test1 = Network_1()
print(summary(test1,input_size=(3,224,224)))

# specify loss function (categorical cross-entropy)
criterion = nn.TripletMarginLoss()
# specify optimizer
optimizer = optim.SGD(test1.parameters(), lr=0.01,momentum=0.9,nesterov=True)

#Uncomment after writing trainloader to load data.

'''
n_epochs = 30
n_epochs = 30

valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # validate the model #
    ######################
    model.eval()
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
    
    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model.pt')
        valid_loss_min = valid_loss

'''
