import torch.nn as nn
import torch.nn.functional as F
import torchvision
from base import BaseModel
from .vgg_cifar import VGG, make_layers, cfg
from .resnet_cifar import *
from .decision_tree import Tree
from sklearn import svm

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def vgg16_bn(norm_layer=nn.BatchNorm2d, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True, norm_layer=norm_layer), **kwargs)
    
    return model

def vgg16_bn_imagenet(num_classes=10):
    model = torchvision.models.vgg16(weights = 'IMAGENET1K_V1')
        
    num_ftrs = model.classifier[0].in_features
    model.classifier = nn.Sequential(nn.Linear(num_ftrs,1024),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(1024, num_classes))
    return model

def resnet18_imagenet(num_classes=10):
    model = torchvision.models.resnet18(weights = 'IMAGENET1K_V1') 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def vgg16_bn(norm_layer=nn.BatchNorm2d, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True, norm_layer=norm_layer), **kwargs)
    
    return model

def resnet18_bn(num_classes=10, norm_layer=nn.BatchNorm2d):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, norm_layer)

class LinModel(BaseModel):
    def __init__(self, num_classes=2):
        super().__init__()
        self.lin1 = nn.Linear(7,6)
        self.lin2 = nn.Linear(6,4)
        self.lin3 = nn.Linear(4,num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        return x

class LinModelIncome(BaseModel):
    def __init__(self, num_classes=2):
        super().__init__()
        self.lin1 = nn.Linear(106,60)
        self.lin2 = nn.Linear(60,15)
        self.lin3 = nn.Linear(15,num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        #self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        return x


class LinModelCompas(BaseModel):
    def __init__(self, num_classes=2):
        super().__init__()
        self.lin1 = nn.Linear(21,200)
        self.lin2 = nn.Linear(200,20)
        self.lin3 = nn.Linear(20,num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.relu(self.lin1(x))
        #x = self.dropout(x)
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        return x

class TesseractDrebinNetSpecial(BaseModel):
    def __init__(self, num_classes=2):
        super().__init__()
        self.lin1 = nn.Linear(68,200)
        self.lin2 = nn.Linear(200,20)
        self.lin3 = nn.Linear(20,num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.relu(self.lin1(x))
        #x = self.dropout(x)
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        return x
    
def decision_tree():
    return Tree()




class TesseractDrebinNet(nn.Module):
    def __init__(self, input_size=84725):
        super(TesseractDrebinNet, self).__init__()
        self.l1 = nn.Linear(input_size, 200)
        self.l2 = nn.Linear(200,200)
        self.l3 = nn.Linear(200,2)
        self.activation = F.relu
    def forward(self, x):
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        x = self.activation(self.l3(x))
        return x

class TesseractDrebinNet2(nn.Module):
    def __init__(self, input_size=84725):
        super(TesseractDrebinNet2, self).__init__()
        self.l1 = nn.Linear(input_size, 400)
        self.l2 = nn.Linear(400,400)
        self.l3 = nn.Linear(400,400)
        self.l5 = nn.Linear(400,2)
        self.activation = F.relu
    def forward(self, x):
        x = self.activation(F.dropout(self.l1(x), training=self.training))
        x = self.activation(F.dropout(self.l2(x), training=self.training))
        x = self.activation(F.dropout(self.l3(x), training=self.training))
        x = self.l5(x)
        return x



class AndroZooDrebin(nn.Module):
    def __init__(self, input_size=62628):#65278, 67991, 66039
        super(AndroZooDrebin, self).__init__()
        self.l1 = nn.Linear(input_size, 400)
        self.l2 = nn.Linear(400,400)
        self.l3 = nn.Linear(400,400)
        self.l5 = nn.Linear(400,2)
        self.activation = F.relu
    def forward(self, x):
        x = self.activation(F.dropout(self.l1(x), training=self.training))
        x = self.activation(F.dropout(self.l2(x), training=self.training))
        x = self.activation(F.dropout(self.l3(x), training=self.training))
        x = self.l5(x)
        return x
    
     

     
class nonLinSVM():
    def __init__(self):
        self.NuSVC = svm.SVC(cache_size=1000, verbose=True, C = 1.0, gamma=0.1, probability=True, class_weight="balanced") #.SVC(kernel = 'poly', degree=3, verbose=True, cache_size=1000, probability=True)#
    
    def fit(self, x, y):
        self.NuSVC.fit(x,y)
    
    def predict(self, x):
        return self.NuSVC.predict(x)
    
    def predict_proba(self, x):
        return self.NuSVC.predict_proba(x)