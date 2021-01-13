import torch # All the torch modules
import torch.nn as nn
import resnet_torch2 as pyresnet
#import alexnet_torch as pyalexnet
import torch.nn.functional as F
#import torchvision.models as models

class Net(nn.Module):
    # Just a basic conv + pooling + FC network
    # Suitable only for 28 x 28 greyscale images. Was used for MNIST, FashionMNIST 
    # Change conv1 parameters for using it for color images. 
    # Change the fc1 n_input features for images of different sizes (eg. CIFAR 32 x 32 images)
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        fc1_out = self.fc1(x)
        x = F.relu(fc1_out)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output
        #return [x, fc1_out]



class Net2(nn.Module):
    # Network for MNIST like data
    # Explain the nomenclature, self.features and .classifier structure
    # The network architecture is very similar to Net1 and both have been used for MNIST, FashionMNIST but this network was separately created
    # because I needed a structure like that of convolutional layers in a single attribute and fully-connected layers as other attribute

    def __init__(self):
        super(Net2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            #nn.MaxPool2d(2),
            #nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.Dropout(),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 9216)
        x = self.classifier(x)
        return x




class Net2_CIFAR(nn.Module):
    # Network for CIFAR like data
    def __init__(self, output_classes=100):
        super(Net2_CIFAR, self).__init__()
        # Use n_classes=10 for CIFAR10 network

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.Dropout(),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, output_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 12544)
        x = self.classifier(x)
        return x


#-------------------------------------------------------



class MyResNet(nn.Module):
    def __init__(self, version='4', bn=False, n_classes=10, in_maps=1):
        super(MyResNet, self).__init__()
        #self.convnet = pyresnet.resnet18(pretrained=True)
        self.convnet = pyresnet.ResNet18(version=version, bn=bn, n_classes=n_classes, in_maps=in_maps)
        #self.convnet.requires_grad = False 
        num_ftrs = self.convnet.fc.in_features
        self.fc2 = nn.Linear(num_ftrs, n_classes)
        self.convnet.fc = nn.Linear(num_ftrs, num_ftrs) # 200 is the number of output classes in CUB_2011/200 dataset

    def forward(self, inp):
        output, output_intermediate = self.convnet(inp)     
        return self.fc2(output), output_intermediate



class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()
        self.convnet = pyalexnet.alexnet(pretrained=True) 
        num_ftrs = self.convnet.classifier[6].in_features
        self.convnet.classifier[6] = nn.Linear(num_ftrs, 200, bias=True) # 200 is the number of output classes in CUB_2011/200 dataset

    def forward(self, inp):
        output, output_intermediate = self.convnet(inp)     
        return output, output_intermediate



class attr_encoder(nn.Module):
    def __init__(self, out_size=0):
        super(attr_encoder, self).__init__()
        self.conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)
        #self.conv3 = nn.Conv2d(512, 1, kernel_size=1, padding=0)
        #self.fc1 = nn.Linear(28*28, out_size, bias=True)
        self.fc1 = nn.Linear(512*3*3, out_size, bias=True)
        self.activ = nn.LeakyReLU()
        self.pool = nn.AdaptiveAvgPool2d(3)

    def forward(self, inp):
        x = self.activ( self.bn1( self.conv1(inp) ) )
        x = self.activ( self.bn2( self.conv2(x) ) )
        #x = self.activ( self.conv3(x) )
        x = self.pool(x)
        x = x.view(-1, 512*3*3)
        x = self.fc1(x)
        return x

class convBlock(nn.Module):
    def __init__(self, in_maps, out_maps):
        super(convBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_maps, out_maps, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_maps, out_maps, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.bn = nn.BatchNorm2d(out_maps)
        self.activ = nn.LeakyReLU()

    def forward(self, inp):
        x = self.activ( self.conv1(inp) )
        x = self.bn( self.conv2(x) )
        return self.pool( self.activ(x) )


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.conv1 = convBlock(3, 16)
        self.conv2 = convBlock(16, 32)
        self.conv3 = convBlock(32, 64)
        self.conv4 = convBlock(64, 128)
        self.conv5 = convBlock(128, 256)
        self.fc = nn.Linear(256*7*7, 256*7*7, bias=True)
        self.activ = nn.ReLU()

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc(x.view(-1, 256*7*7))
        return x
        

class explainer(nn.Module):
    def __init__(self, in_size=30, max_elem=15, n_classes=200):
        super(explainer, self).__init__()
        self.max_elem = max_elem
        self.fc1 = nn.Linear(in_size, n_classes, bias=True)
        self.drop = nn.Dropout(0.01)

    def forward(self, inp):
        # Select the max elems by multiplying input by the appropriate tensor
        x = self.drop(inp)
        return self.fc1(x)
	

class decoder(nn.Module):
    def __init__(self, in_size=9216):
        super(decoder, self).__init__()
        self.fc1 = nn.Linear(in_size, 7*7*256, bias=True)
        self.trconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.trconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.trconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.trconv4 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.trconv5 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        #self.conv4 = nn.Conv2d(8, 16, kernel_size=5, padding=2, bias=True)
        #self.trconv5 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        #self.final_conv = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool = nn.MaxUnpool2d(2)
   
        self.activ = nn.ReLU()

    def forward(self, inp):
        x = self.fc1(inp)
        x = x.view(-1, 256, 7, 7)
        x = F.interpolate(x, scale_factor=2)
        x = self.activ( self.bn1( self.conv1(x) ) )
        x = F.interpolate(x, scale_factor=2)
        x = self.activ( self.bn2( self.conv2(x) ) )
        x = F.interpolate(x, scale_factor=2)
        x = self.activ( self.bn3( self.conv3(x) ) )
        x = F.interpolate(x, scale_factor=2)
        x = self.activ( self.bn4( self.conv4(x) ) )
        x = F.interpolate(x, scale_factor=2)
        x = self.activ( self.conv5(x) )
        #x = torch.sigmoid(x)
        #x = self.activ( self.conv4(x) )
        #x = self.activ( self.trconv5(x) )
        #x = self.final_conv(x)
        return x


class ssae(nn.Module):
    def __init__(self):
        super(ssae, self).__init__()
        self.conv1 = convBlock(3, 16)
        self.conv2 = convBlock(16, 32)
        self.trconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.trconv5 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.conv2 = nn.Conv2d(3,3, kernel_size=3, padding=1)
    def forward(self, inp):
        x = self.conv1(inp)
        x = self.conv2(x)
        x = self.trconv3(x)
        return self.trconv5( self.conv1(inp) )



#--------------------------------------------------------------------

class Net2_MNIST_old(nn.Module):
    # Network for MNIST like data but for the new experiments
    def __init__(self, inp_shape=[28, 28]):
        super(Net2_MNIST_old, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.Dropout(), # Some old models may contain dropout. Be aware of that 
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 14 * 14, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x_inter = self.features(x)
        x = x_inter.view(-1, 32 * 14 * 14)
        x = self.classifier(x)
        return x, x_inter

class Net2_MNIST_new(nn.Module):
    # Network for MNIST like data but for the new experiments. New version with no dropout
    def __init__(self, inp_shape=[28, 28]):
        super(Net2_MNIST_new, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            #nn.Dropout(), # Some old models may contain dropout. Be aware of that 
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 14 * 14, 128),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x_inter = self.features(x)
        x = x_inter.view(-1, 32 * 14 * 14)
        x = self.classifier(x)
        return x, x_inter


class attr_MNIST(nn.Module):
    def __init__(self, out_size=49*2):
        super(attr_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*3*3, out_size, bias=True)
        self.activ = nn.LeakyReLU()
        self.pool = nn.AdaptiveAvgPool2d(3)

    def forward(self, inp):
        x = self.activ( self.conv1(inp) )
        #x = self.activ( self.conv2(x) )
        x = self.pool(x).view(-1, 64*3*3)
        x = self.activ(self.fc1(x))
        #x = self.fc1(x)
        return x


class decode_MNIST(nn.Module):
    def __init__(self, in_size=49*2):
        super(decode_MNIST, self).__init__()
        self.fc1 = nn.Linear(in_size, 49*2, bias=True)
        self.trconv1 = nn.ConvTranspose2d(2, 8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.trconv2 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.activ = nn.ReLU()

    def forward(self, inp):
        x = self.fc1(inp)
        x = x.view(-1, 2, 7, 7)
        x = self.activ( self.trconv1(x) )
        x = self.activ( self.trconv2(x) )
        return x

class decode2_MNIST(nn.Module):
    def __init__(self, in_size=49*2):
        super(decode2_MNIST, self).__init__()
        self.fc1 = nn.Linear(in_size, 49*2, bias=True)
        self.fc2 = nn.Linear(8*28*28, 28*28, bias=True)
        self.trconv1 = nn.ConvTranspose2d(2, 4, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.trconv2 = nn.ConvTranspose2d(4, 8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.activ = nn.ReLU()

    def forward(self, inp):
        x = self.fc1(inp)
        x = x.view(-1, 2, 7, 7)
        x = self.activ( self.trconv1(x) )
        x = self.activ( self.trconv2(x) )
        x = x.view(-1, 8*28*28)
        x = self.activ( self.fc2(x) ).view(-1, 1, 28, 28)
        return x

#--------------------------------------------------------------------

class Net2_CIFAR(nn.Module):
    # Network for CIFAR like data but for the new experiments
    def __init__(self, inp_shape=[3, 32, 32]):
        super(Net2_CIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x_inter = self.features(x)
        x = x_inter.view(-1, 32 * 16 * 16)
        x = self.classifier(x)
        return x, x_inter

class attr_CIFAR(nn.Module):
    def __init__(self, out_size=49*2, in_maps=256):
        super(attr_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(in_maps, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*3*3, out_size, bias=True)
        #self.activ = nn.LeakyReLU()
        self.activ = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(3)

    def forward(self, inp):
        x = self.activ( self.conv1(inp) )
        #x = self.activ( self.conv2(x) )
        x = self.pool(x).view(-1, 64*3*3)
        #x = self.fc1(x) # Till model 9
        #x = x.view(-1, 64*3*)
        x = self.activ(self.fc1(x))
        #x = self.fc1(x)
        return x



class decode_CIFAR(nn.Module):
    def __init__(self, in_size=64*2):
        super(decode_CIFAR, self).__init__()
        self.fc1 = nn.Linear(in_size, 64*2, bias=True)
        self.trconv1 = nn.ConvTranspose2d(2, 8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.trconv2 = nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.activ = nn.ReLU()

    def forward(self, inp):
        x = self.fc1(inp)
        x = x.view(-1, 2, 8, 8)
        x = self.activ( self.trconv1(x) )
        x = self.activ( self.trconv2(x) )
        return x



class attr_RN18_multi(nn.Module):
    def __init__(self, out_size=49*2, in_maps1=256, in_maps2=512):
        super(attr_RN18_multi, self).__init__()
        self.conv1 = nn.Conv2d(in_maps1, 64, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(in_maps2, 64, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*3*3, out_size, bias=True)
        #self.activ = nn.LeakyReLU()
        self.activ = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(3)

    def forward(self, inter_list):
        x1 = self.activ( self.conv1(inter_list[0]) )
        x2 = self.activ( self.conv2(inter_list[1]) )
        x = torch.cat((x1, x2), 1)
        x = self.activ( self.conv3(x) )
        x = self.pool(x).view(-1, 64*3*3)
        #x = self.fc1(x) # Till model 9
        #x = x.view(-1, 64*3*)
        x = self.activ(self.fc1(x))
        #x = self.fc1(x)
        return x









if __name__ == '__main__':
    #inp = torch.zeros(16, 196)
    inp2 = torch.zeros(2, 1, 28, 28)
    #inp3 = torch.zeros(2, 254, 14, 14)
    #d = decoder(in_size=9216)
    #g = attr_encoder(out_size=9216)
    #e = encoder()
    #print (e(inp2).shape)
    #h = explainer()
    #f = MyResNet()
    #out = d(inp)
    #print (out.shape)
    #out = h(inp2)
    #out, inter = f(inp3)	
    #print (inter.shape)
    print ('All good')








