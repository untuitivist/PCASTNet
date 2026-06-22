import torch
import torch.nn as nn
import warnings

from function import calc_mean_std

#----网络参数----

num_classes = 4

#---------------

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),  # 0
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 1
    nn.Conv2d(3, 64, (3, 3)),  # 2
    nn.ReLU(),  # 3 relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 4
    nn.Conv2d(64, 64, (3, 3)),  # 5
    nn.ReLU(),  # 6 relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # 7
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 8
    nn.Conv2d(64, 128, (3, 3)),  # 9
    nn.ReLU(),  # 10 relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 11
    nn.Conv2d(128, 128, (3, 3)),  # 12
    nn.ReLU(),  # 13 relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # 14
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 15
    nn.Conv2d(128, 256, (3, 3)),  # 16
    nn.ReLU(),  # 17 relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 18
    nn.Conv2d(256, 256, (3, 3)),  # 19
    nn.ReLU(),  # 20 relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 21
    nn.Conv2d(256, 256, (3, 3)),  # 22
    nn.ReLU(),  # 23 relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 24
    nn.Conv2d(256, 256, (3, 3)),  # 25
    nn.ReLU(),  # 26 relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # 27
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 28
    nn.Conv2d(256, 512, (3, 3)),  # 29
    nn.ReLU(),  # 30 relu4-1, this is the last layer used to style transfer
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 31
    nn.Conv2d(512, 512, (3, 3)),  # 32
    nn.ReLU(),  # 33 relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 34
    nn.Conv2d(512, 512, (3, 3)),  # 35
    nn.ReLU(),  # 36 relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 37
    nn.Conv2d(512, 512, (3, 3)),  # 38
    nn.ReLU(),  # 39 relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # 40
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 41
    nn.Conv2d(512, 512, (3, 3)),  # 42
    nn.ReLU(),  # 43 relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 44
    nn.Conv2d(512, 512, (3, 3)),  # 45
    nn.ReLU(),  # 46 relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 47
    nn.Conv2d(512, 512, (3, 3)),  # 48
    nn.ReLU(),  # 49 relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 50
    nn.Conv2d(512, 512, (3, 3)),  # 51
    nn.ReLU()  # 52 relu5-4
)

classifier = lambda num_classes: nn.Sequential(
    nn.AdaptiveAvgPool2d((7, 7)),
    nn.Flatten(),
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, num_classes),
)



class VGG(nn.Module):
    def __init__(self, encoder, classifier):
        super(VGG, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.mse_loss = nn.MSELoss()

        self.make_encoder()

    def make_encoder(self):
        enc_layers = list(self.encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:])  # relu4_1 -> relu5_4


    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input


    def forward(self, content):
        """ 执行分类任务 """
        features = self.enc_5(self.encode(content))
        return self.classifier(features)
    
    def save(self, save_pth):
        torch.save(self.encoder.state_dict(), save_pth.replace(".pth", "_encoder.pth"))
        torch.save(self.classifier.state_dict(), save_pth.replace(".pth", "_classifier.pth"))
        print(' - Model saved at', save_pth)

def build_VGG_module(num_classes, 
        encoder_pth=None, 
        classifier_pth=None,
        ): 

    net = VGG(vgg, classifier(num_classes))

    if encoder_pth:
        net.encoder.load_state_dict(torch.load(encoder_pth))
        net.make_encoder()
    if classifier_pth:
        net.classifier.load_state_dict(torch.load(classifier_pth))

    return net


def build_VGG_overall(num_classes, net_pth=None): 
    
    net = VGG(vgg, classifier(num_classes))

    if net_pth:
        net.load_state_dict(torch.load(net_pth))
        net.make_encoder()

    
    return net

