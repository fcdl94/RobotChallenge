import torch.nn as nn
import torch


class RGBDNet(nn.Module):
    def __init__(self, rgb_network, depth_network, classes):
        super(RGBDNet, self).__init__()
        self.net1 = rgb_network
        self.net2 = depth_network
        self.fc = nn.Linear(classes*2, classes)
        
        self.classes = classes
        
    def forward(self, x):
        
        rgb, depth = x
        rgb = self.net1(rgb)
        depth = self.net2(depth)
        
        out = torch.stack((rgb, depth), 1).view(rgb.shape[0], -1)
        
        out = self.fc(out)
        
        return out
    
    
def double_resnet18(classes):
    from OBC.networks import resnet18
    rgb_net = resnet18(classes)
    depth_net = resnet18(classes)
    net = RGBDNet(rgb_net, depth_net, classes)
    return net
