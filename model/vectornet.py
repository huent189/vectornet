import torch
import torch.nn as nn
class PathNet(nn.Module):
    """
    VDSR
    """
    def __init__(self, repeat_num, hidden_channel, last_activation='relu', k=3, s=1, p=1):
        super(PathNet, self).__init__()
        first_layer = [nn.Conv2d(2, hidden_channel, kernel_size=k, stride=s, padding=p),
                 nn.ReLU(),
                 nn.BatchNorm2d(hidden_channel)]
        if repeat_num > 2:
            mid_layer = [nn.Sequential([nn.Conv2d(hidden_channel, hidden_channel, kernel_size=k, stride=s, padding=p),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(hidden_channel)]) for _ in range((repeat_num - 2))]
        if last_activation == 'relu':
            last_layer = [nn.Conv2d(hidden_channel, 1, kernel_size=k, stride=s, padding=p),
                        nn.ReLU()]
        else:
            last_layer = [nn.Conv2d(hidden_channel, 1, kernel_size=k, stride=s, padding=p),
                 nn.Sigmoid()]
        if repeat_num > 2:
            layers = first_layer + mid_layer + last_layer
        else:
            layers = first_layer + last_layer
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)
