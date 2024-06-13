import torch
import torch.nn as nn

def bn_rl_conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    layers = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    )
    return layers

def dense_block(x, num_layers, growth_rate):
    for _ in range(num_layers):
        layers = nn.Sequential(
            bn_rl_conv(x.size(1), 4 * growth_rate, kernel_size=1),
            bn_rl_conv(4 * growth_rate, growth_rate, kernel_size=3, padding=1)
        )
        x = torch.cat([x, layers.to(x.device)(x)], 1)
    return x

def transition_block(x):
    in_channels = x.size(1)
    layers = nn.Sequential(
        bn_rl_conv(in_channels, in_channels // 2, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )
    return layers.to(x.device)(x)

def densenet(img_shape, n_classes, growth_rate=32):
    repetitions = [6, 12, 24, 16]
    
    input_tensor = torch.randn(1, *img_shape).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    layers = nn.Sequential(
        nn.Conv2d(img_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    ).to(input_tensor.device)
    
    x = layers(input_tensor)
    
    for r in repetitions:
        x = dense_block(x, r, growth_rate)
        x = transition_block(x)
    
    x = nn.AdaptiveAvgPool2d((7, 7)).to(x.device)(x)
    x = torch.flatten(x, 1)
    
    output = nn.Linear(x.size(1), n_classes).to(x.device)(x)
    
    return output

img_shape = (3, 224, 224)
n_classes = 1000
output = densenet(img_shape, n_classes)
print(output)
