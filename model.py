import torch
import torch.nn as nn

def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class CustomPetNet(nn.Module):
    """
    My custom ResNet-9 setup. Needed something lightweight that converges super fast 
    so I don't fail the 30-epoch constraint.
    """
    def __init__(self, num_classes=37):
        super(CustomPetNet, self).__init__()
        
        # prep layer - added pooling here to make it run way faster on my machine
        self.prep = conv_block(3, 64, pool=True)
        
        # first block + skip connection
        self.layer1 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(
            conv_block(128, 128),
            conv_block(128, 128)
        )
        
        # second block (skipped the res connection here to save on params)
        self.layer2 = conv_block(128, 256, pool=True)
        
        # third block + skip connection
        self.layer3 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(
            conv_block(512, 512),
            conv_block(512, 512)
        )
        
        # final classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.prep(x)
        
        out = self.layer1(out)
        out = self.res1(out) + out
        
        out = self.layer2(out)
        
        out = self.layer3(out)
        out = self.res2(out) + out
        
        out = self.classifier(out)
        return out

if __name__ == "__main__":
    model = CustomPetNet(num_classes=37)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total weights/biases: {total_params}")
