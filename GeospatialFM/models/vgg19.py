import torch
import torch.nn as nn

class VGG19(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True, input_channels=1, input_size=256):
        super(VGG19, self).__init__()
        
        # Calculate the size of the final feature map
        # 5 maxpool layers with stride 2: input_size / 2^5
        final_size = input_size // 32
        
        # Feature layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * final_size * final_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class MultiChannelVGG19(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True, input_size=256):
        super(MultiChannelVGG19, self).__init__()
        
        # Single VGG19 model for processing any channel
        self.vgg = VGG19(num_classes=num_classes, init_weights=init_weights, 
                         input_channels=1, input_size=input_size)
        
        # Enhanced channel attention mechanism
        self.channel_attention = nn.Sequential(
            nn.Linear(num_classes, num_classes // 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_classes // 2, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, optical=None, radar=None, optical_channel_wv=None, radar_channel_wv=None, spatial_resolution=10, labels=None):
        assert optical is not None or radar is not None
        x = []
        if optical is not None:
            x.append(optical)
        if radar is not None:
            x.append(radar)
        x = torch.cat(x, dim=1)

        # x shape: [batch_size, num_channels, height, width]
        batch_size, num_channels = x.size(0), x.size(1)
        
        # Reshape to process all channels in parallel
        # [batch_size, num_channels, height, width] -> [batch_size * num_channels, 1, height, width]
        x_reshaped = x.reshape(-1, 1, x.size(2), x.size(3))
        
        # Process all channels in parallel
        # Shape: [batch_size * num_channels, num_classes]
        channel_outputs = self.vgg(x_reshaped)
        
        # Reshape back to separate batch and channel dimensions
        # Shape: [batch_size, num_channels, num_classes]
        channel_outputs = channel_outputs.reshape(batch_size, num_channels, -1)
        
        # Apply channel attention
        # Shape: [batch_size, num_channels, num_classes]
        attention_weights = self.channel_attention(channel_outputs)
        weighted_outputs = channel_outputs * attention_weights
        
        # Average across channels
        # Shape: [batch_size, num_classes]
        final_output = weighted_outputs.mean(dim=1)
        
        return {"logits": final_output}

def vgg19(pretrained=False, num_classes=1000, input_channels=1, input_size=256):
    """
    VGG 19-layer model (configuration 'E')
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): Number of classes for the classifier layer
        input_channels (int): Number of input channels (default: 1 for single channel)
        input_size (int): Input image size (default: 256)
    """
    model = VGG19(num_classes=num_classes, input_channels=input_channels, input_size=input_size)
    if pretrained:
        # Note: You would need to implement the loading of pretrained weights here
        pass
    return model

def multi_channel_vgg19(num_classes=1000, init_weights=True, input_size=256):
    """
    Multi-channel VGG19 model that processes each channel separately and fuses the results
    using channel attention
    
    Args:
        num_classes (int): Number of classes for the classifier layer
        init_weights (bool): Whether to initialize weights
        input_size (int): Input image size (default: 256)
    """
    return MultiChannelVGG19(num_classes=num_classes, init_weights=init_weights, input_size=input_size) 