import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationEncoder(torch.nn.Module):
    def __init__(self, backbone, feature_indices, feature_channels, diff=True):
        super().__init__()
        self.feature_indices = list(sorted(feature_indices))

        self._out_channels = feature_channels  
        self._in_channels = 3

        self.encoder = backbone 
        self.diff = diff # if set to True, we simply do |image1 - image2| and then feed it into encoder

    def forward(self, x1, x2):
        features = [self.concatenate(x1, x2)]
        i = 0
        for name, module in self.encoder.named_children():
            if name in ['patch_embed', 'norm', 'head']:
                x1 = module(x1)
                x2 = module(x2)
                if i in self.feature_indices:
                    features.append(self.concatenate(x1, x2))
                if i == self.feature_indices[-1]:
                    break
                i += 1 # increment i
            elif name.startswith('blocks'): # always when i = 1
                tmp = i # tmp = 1
                update = 0
                for j, block in enumerate(self.encoder.blocks):
                    x1 = block(x1)
                    x2 = block(x2)
                    if j + tmp in self.feature_indices:
                        features.append(self.concatenate(x1, x2))
                    # no need to incremenet j since it will be auto incremented
                    update = j + tmp
                # done with all blocks, update i
                i = update + 1
            else:
                raise NotImplementedError

        return features

    def concatenate(self, x1, x2):
        if self.diff:
            return torch.abs(x1 - x2)
        else:
            torch.cat([x1, x2], 1)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, patch=96):
        super(OutConv, self).__init__()
        self.patch = patch
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pooling = nn.AdaptiveAvgPool2d((self.patch, self.patch))
        self.linear = nn.Linear(self.patch ** 2, self.patch ** 2)

    def forward(self, x):
        x = self.conv(x)
        #x = self.pooling(x).view(x.size(0), -1)
        #x = self.linear(x)
        return x

class UNet(nn.Module):
    def __init__(self, feature_channels, n_classes=1, concat_mult=1, bilinear=True, dropout_rate=0.3, patch_size=96):
        super(UNet, self).__init__()
        self.patch = patch_size
        self.n_classes = n_classes # n_classes=1 in our case
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.feature_channels = feature_channels
        self.dropout = torch.nn.Dropout2d(dropout_rate)
        for i in range(0, len(feature_channels) - 1):
            in_ch = feature_channels[i + 1] * concat_mult
            setattr(self, 'shrink%d' % i,
                    nn.Conv2d(in_ch, feature_channels[i] * concat_mult, kernel_size=3, stride=1, padding=1))
            setattr(self, 'shrink2%d' % i,
                    nn.Conv2d(feature_channels[i] * concat_mult * 2, feature_channels[i] * concat_mult, kernel_size=3, stride=1, padding=1, bias=False))
            setattr(self, 'batchnorm%d' % i,
                    nn.BatchNorm2d(feature_channels[i] * concat_mult))
        self.conv = OutConv(feature_channels[0] * concat_mult, self.n_classes, self.patch)
        self.concat_mult = concat_mult
        #self.encoder = encoder

    def forward(self, features):
        #features = self.encoder(*in_x)
        features = features[1:]

        # resize the vit features
        batch_size, seq_length, embedding_dim = features[-1].shape
        height_width = int(seq_length ** 0.5)
        for i in range(len(features)):
            x = features[i].permute(0, 2, 1).view(batch_size, embedding_dim, height_width, height_width)
            x = F.interpolate(x, scale_factor=2 ** (len(features)-1-i))
            x = nn.Conv2d(
                embedding_dim, 
                self.feature_channels[i],
                kernel_size=1,
                stride=1,
                padding=0,
                device=x.device,
                dtype=x.dtype
            )(x) 
            features[i] = x
        x = features[-1]

        for i in range(len(features) - 2, -1, -1):
            conv = getattr(self, 'shrink%d' % i)
            x = F.interpolate(x, scale_factor=2)
            x = conv(x)
            if features[i].shape[-1] != x.shape[-1]:
                x2 = F.interpolate(features[i], scale_factor=2)
            else:
                x2 = features[i]
            x = torch.cat([x, x2], 1)
            conv2 = getattr(self, 'shrink2%d' % i)
            x = conv2(x)
            bn = getattr(self, 'batchnorm%d' % i)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        logits = self.conv(x)
        logits = nn.AdaptiveAvgPool2d((self.patch, self.patch))(logits).view(x.size(0), -1)
        logits = nn.Linear(self.patch ** 2, self.patch ** 2, device=logits.device)(logits)
        return logits