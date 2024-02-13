import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(PyramidPoolingModule, self).__init__()
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(output_size=size) for size in pool_sizes])
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, in_channels, kernel_size=1) for _ in pool_sizes])

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        pooled_features = [x]
        for pool, conv in zip(self.pools, self.convs):
            p = pool(x)
            p = conv(p)
            p = F.relu(p, inplace=True)
            p = F.interpolate(p, size=(h, w), mode='bilinear', align_corners=False)
            pooled_features.append(p)
        return torch.cat(pooled_features, dim=1).contiguous()


class PSPNetDecoder(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dim=512, img_size=224):
        super(PSPNetDecoder, self).__init__()
        self.pyramid_pooling = PyramidPoolingModule(in_features, [1, 2, 3, 6])
        self.decoder = nn.Sequential(
            nn.Conv2d(in_features * 5, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        )
        self.img_size = img_size

    def forward(self, x):
        assert len(x.shape) == 3, 'Input shape must be (B, N, D)'
        # Convert patch embeddings back to feature map
        B, N, D = x.shape
        num_patches_side = int(N ** 0.5)
        H, W = num_patches_side, num_patches_side
        assert H * W == N, 'Number of patches must be a square number'
        vit_features = x.permute(0, 2, 1).reshape(B, D, H, W).contiguous()
        
        # Pyramid Pooling Module
        ppm_features = self.pyramid_pooling(vit_features)
        
        # Decoder
        out = self.decoder(ppm_features)
        
        # Upsample to original image size
        out = F.interpolate(out, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        return out


class CDEncoder(torch.nn.Module):
    def __init__(self, encoder, diff=True, use_mlp=False):
        super().__init__()

        self.encoder = encoder 
        self.diff = diff # if set to True, we simply do |image1 - image2| and then feed it into encoder
        self.use_mlp = use_mlp
        self.mlp = None

    def forward(self, x1, x2):
        assert x1.shape == x2.shape, 'Input images must have the same shape'
        B = x1.shape[0]
        x_all = torch.cat([x1, x2], dim=0)
        model_out = self.encoder(x_all, return_dict=True)
        patch_tokens = model_out['patch_tokens']
        out_1, out_2 = patch_tokens.split(B, dim=0)
        merged_out = self.concatenate(out_1, out_2)
        if self.use_mlp:
            raise NotImplementedError
        return merged_out

    def concatenate(self, x1, x2):
        if self.diff:
            return torch.abs(x1 - x2)
        else:
            return torch.cat([x1, x2], 1)
