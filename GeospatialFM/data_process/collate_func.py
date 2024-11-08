import torch
import torchvision.transforms.functional as TF
import numpy as np

# data normalization applied to datasets directly
def apply_normalization(example, optical_mean, optical_std, radar_mean, radar_std, use_8bit=False):
    # Apply your transforms here
    optical = example.get('optical', None)
    radar = example.get('radar', None)
        
    # normalize
    def normalize(x, mean, std):
        x = x.float()
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        min_values = torch.tensor(mean) - 2 * torch.tensor(std)
        max_values = torch.tensor(mean) + 2 * torch.tensor(std)            
        
        x_normalized = (x - min_values[None, :, None, None]) / (max_values[None, :, None, None] - min_values[None, :, None, None])
        
        if use_8bit:
            x_clipped = x_normalized * 255
            x_clipped = torch.clip(x_clipped, 0, 255).to(torch.uint8)
        else:
            x_clipped = torch.clip(x_normalized, 0, 1)
            
        return x_clipped.squeeze(0)
    
    if optical is not None:
        # to tensor
        if not isinstance(optical, torch.Tensor):
            optical = torch.tensor(optical)
        # # center crop
        # optical = TF.center_crop(optical, [256, 256])
        # normalize
        optical = normalize(optical, optical_mean, optical_std)
        example['optical'] = optical.numpy()
        
    if radar is not None:
        # to tensor
        if not isinstance(radar, torch.Tensor):
            radar = torch.tensor(radar)
        # # center crop
        # radar = TF.center_crop(radar, [256, 256])
        # normalize
        radar = normalize(radar, radar_mean, radar_std)
        example['radar'] = radar.numpy()
    
    return example

# collate function for dataloader of multimodal data
def multimodal_collate_fn(batch, transform=None, random_crop=True, normalization=None):
    optical_list, radar_list = [], []
    optical_channel_wv, radar_channel_wv = None, None
    spatial_resolution = None
    
    # crop_size = np.random.choice([128, 224, 256]) if random_crop else None
    crop_size = 128

    for example in batch:
        if normalization:
            example = normalization(example)
        # to tensor
        example['optical'] = torch.tensor(example['optical'])
        example['radar'] = torch.tensor(example['radar'])
        example['optical_channel_wv'] = torch.tensor(example['optical_channel_wv']).unsqueeze(0)
        example['radar_channel_wv'] = torch.tensor(example['radar_channel_wv']).unsqueeze(0)
        example['spatial_resolution'] = example['spatial_resolution']
        
        if transform:
            example = transform(example, crop_size, scale=2)
            
        if optical_channel_wv is None and radar_channel_wv is None:
            optical_channel_wv = example['optical_channel_wv']
            radar_channel_wv = example['radar_channel_wv']
        else:
            # ensure the same optical and radar channel wv across the batch
            assert (example['optical_channel_wv'] == optical_channel_wv).all() 
            assert (example['radar_channel_wv'] == radar_channel_wv).all()

        if spatial_resolution is None:
            spatial_resolution = example['spatial_resolution']
        else:
            assert example['spatial_resolution'] == spatial_resolution
            
        optical_list.append(example['optical'])
        radar_list.append(example['radar'])
    
    assert optical_channel_wv is not None and radar_channel_wv is not None
    assert spatial_resolution is not None
    
    return {
        'optical': torch.stack(optical_list),
        'radar': torch.stack(radar_list),
        'optical_channel_wv': optical_channel_wv,
        'radar_channel_wv': radar_channel_wv,
        'spatial_resolution': spatial_resolution
    }

def modal_specific_collate_fn(batch, modal='optical'):
    data_list = {'optical': [], 'radar': []}
    channel_wv = {'optical': [], 'radar': []}
    spatial_resolution = []
    labels = []
    
    modal_list = ['optical', 'radar'] if modal == 'multi' else [modal]

    for example in batch:        
        for m in modal_list:
            assert m in example, f"{m} is not available in the example"
            example[m] = torch.tensor(example[m]) if not isinstance(example[m], torch.Tensor) else example[m]
            data_list[m].append(example[m])
            
            # example[f'{m}_channel_wv'] = torch.tensor(example[f'{m}_channel_wv']).unsqueeze(0)
            example[f'{m}_channel_wv'] = torch.tensor(example[f'{m}_channel_wv']).unsqueeze(0) \
                if not isinstance(example[f'{m}_channel_wv'], torch.Tensor) \
                else example[f'{m}_channel_wv'].clone().detach().unsqueeze(0)
            channel_wv[m].append(example[f'{m}_channel_wv'])

        spatial_resolution.append(example['spatial_resolution'])
        labels.append(example['label'])

    # at least one of the two is not None
    assert data_list['optical'] or data_list['radar']
    
    assert np.mean(spatial_resolution) == spatial_resolution[0]
    spatial_resolution = spatial_resolution[0]
    
    return_dict = {
        'spatial_resolution': np.array(spatial_resolution),
        'labels': torch.tensor(labels) if not isinstance(labels, list) else torch.tensor(np.array(labels))
    }
    
    if data_list['optical']:
        return_dict['optical'] = torch.stack(data_list['optical'])
        # assert the same channel wv across the batch
        assert (torch.stack(channel_wv['optical']) == channel_wv['optical'][0]).all()
        return_dict['optical_channel_wv'] = channel_wv['optical'][0]
    if data_list['radar']:
        return_dict['radar'] = torch.stack(data_list['radar'])
        # assert the same channel wv across the batch
        assert (torch.stack(channel_wv['radar']) == channel_wv['radar'][0]).all()
        return_dict['radar_channel_wv'] = channel_wv['radar'][0]
        
    return return_dict  

def linear_probe_collate_fn(batch):
    features = []
    labels = []
    
    for example in batch:
        features.append(example['features'])
        labels.append(example['label'])
        
    return {'features': torch.stack(features), 'labels': torch.tensor(labels)}