import os
import torch
import torchgeo.models as tgm
    
def torch_save(classifier, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # with open(save_path, 'wb') as f:
    #     pickle.dump(classifier.cpu(), f)
    torch.save(classifier, save_path,)

def torch_load(save_path, device=None):
    # with open(save_path, 'rb') as f:
    #     classifier = pickle.load(f)
    classifier = torch.load(save_path)
    if device is not None:
        classifier = classifier.to(device)
    return classifier

def get_tgm_model(model_cfg):
    if model_cfg['load_encoder'] is not None:
        weights = tgm.get_weight(model_cfg['load_encoder'])
    else:
        weights = None
    return tgm.get_model(model_cfg['name'], weights=weights)