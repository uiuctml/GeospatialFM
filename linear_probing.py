from GeospatialFM.utils import setup, get_args_parser
from GeospatialFM.data import *
from GeospatialFM.models import *

import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def extract_features(model, dataloader, device):
    x_all = []
    y_all = []

    model.eval()
    for batch in tqdm(dataloader):
        images = batch["image"].to(device)
        labels = batch["label"].numpy()
        
        with torch.no_grad():
            features = model(images).cpu().numpy()
        
        x_all.append(features)
        y_all.append(labels)

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    return x_all, y_all

if __name__ == '__main__':
    cache_feat = True

    args = get_args_parser().parse_args()
    cfg, _ = setup(args, wandb=False)
    cache_dir = os.path.join(cfg['LOGGER']['dir'], 'cache', cfg['DATASET']['name'])
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cfg['MODEL']['architecture'].replace('/', '-')}_{cfg['MODEL']['bands']}_{cfg['MODEL']['pretrained_ckpt'].split('.')[-1]}")
    print(f'cache path: {cache_path}')
    model = construct_model(cfg['MODEL'])

    train_ds, val_ds, test_ds = get_datasets(cfg['DATASET'])
    encoder = model.base_model.to('cuda')

    train_dl = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=8)
    test_dl = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=8)

    if os.path.exists(cache_path):
        print(f"Loading features from cache")
        train_features = np.load(os.path.join(cache_path, 'train_features.npy'))
        train_labels = np.load(os.path.join(cache_path, 'train_labels.npy'))
        test_features = np.load(os.path.join(cache_path, 'test_features.npy'))
        test_labels = np.load(os.path.join(cache_path, 'test_labels.npy'))
    else:
        train_features, train_labels = extract_features(encoder, train_dl, 'cuda')
        test_features, test_labels = extract_features(encoder, test_dl, 'cuda')
        if cache_feat:
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
            np.save(os.path.join(cache_path, 'train_features.npy'), train_features)
            np.save(os.path.join(cache_path, 'train_labels.npy'), train_labels)
            np.save(os.path.join(cache_path, 'test_features.npy'), test_features)
            np.save(os.path.join(cache_path, 'test_labels.npy'), test_labels)
    
    train_features = torch.from_numpy(train_features)
    train_labels = torch.from_numpy(train_labels)
    test_features = torch.from_numpy(test_features)

    encoder.cpu()

    task_head = model.task_head.to('cuda')

    # simple training loop for the task head
    # epoches = cfg['TRAINER']['num_train_epochs']
    epoches = 10000
    # lr = cfg['TRAINER']['learning_rate']
    lr = 0.1
    optimizer = AdamW(task_head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches, eta_min=1e-6)
    pbar = tqdm(range(epoches))
    for e in pbar:
        logits = task_head(train_features.to('cuda'))
        loss = criterion(logits, train_labels.to('cuda'))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_description(f"Epoch {e+1} Loss: {loss.item():.4f}")
    # evaluate the task head
    task_head.eval()
    with torch.no_grad():
        logits = task_head(test_features.to('cuda'))
    preds = logits.argmax(dim=1).cpu().numpy()
    acc = (preds == test_labels).mean()
    print(f"{cfg['NAME']} Test Accuracy: {acc:.4f}")
