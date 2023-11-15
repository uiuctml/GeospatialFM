import argparse

from GeospatialFM.data import get_datasets
from GeospatialFM.models import *

from GeospatialFM.utils import *
from GeospatialFM.data import *
from GeospatialFM.models import *
from GeospatialFM.loss import *

from tqdm import trange, tqdm

from collections import OrderedDict

def unwrap_model(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    
    return new_state_dict

def extract_features(model, dataloader, device):
    split_dict = dict(radar_feature=[], optical_feature=[], label=[])
    for i, batch in enumerate(tqdm(dataloader)):
        images, radar, label = batch['image'], batch['radar'], batch['label']
        images = images.to(device=device, non_blocking=True)
        radar = radar.to(device=device, non_blocking=True)

        with torch.no_grad():
            optical_features = model.optical_encoder(images, return_dict=True)
            radar_features = model.radar_encoder(radar, return_dict=True)
            split_dict['radar_feature'].append(radar_features['cls_token'].detach().cpu().numpy())
            split_dict['optical_feature'].append(optical_features['cls_token'].detach().cpu().numpy())
            split_dict['label'].append(label.numpy())
    split_dict['radar_feature'] = np.concatenate(split_dict['radar_feature'], axis=0)
    split_dict['optical_feature'] = np.concatenate(split_dict['optical_feature'], axis=0)
    split_dict['label'] = np.concatenate(split_dict['label'], axis=0)
    return split_dict

if __name__ == '__main__':
    cache_feat = True

    args = get_args_parser().parse_args()
    args.debug = True
    if args.debug:
        logging.basicConfig(level=logging.INFO)
    cfg, _ = setup(args)

    training_args = dict(
        device_ids = args.device_ids,
        device = 'cpu',
        precision = None,
        accum_freq = cfg['TRAINER']['gradient_accumulation_steps'],
        grad_clip_norm = None,
        log_every_n_steps = cfg['TRAINER']['logging_steps'],
        wandb = cfg['TRAINER']['report_to'] == 'wandb',
        batch_size = cfg['TRAINER']['per_device_train_batch_size'],
        val_frequency = 1,
        epochs = cfg['TRAINER']['num_train_epochs'],
        save_logs = True,
        checkpoint_path = cfg['TRAINER']['logging_dir'],
        mask_ratio = cfg['MODEL']['mask_ratio']
    )
    training_args = argparse.Namespace(**training_args)
    training_args.device = f'cuda:{training_args.device_ids[0]}'

    save_path = os.path.join(cfg.TRAINER['output_dir'], 'final_model.pth')

    model = construct_mae(cfg.MODEL)
    state_dict = unwrap_model(torch.load(save_path))
    model.load_state_dict(state_dict)
    model = model.to(training_args.device)
    if hasattr(model, 'optical_decoder'):
        del model.optical_decoder
    if hasattr(model, 'radar_decoder'):
        del model.radar_decoder
    if hasattr(model, 'decoder'):
        del model.decoder

    data = get_data(cfg)

    cache_dir = os.path.join(cfg['LOGGER']['dir'], 'cache', cfg['DATASET']['name'])
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cfg['NAME'])
    print(f'cache path: {cache_path}')

    splits = ['train', 'test', 'val']
    data_dict = dict()


    for split in splits:
        cache_data_path = os.path.join(cache_path, f'{split}_data.npy')
        if os.path.exists(cache_data_path):
            print(f"Loading features from cache")
            for split in splits:
                data_dict[split] = np.load(cache_data_path)
        else:
            for split in splits:
                data_dict[split] = extract_features(model, data[split].dataloader, training_args.device)
                if cache_feat:
                    if not os.path.exists(cache_path):
                        os.makedirs(cache_path)
                    np.save(cache_data_path, data_dict[split])
              
    # train_features = torch.from_numpy(train_data)
    # test_features = torch.from_numpy(test_data)
    # val_features = torch.from_numpy(val_data)

    # model.cpu()

    # task_head = model.task_head.to('cuda')

    # # simple training loop for the task head
    # # epoches = cfg['TRAINER']['num_train_epochs']
    # epoches = 10000
    # # lr = cfg['TRAINER']['learning_rate']
    # lr = 0.1
    # optimizer = AdamW(task_head.parameters(), lr=lr)
    # criterion = nn.CrossEntropyLoss()
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches, eta_min=1e-6)
    # pbar = tqdm(range(epoches))
    # for e in pbar:
    #     logits = task_head(train_features.to('cuda'))
    #     loss = criterion(logits, train_labels.to('cuda'))
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     scheduler.step()
    #     pbar.set_description(f"Epoch {e+1} Loss: {loss.item():.4f}")
    # # evaluate the task head
    # task_head.eval()
    # with torch.no_grad():
    #     logits = task_head(test_features.to('cuda'))
    # preds = logits.argmax(dim=1).cpu().numpy()
    # acc = (preds == test_labels).mean()
    # print(f"{cfg['NAME']} Test Accuracy: {acc:.4f}")
