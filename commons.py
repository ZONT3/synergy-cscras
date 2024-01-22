import sys
from pathlib import Path

import torch
import torch.nn
import torch.optim

from custom_data import get_dataloader as get_custom_dataloader
from datasets.affect.get_data import get_dataloader as get_mb_dataloader
from training_structures.unimodal import train, test

DATA_FILE_MAP = {
    'mosi': 'mosi_raw.pkl',
    'mosei': 'mosei_raw.pkl',
    'sarcasm': 'sarcasm.pkl',
    'humor': 'humor.pkl',
}

DATA_PATH = Path('data')
CKP_PATH = Path('checkpoints')


# audio_feat_dim
# mosei 73
# mosi 73
# sarcasm 81
# humor 81


def get_dataset_from_argv():
    dataset = 'mosei' if len(sys.argv) < 2 else sys.argv[1]
    print(f'Using {dataset.upper()} dataset')
    return dataset


def get_data_path(dataset):
    return str(DATA_PATH / DATA_FILE_MAP[dataset])


def get_ckp_name(dataset, identification, head=False, encoder=False, suffix="best"):
    if head:
        suffix = 'H_' + suffix
    elif encoder:
        suffix = 'E_' + suffix
    return str(CKP_PATH / f'{dataset}_{identification}_{suffix}.pt')


def get_dataloader(dataset, task='regression', max_seq_len=50, max_pad=True, num_workers=1, balanced=False):
    if dataset in DATA_FILE_MAP.keys():
        return get_mb_dataloader(get_data_path(dataset),
                                 robust_test=False, max_pad=max_pad, data_type=dataset, max_seq_len=max_seq_len,
                                 task=task, num_workers=num_workers, mosei_classes=[3])
    else:
        return get_custom_dataloader(num_workers=num_workers, balanced=balanced)


def train_regression(dataset, identification, encoder, head, traindata=None, validdata=None, epochs=100, early_stop=12,
                     modalnum=1, criterion=torch.nn.L1Loss(), max_seq_len=50, max_pad=True,
                     task="regression", track_complexity=True, lr=6e-5, optimtype=torch.optim.AdamW, weight_decay=0.01,
                     num_workers=1):
    if traindata is None or validdata is None:
        traindata, validdata, testdata = get_dataloader(dataset, task, max_seq_len=max_seq_len, max_pad=max_pad,
                                                        num_workers=num_workers)
    else:
        testdata = None

    save_encoder = get_ckp_name(dataset, identification, encoder=True)
    save_head = get_ckp_name(dataset, identification, head=True)

    train(encoder, head, traindata, validdata, epochs, early_stop=early_stop, modalnum=modalnum, task=task,
          optimtype=optimtype, lr=lr, track_complexity=track_complexity,
          weight_decay=weight_decay, criterion=criterion,
          save_encoder=save_encoder,
          save_head=save_head)

    return save_encoder, save_head, testdata


def train_classification(encoder, head, dataset, method_name, num_workers=1):
    train_data, val_data, test_data = get_dataloader(dataset, 'classification', max_seq_len=50, max_pad=True,
                                                     num_workers=num_workers)

    encoder_ckp = get_ckp_name(dataset, method_name, encoder=True)
    head_ckp = get_ckp_name(dataset, method_name, head=True)
    train(encoder, head, train_data, val_data, 200, early_stop=20, modalnum=1, task='multilabel',
          optimtype=torch.optim.AdamW, lr=2e-5, track_complexity=True,
          weight_decay=0.01, criterion=torch.nn.BCEWithLogitsLoss(),
          save_encoder=encoder_ckp, save_head=head_ckp)

    print('Testing:')
    encoder_model = torch.load(encoder_ckp).cuda()
    head_model = torch.load(head_ckp).cuda()
    return test(encoder_model, head_model, test_dataloaders_all=test_data,
                dataset=dataset, method_name=method_name, modalnum=1,
                task='multilabel', no_robust=True)
