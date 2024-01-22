import _pickle as pkl

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from torch.utils.data.dataloader import DataLoader


class Emo2Dataset(Dataset):
    def __init__(self, data_path, test=False, balanced_test=False):
        with open(data_path, 'rb') as fd:
            data_dict: dict = pkl.load(fd)

        data = data_dict['data']
        labels = data_dict['labels']

        zero_indices = np.where(labels == 0)[0]
        self.pos_classes = len(labels) - len(zero_indices)
        self.train_val_len = self.pos_classes + self.pos_classes * 3
        self.test_len = self.pos_classes
        self.test = test
        self.balanced_test = balanced_test

        self.data = torch.tensor(data)
        self.labels = torch.tensor(labels.reshape((-1, 1)))

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return [torch.zeros_like(data), data, torch.zeros_like(data), idx, label]

    def __len__(self):
        return self.train_val_len if not self.test else self.test_len if self.balanced_test else self.labels.shape[0]


def get_dataloader(data_type='emo2', data_path='data/emo2.pkl', num_workers=4, batch_size=32, balanced=False):
    if data_type == 'emo2':
        data_class = Emo2Dataset
    else:
        raise NotImplementedError()

    dataset = data_class(data_path, test=False)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
    test_dataset = data_class(data_path, test=True, balanced_test=balanced)

    print('Train size:', len(train_dataset))
    print('  Val size:', len(val_dataset))
    print(' Test size:', len(test_dataset))
    print(' Total pos:', test_dataset.pos_classes)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )
