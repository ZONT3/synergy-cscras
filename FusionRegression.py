import os
import sys

import torch

from commons import get_data_path, get_ckp_name

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from training_structures.Supervised_Learning import train, test
from unimodals.common_models import Transformer, MLP
from datasets.affect.get_data import get_dataloader
from fusions.common_fusions import Concat

if __name__ == '__main__':
    dataset = 'mosei'
    traindata, validdata, testdata = get_dataloader(get_data_path(dataset), robust_test=False)

    # mosi/mosei
    encoders = [Transformer(35, 70).cuda(),
                Transformer(74, 200).cuda(),
                Transformer(300, 600).cuda()]
    fusion = Concat().cuda()
    head = MLP(870, 256, 1).cuda()

    # humor/sarcasm
    # encoders=[Transformer(371,400).cuda(), \
    #     Transformer(81,100).cuda(),\
    #     Transformer(300,600).cuda()]
    # head=MLP(1100,256,1).cuda()

    ckp_name = get_ckp_name(dataset, 'lf')
    train(encoders, fusion, head, traindata, validdata, 100, task='regression', optimtype=torch.optim.AdamW,
          early_stop=False, is_packed=True, lr=1e-4, save=ckp_name,
          weight_decay=0.01, objective=torch.nn.L1Loss())

    print('Testing:')
    model = torch.load(ckp_name).cuda()

    test(model=model, test_dataloaders_all=testdata, dataset=dataset, is_packed=True,
         criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=True)
