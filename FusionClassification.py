import torch

from commons import get_data_path, get_ckp_name, get_dataset_from_argv
from datasets.affect.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.Supervised_Learning import train, test
from unimodals.common_models import Transformer, MLP


def main():
    dataset = get_dataset_from_argv()

    if dataset in ['mosi', 'mosei']:
        # mosi/mosei
        is_mosi = dataset == 'mosi'
        encoders = [
            Transformer(35, 70).cuda()
            if is_mosi else
            Transformer(713, 715).cuda(),
            Transformer(74, 200).cuda(),
            Transformer(300, 600).cuda()
        ]
        head = MLP(870 if is_mosi else 1515, 256, 1, dropout=True).cuda()

    elif dataset in ['humor', 'sarcasm']:
        # humor/sarcasm
        encoders = [Transformer(371, 400).cuda(),
                    Transformer(81, 200).cuda(),
                    Transformer(300, 600).cuda()]
        head = MLP(1200, 256, 1).cuda()

    else:
        raise NotImplementedError()

    fusion = Concat().cuda()

    traindata, validdata, testdata = get_dataloader(get_data_path(dataset), robust_test=False, task='classification',
                                                    data_type=dataset, mosei_classes=[3], num_workers=1,
                                                    batch_size=32)

    ckp_name = get_ckp_name(dataset, 'lf_c')
    train(encoders, fusion, head, traindata, validdata, 200, task='multilabel', optimtype=torch.optim.AdamW,
          early_stop=70, is_packed=True, lr=2e-5, save=ckp_name, clip_val=1,
          weight_decay=0.01, objective=torch.nn.BCEWithLogitsLoss())

    print('Testing:')
    model = torch.load(ckp_name).cuda()

    test(model=model, test_dataloaders_all=testdata, dataset=dataset, is_packed=True,
         criterion=torch.nn.BCEWithLogitsLoss(), task='multilabel', no_robust=True)


if __name__ == '__main__':
    main()
