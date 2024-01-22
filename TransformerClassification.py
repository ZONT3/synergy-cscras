from commons import train_classification, get_dataset_from_argv
from unimodals.common_models import Transformer, MLP


def main():
    dataset = get_dataset_from_argv()

    audio_features_dim = 74
    tr_dim = 200
    mlp_hid_dim = 128

    encoder = Transformer(n_features=audio_features_dim, dim=tr_dim).cuda()
    head = MLP(indim=tr_dim, hiddim=mlp_hid_dim, outdim=1, dropoutp=0.1, dropout=True).cuda()

    train_classification(encoder, head, dataset, 'tr_c', num_workers=1)


if __name__ == '__main__':
    main()
