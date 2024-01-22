from commons import train_regression
from unimodals.common_models import Transformer, MLP

if __name__ == '__main__':
    # mosi/mosei
    audio_features_dim = 74
    tr_dim = 370
    mlp_hid_dim = 128

    # humor/sarcasm
    # audio_features_dim = 81
    # tr_dim = 405
    # mlp_hid_dim = 128

    encoder = Transformer(n_features=audio_features_dim, dim=tr_dim).cuda()
    head = MLP(indim=tr_dim, hiddim=mlp_hid_dim, outdim=1, dropoutp=0.1, dropout=True).cuda()

    train_regression('mosi', 'tr_r', encoder, head, epochs=400, early_stop=70)
