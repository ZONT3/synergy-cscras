from commons import train_regression
from unimodals.common_models import MLP, GRU

if __name__ == '__main__':
    # mosi/mosei
    # audio_features_dim = 74
    # gru_hidden_dim = 200
    # mlp_hid_dim = 100

    # humor/sarcasm
    audio_features_dim = 81
    gru_hidden_dim = 240
    mlp_hid_dim = 120

    encoder = GRU(audio_features_dim, gru_hidden_dim, dropout=True, has_padding=False,
                  batch_first=True, last_only=True, dropoutp=0.1).cuda()
    head = MLP(gru_hidden_dim, mlp_hid_dim, 1).cuda()

    train_regression('sarcasm', 'gru_r', encoder, head, epochs=400, early_stop=40)
