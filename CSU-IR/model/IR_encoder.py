import torch
import torch.nn as nn
import numpy as np


class MlpBlock(nn.Module):
    def __init__(self, channels):
        super(MlpBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, input):
        return input + self.block(input)


class CnnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CnnBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

    def forward(self, input):
        return self.layer(input)


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class IRModel(nn.Module):
    def __init__(self, output_channels=768, channels=32, dim=1024, no_txtnorm=False):
        super(IRModel, self).__init__()
        # CCTE
        self.cnn_layers = nn.Sequential(
            CnnBlock(1, channels, kernel_size=3, stride=1, padding=1),
            CnnBlock(channels, channels * 2, kernel_size=3, stride=2, padding=1),
            CnnBlock(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1),
            CnnBlock(channels * 4, channels * 8, kernel_size=3, stride=2, padding=1),
            CnnBlock(channels * 8, channels * 16, kernel_size=3, stride=2, padding=1),
        )
        self.linear_2_768 = nn.Sequential(nn.Linear(512, 768), nn.GELU())
        self.positional_encoding = PositionalEncoding(768)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(768)

        # ESA
        self.linear = nn.Linear(output_channels, dim)
        self.linear1 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)
        self.no_txtnorm = no_txtnorm
        self.init_weights()

    def forward(self, input):
        x_ir = self.cnn_layers(input.unsqueeze(1))

        x_ir = x_ir.transpose(1, 2)
        x_ir = self.linear_2_768(x_ir)
        x_ir = self.positional_encoding(x_ir)
        x_ir = self.transformer_encoder(x_ir)
        x_ir = self.norm(x_ir)

        cap_emb = self.linear(x_ir)
        cap_emb = self.dropout(cap_emb)
        features_in = self.linear1(cap_emb)
        features_k_softmax = nn.Softmax(dim=1)(features_in - torch.max(features_in, dim=1)[0].unsqueeze(1))
        feature_cap = torch.sum(features_k_softmax * cap_emb, dim=1)

        if not self.no_txtnorm:
            feature_cap = l2norm(feature_cap, dim=-1)


        return feature_cap

    def load_weights(self, path):
        if path is not None:
            model_dict = torch.load(path, map_location=torch.device('cpu'))
            self.load_state_dict(model_dict)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
