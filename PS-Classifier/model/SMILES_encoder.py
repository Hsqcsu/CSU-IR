import torch
from torch import nn
import numpy as np
import os
from transformers import RobertaModel
from transformers import RobertaTokenizer
from transformers import RobertaConfig


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class SmilesModel(nn.Module):
    def __init__(self,
                 roberta_model_path,
                 roberta_tokenizer_path,
                 smiles_maxlen=300,
                 vocab_size=181,
                 max_position_embeddings=505,
                 num_attention_heads=12,
                 num_hidden_layers=6,
                 type_vocab_size=1,
                 no_txtnorm=False,
                 feature_dim=768,
                 embed_size=1024,
                 **kwargs
                 ):
        super(SmilesModel, self).__init__(**kwargs)
        self.smiles_maxlen = smiles_maxlen
        self.feature_dim = feature_dim
        self.t_prime = nn.Parameter(torch.tensor(np.log(10.0)))
        self.bias = nn.Parameter(torch.tensor(-10.0))
        if roberta_tokenizer_path is not None:
            self.smiles_tokenizer = RobertaTokenizer.from_pretrained(
                roberta_tokenizer_path, max_len=self.smiles_maxlen)


            new_tokens = ["@", "/", "\\"]


            self.smiles_tokenizer.add_tokens(new_tokens)

        if roberta_model_path is None or not os.path.exists(roberta_model_path):
            self.smiles_config = RobertaConfig(
                vocab_size=vocab_size,
                max_position_embeddings=max_position_embeddings,
                num_attention_heads=num_attention_heads,
                num_hidden_layers=num_hidden_layers,
                type_vocab_size=type_vocab_size,
                hidden_size=self.feature_dim
            )
            self.smiles_model = RobertaModel(config=self.smiles_config)
        else:
            self.smiles_config = RobertaConfig.from_pretrained(
                roberta_model_path)
            self.smiles_model = RobertaModel.from_pretrained(
                roberta_model_path)
        self.smiles_model.resize_token_embeddings(len(self.smiles_tokenizer))
        self.model = self.smiles_model

        # ESA
        self.linear = nn.Linear(768, embed_size)
        self.linear1 = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(0.1)
        self.no_txtnorm = no_txtnorm

        self.init_weights()

    def encode(self, input, lengths):
        lengths = lengths -1
        input_ids, attention_mask = input
        hidden_states = self.model(input_ids, attention_mask)[0]
        hidden_states = hidden_states[:, 1:, :]
        cap_emb = self.linear(hidden_states)
        cap_emb = self.dropout(cap_emb)
        max_len = int(lengths.max())
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        cap_emb = cap_emb[:, :int(lengths.max()), :]
        features_in = self.linear1(cap_emb)
        features_in = features_in.masked_fill(mask == 0,-10000)
        features_k_softmax = nn.Softmax(dim=1)(features_in-torch.max(features_in,dim=1)[0].unsqueeze(1))
        attn = features_k_softmax.masked_fill(mask == 0,0)
        feature_cap = torch.sum(attn * cap_emb,dim=1)

        if not self.no_txtnorm:
            feature_cap = l2norm(feature_cap, dim=-1)

        return feature_cap

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

    def load_weights(self, path):
        if path is not None:
            model_dict = torch.load(path, map_location=torch.device('cpu'))
            self.load_state_dict(model_dict)

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return 'Total parameters: {}, Trainable parameters: {}'.format(total_num, trainable_num)

