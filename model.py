import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch_model_summary import summary


class Model(nn.Module):
    def __init__(self):
        super().__init__()


class CharModel(nn.Module):

    def __init__(self, n_chars, padding_idx, emb_dim=30, dropout_p=0.5, embed_chars=True):
        super(CharModel, self).__init__()

        self.input_dim = n_chars
        self.dropout_p = dropout_p
        self.padding_idx = padding_idx
        self.emb_dim = emb_dim
        self.embed_chars = embed_chars

        if embed_chars:
            self.embeddings = nn.Embedding(n_chars, emb_dim, padding_idx=padding_idx)
        self.char_emb_dropout = nn.Dropout(p=dropout_p)

    def forward(self, sentence: Variable) -> torch.Tensor:
        # embed characters
        if self.embed_chars:
            embedded = self.embeddings(sentence)
            embedded = self.char_emb_dropout(embedded)
        else:
            embedded = sentence

        # character model
        output = self.char_model(embedded)
        return output


class CharCNN(CharModel):

    def __init__(self, n_chars, padding_idx, emb_dim, dropout_p, n_classes, max_seq_length):
        super(CharCNN, self).__init__(n_chars, padding_idx, emb_dim=emb_dim,
                                      dropout_p=dropout_p, embed_chars=True)

        self.n_chars = n_chars

        # in_channels, out_channels, kernel_size, stride, padding
        conv_stride = 1
        max_pool_kernel_size = 3
        max_pool_stride = 3
        padding = 0
        conv_spec_1 = dict(in_channels=emb_dim, out_channels=256, kernel_size=7, padding=0)
        conv_spec_2 = dict(in_channels=256, out_channels=256, kernel_size=7, padding=0)
        conv_spec_3 = dict(in_channels=256, out_channels=256, kernel_size=3, padding=0)
        conv_spec_4 = dict(in_channels=256, out_channels=256, kernel_size=3, padding=0)
        conv_spec_5 = dict(in_channels=256, out_channels=256, kernel_size=3, padding=0)
        conv_spec_6 = dict(in_channels=256, out_channels=256, kernel_size=3, padding=0)
        network = [conv_spec_1, 'MaxPool', conv_spec_2, 'MaxPool', conv_spec_3,
                   conv_spec_4, conv_spec_5, conv_spec_6, 'MaxPool']

        layers = []
        for layer in network:
            if layer == 'MaxPool':
                layers.append(nn.MaxPool1d(kernel_size=max_pool_kernel_size, stride=max_pool_stride, padding=padding))
            else:
                conv = nn.Conv1d(layer['in_channels'], layer['out_channels'],
                                 kernel_size=layer['kernel_size'], stride=conv_stride, padding=layer['padding'])
                relu = nn.ReLU(inplace=True)
                layers.extend([conv, relu])

        self.layers = nn.Sequential(*layers)
        self.fc1 = nn.Linear(int((max_seq_length - 96)/27) * 256, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, n_classes)
        self.xavier_uniform()

    def gaussian_init(self, mean=0., std=0.05):
        for name, weight in self.named_parameters():
            if len(weight.size()) > 1:
                nn.init.normal_(weight.data, mean=mean, std=std)
            elif "bias" in name:
                weight.data.fill_(0.)

    def xavier_uniform(self, gain=1.):
        # default pytorch initialization
        for name, weight in self.named_parameters():
          if len(weight.size()) > 1:
              nn.init.xavier_uniform_(weight.data, gain=gain)
          elif "bias" in name:
            weight.data.fill_(0.)

    def char_model(self, embedded=None):

        embedded = torch.transpose(embedded, 1, 2)  # (bsz, dim, time)

        # conv net
        bsz = embedded.shape[0]
        chars_conv = self.layers(embedded)

        # print(chars_conv.shape)
        # fully connected layers
        output = self.fc1(chars_conv.view(bsz, -1))
        output = self.char_emb_dropout(output)
        output = self.fc2(output)

        # dropout and classify
        output = self.char_emb_dropout(output)
        labels = self.classifier(output)

        # softmax
        log_probs = F.log_softmax(labels, 1)
        return log_probs


if __name__ == '__main__':
    # test model
    model = CharCNN(n_chars=6010, padding_idx=1, emb_dim=100, dropout_p=0, n_classes=235, max_seq_length=250)
    print(summary(model, torch.ones((128, 251)).to(dtype=torch.int), show_input=False, show_hierarchical=False))