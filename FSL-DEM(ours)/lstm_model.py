import torch
import torch.nn as nn

class lstm_400Hz(nn.Module):
    def __init__(self):
        super(lstm_400Hz, self).__init__()
        self.layer0 = nn.LSTM(input_size=1, hidden_size=24, num_layers=4, batch_first=True)
        self.layer1 = nn.Linear(24, 1)
        self.layer2 = nn.Sequential(
            nn.Linear(400, 1000),
            nn.GELU(),
            nn.Linear(1000, 400),
            nn.GELU(),
            nn.Linear(400, 100),
        )


    def forward(self, x):
        batch_size, input_size = x.size(0), x.size(1)
        y = x.reshape(batch_size, 1, input_size)

        y = self.layer0(y.permute([0, 2, 1]))[0]
        y = y.contiguous().view(-1, y.shape[2])
        y = self.layer1(y)
        y = y.contiguous().view(batch_size, -1)

        y = y + x
        y = self.layer2(y)
        y = torch.unsqueeze(y, 1)
        return y

class lstm_200Hz(nn.Module):
    def __init__(self):
        super(lstm_200Hz, self).__init__()
        self.layer0 = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.layer1 = nn.Linear(64, 1)
        self.layer2 = nn.Sequential(
            nn.Linear(200, 1000),
            nn.GELU(),
            nn.Linear(1000, 200),
            nn.GELU(),
            nn.Linear(200, 100),
        )


    def forward(self, x):
        batch_size, input_size = x.size(0), x.size(1)
        y = x.reshape(batch_size, 1, input_size)

        y = self.layer0(y.permute([0, 2, 1]))[0]
        y = y.contiguous().view(-1, y.shape[2])
        y = self.layer1(y)
        y = y.contiguous().view(batch_size, -1)

        y = y + x
        y = self.layer2(y)
        y = torch.unsqueeze(y, 1)
        return y

class lstm_100Hz(nn.Module):
    def __init__(self):
        super(lstm_100Hz, self).__init__()
        self.layer0 = nn.LSTM(input_size=1, hidden_size=24, num_layers=4, batch_first=True)
        self.layer1 = nn.Linear(24, 1)
        self.layer2 = nn.Sequential(
            nn.Linear(100, 1000),
            nn.GELU(),
            nn.Linear(1000, 100),
            nn.GELU(),
            nn.Linear(100, 100),
        )


    def forward(self, x):
        batch_size, input_size = x.size(0), x.size(1)
        y = x.reshape(batch_size, 1, input_size)

        y = self.layer0(y.permute([0, 2, 1]))[0]
        y = y.contiguous().view(-1, y.shape[2])
        y = self.layer1(y)
        y = y.contiguous().view(batch_size, -1)

        y = y + x
        y = self.layer2(y)
        y = torch.unsqueeze(y, 1)
        return y  # [batch_size, 1, 100]

class self_attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(self_attention, self).__init__()
        self.linear_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)  # batch_first为True保证（batch,seq,feature）形式

    def forward(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        attn_output, attn_output_weights = self.multihead_attn(q, k, v)
        return attn_output  # (batch,seq,feature)

class LSTMEncoder(nn.Module):
    def __init__(self):
        super(LSTMEncoder, self).__init__()
        self.net_100Hz = lstm_100Hz()
        self.net_200Hz = lstm_200Hz()
        self.net_400Hz = lstm_400Hz()
        self.self_attention = self_attention(embed_dim=100, num_heads=4)

        self.layer0 = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=5, padding=2),  # 64 * 100 # todo
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 64 * 50

            nn.Conv1d(64, 128, kernel_size=5, padding=2),  # 128 * 50
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 128 * 25


            nn.Conv1d(128, 64, kernel_size=3, padding=1),  # 64 * 25
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, x0, x1, x2, x3, x4): # todo
        x0 = self.net_400Hz(x0) # todo
        x1 = self.net_400Hz(x1)
        x2 = self.net_400Hz(x2)
        x3 = self.net_200Hz(x3)
        x4 = self.net_100Hz(x4)

        x = torch.cat((x0, x1, x2, x3, x4), dim=1)  # （batch, 5, 100） # todo

        x = self.self_attention(x)

        x = self.layer0(x)
        x = x.contiguous().view(x.shape[0], -1)


        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.encoder_layer = LSTMEncoder()
        self.layer = nn.Sequential(
            nn.Linear(64 * 25, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(),

            nn.Linear(256, 64),
            nn.LeakyReLU(),

            nn.Linear(64, 2),
            nn.Softmax(dim=1),

        )

    def forward(self, x0, x1, x2, x3, x4):
        x = self.encoder_layer(x0, x1, x2, x3, x4)
        x = self.layer(x)
        return x
