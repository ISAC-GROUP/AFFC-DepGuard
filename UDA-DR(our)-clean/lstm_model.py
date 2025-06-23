import torch
import torch.nn as nn

class lstm_400Hz(nn.Module):
    def __init__(self):
        super(lstm_400Hz, self).__init__()
        self.layer0 = nn.LSTM(input_size=1, hidden_size=24, num_layers=4, batch_first=True)
        self.layer1 = nn.Linear(24, 1)
        self.layer2 = nn.Sequential(
            nn.Linear(400, 1000),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1000, 400),
            nn.LeakyReLU(inplace=True),
            nn.Linear(400, 100),
            # nn.InstanceNorm1d(100),
            # nn.BatchNorm1d(100),
        )

        # self.layer2_1 = nn.Sequential(
        #     nn.Linear(800, 1000),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(1000, 800),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(800, 400),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(400, 100),
        #     # nn.InstanceNorm1d(100),
        #     # nn.BatchNorm1d(100),
        # )

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

class lstm_200Hz(nn.Module):
    def __init__(self):
        super(lstm_200Hz, self).__init__()
        self.layer0 = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.layer1 = nn.Linear(64, 1)
        self.layer2 = nn.Sequential(
            nn.Linear(200, 1000),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1000, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 100),
            # nn.InstanceNorm1d(100),
            # nn.BatchNorm1d(100),
        )
        # self.layer2_1 = nn.Sequential(
        #     nn.Linear(400, 1000),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(1000, 400),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(400, 200),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(200, 100),
        #     # nn.InstanceNorm1d(100),
        #     # nn.BatchNorm1d(100),
        # )

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

class lstm_100Hz(nn.Module):
    def __init__(self):
        super(lstm_100Hz, self).__init__()
        self.layer0 = nn.LSTM(input_size=1, hidden_size=24, num_layers=4, batch_first=True)
        self.layer1 = nn.Linear(24, 1)
        self.layer2 = nn.Sequential(
            nn.Linear(100, 1000),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1000, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 100),
            # nn.InstanceNorm1d(100),
            # nn.BatchNorm1d(100),
        )
        # self.layer2_1 = nn.Sequential(
        #     nn.Linear(200, 1000),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(1000, 200),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(200, 100),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(100, 100),
        #     # nn.InstanceNorm1d(100),
        #     # nn.BatchNorm1d(100),
        # )

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

class LSTMEncoder(nn.Module):
    def __init__(self):
        super(LSTMEncoder, self).__init__()
        self.net_100Hz = lstm_100Hz()
        self.net_200Hz = lstm_200Hz()
        self.net_400Hz = lstm_400Hz()
        # self.layer0 = nn.Sequential(
        #     nn.Conv1d(5, 64, kernel_size=5, padding=2),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2),
        #
        #     nn.Conv1d(64, 128, kernel_size=5, padding=2),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2),
        #
        #     nn.Conv1d(128, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        # )
        self.layer0 = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=5, padding=2),  # 64 * 100
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 64 * 50

            nn.Conv1d(64, 128, kernel_size=5, padding=2),  # 128 * 50
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 128 * 25

            nn.Conv1d(128, 256, kernel_size=3, padding=1),  # 256 * 25
            nn.ReLU(),

            nn.Conv1d(256, 128, kernel_size=3, padding=1),  # 128 * 25
            nn.ReLU(),

            nn.Conv1d(128, 64, kernel_size=3, padding=1),  # 64 * 25
            nn.ReLU(),
        )

    def forward(self, x0, x1, x2, x3, x4):
        x0 = self.net_400Hz(x0)
        x1 = self.net_400Hz(x1)
        x2 = self.net_400Hz(x2)
        x3 = self.net_200Hz(x3)
        x4 = self.net_100Hz(x4)

        x = torch.cat((x0, x1, x2, x3, x4), dim=1)
        x = self.layer0(x)
        x = x.contiguous().view(x.shape[0], -1)
        return x

class LSTMClassifier(nn.Module):
    def __init__(self):
        super(LSTMClassifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(64 * 25, 512),
            # nn.Linear(500, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(),

            nn.Linear(256, 64),
            nn.LeakyReLU(),

            nn.Linear(64, 2),
            nn.Softmax(dim=1)
            # nn.Sigmoid()
        )
    def forward(self, x):
        x = self.layer1(x)
        return x

class ReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer2 = nn.Sequential(
            nn.Linear(64 * 25, 1000),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(1000, 512),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
    def forward(self, x, alpha):
        x_reverse = ReversalLayer.apply(x, alpha)
        domain_output = self.layer2(x_reverse)
        return domain_output