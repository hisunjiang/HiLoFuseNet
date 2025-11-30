import torch
import torch.nn as nn
import torch.nn.init as init

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class LSTM(nn.Module):
    def __init__(self, input_size=100, hidden_size=256, output_size=1, num_layers=1, dropout_prob=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),  # e.g., 256 -> 128
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)

        x = x[:, -1, :]
        x = self.mlp(x)  # shape: (batch, seq_len, output_size)
        return x.squeeze(-1)


class CNN_LSTM(nn.Module):
    def __init__(self, input_size, output_size=1, dropout_prob=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_size, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.lstm1 = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)

        self.mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        # x: (batch, C, T, F)
        x = self.cnn(x)  # (batch, 128, T, F)
        x = torch.mean(x, dim=3)  # get (batch, 128, T)
        x = x.permute(0, 2, 1)  # (batch, T, 128)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.mlp(x)
        return x.squeeze(-1)


class HiLoFuseNet(nn.Module):
    def __init__(self, C, F, lstm_hidden=256, output_size=1, D=16, dropout_prob=0.2):
        super().__init__()
        self.D = D
        self.spatialConv = nn.Sequential(
            Conv2dWithConstraint(F, self.D * F, (C, 1),
                                 padding=0, bias=False, max_norm=1,
                                 groups=F),
            nn.BatchNorm2d(self.D * F),
            nn.ELU(),
            nn.Dropout(dropout_prob),

            nn.Conv2d(self.D * F, self.D * F, (1, 20),
                      padding=(0, 20 // 2), bias=False,
                      groups=self.D * F),
            nn.Conv2d(self.D * F, self.D * F, (1, 1), bias=False),
            nn.BatchNorm2d(self.D * F),
            nn.ELU(),
            nn.AvgPool2d((1, 10), stride=10),
            nn.Dropout(dropout_prob),
        )

        self.lstm1 = nn.LSTM(self.D * F, lstm_hidden, batch_first=True, bidirectional=False)

        mlp_in = lstm_hidden  # lstm_hidden + D * F
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_in // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_in // 2, output_size)
        )

    def forward(self, x):
        # x: (batch, C, T, F)
        in_cnn = x  # self.pool(x)
        in_cnn = in_cnn.permute(0, 3, 1, 2)
        out_cnn = self.spatialConv(in_cnn)

        in_lstm = out_cnn.squeeze(2)
        in_lstm = in_lstm.transpose(1, 2)
        x_lstm, _ = self.lstm1(in_lstm)
        out_lstm = x_lstm[:, -1, :]  # (B, lstm_hidden)

        out = self.mlp(out_lstm)  # (B, num_classes)

        return out.squeeze(-1)


# models modified for ablation study
class HiLoFuseNet_woDSConv(nn.Module):
    def __init__(self, input_size=100, hidden_size=256, output_size=1, num_layers=1, dropout_prob=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        mlp_in = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_in // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_in // 2, output_size)
        )

    def forward(self, x):
        # x: (batch, C, T, F)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), -1)
        out_lstm, _ = self.lstm(x)

        out_lstm = out_lstm[:, -1, :]
        out = self.mlp(out_lstm)  # (B, num_classes)

        return out.squeeze(-1)


class HiLoFuseNet_woLSTM(nn.Module):
    def __init__(self, C, F, output_size=1, D=16, dropout_prob=0.2):
        super().__init__()
        self.D = D
        self.spatialConv = nn.Sequential(
            Conv2dWithConstraint(F, self.D * F, (C, 1),
                                 padding=0, bias=False, max_norm=1,
                                 groups=F),
            nn.BatchNorm2d(self.D * F),
            nn.ELU(),
            nn.Dropout(dropout_prob),

            nn.Conv2d(self.D * F, self.D * F, (1, 20),
                      padding=(0, 20 // 2), bias=False,
                      groups=self.D * F),
            nn.Conv2d(self.D * F, self.D * F, (1, 1), bias=False),
            nn.BatchNorm2d(self.D * F),
            nn.ELU(),
            nn.AvgPool2d((1, 10), stride=10),
            nn.Dropout(dropout_prob),
        )

        mlp_in = self.D * F * 20
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_in // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_in // 2, output_size)
        )

    def forward(self, x):
        # x: (batch, C, T, F)
        in_cnn = x  # self.pool(x)
        in_cnn = in_cnn.permute(0, 3, 1, 2)
        out_cnn = self.spatialConv(in_cnn)
        out_cnn = out_cnn.reshape(out_cnn.size(0), -1)

        out = self.mlp(out_cnn)  # (B, num_classes)

        return out.squeeze(-1)
