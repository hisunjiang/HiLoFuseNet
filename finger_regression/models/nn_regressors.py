import torch
import torch.nn as nn
import torch.nn.functional as F
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

# ------------- comparison models ---------------#
# %% MLP
class MLP(nn.Module):
    def __init__(self, input_size = 100, hidden_size = 256, output_size = 1, dropout_prob = 0.5):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size // 2),  # e.g., 256 -> 128
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return x.squeeze(-1)

# %% RNN
class LSTM(nn.Module):
        def __init__(self, input_size = 100, hidden_size = 256, output_size = 1, num_layers = 1, dropout_prob = 0.5):
            super().__init__()
            self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                                num_layers = num_layers,  batch_first = True)

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
            x = self.mlp(x)
            return x.squeeze(-1)

# %% CNN
class EEGNet(nn.Module):
    def featureExtraction_blocks(self, dropoutP, *args, **kwargs):
        block1 = nn.Sequential(
                nn.Conv2d(1, self.F1, (1, self.C1),
                          padding=(0, self.C1 // 2), bias=False),
                nn.BatchNorm2d(self.F1),
                Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.nChan, 1),
                                     padding=0, bias=False, max_norm=1,
                                     groups=self.F1),
                nn.BatchNorm2d(self.F1 * self.D),
                nn.ELU(),
                nn.AvgPool2d((1, 4), stride=4),
                nn.Dropout(p=dropoutP))

        block2 = nn.Sequential(
                nn.Conv2d(self.F1 * self.D, self.F1 * self.D,  (1, 16),
                                     padding=(0, 16//2), bias=False,
                                     groups=self.F1 * self.D),
                nn.Conv2d(self.F1 * self.D, self.F2, (1, 1),
                          stride=1, bias=False, padding=0),
                nn.BatchNorm2d(self.F2),
                nn.ELU(),
                nn.AvgPool2d((1, 8), stride=8),
                nn.Dropout(p=dropoutP)
                )
        return nn.Sequential(block1, block2)

    def classifier_block(self, inF, outF):
        return nn.Sequential(
                nn.Linear(inF, outF))

    def calculateOutSize(self, model, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is an array.
        '''
        data = torch.rand(1, 1, nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, nChan, nTime, nClass=2,
                 dropoutP=0.25, F1=8, D=2,
                 C1=64, *args, **kwargs):
        super(EEGNet, self).__init__()
        self.F2 = D*F1
        self.F1 = F1
        self.D = D
        self.nTime = nTime
        self.nClass = nClass
        self.nChan = nChan
        self.C1 = C1

        self.firstBlocks = self.featureExtraction_blocks(dropoutP)
        self.fSize = self.calculateOutSize(self.firstBlocks, nChan, nTime)
        self.lastLayer = self.classifier_block(self.F2 * self.fSize[1], nClass)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)  # (batch, 1, chan, time)
        x = self.firstBlocks(x)
        x = x.reshape(x.size(0), -1)
        x = self.lastLayer(x)
        return x

# %% UNET
# see from https://github.com/UM-Tao/DeepFingerNet/blob/main/DeepFingerNet.py
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, p_conv_drop=0.1):
        super(ConvBlock, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, bias=False, padding='same')
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(p=p_conv_drop)
        self.downsample = nn.MaxPool1d(kernel_size=stride, stride=stride)

    def forward(self, x):
        x = self.conv1d(x)
        x = torch.transpose(x, -2, -1)
        x = self.norm(x)
        x = torch.transpose(x, -2, -1)
        x = self.activation(x)
        x = self.drop(x)
        x = self.downsample(x)
        return x

class ConvBlock_up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, p_conv_drop=0.1):
        super(ConvBlock_up, self).__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.activation = nn.GELU()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, bias=False, padding='same')
        self.drop = nn.Dropout(p=p_conv_drop)
        self.norm_2 = nn.LayerNorm(out_channels)
        self.conv1d_2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, bias=False, padding='same')

    def forward(self, x):
        x = torch.transpose(x, -2, -1)
        x = self.norm(x)
        x = torch.transpose(x, -2, -1)
        x = self.activation(x)
        x = self.conv1d(x)
        x = self.drop(x)

        x = torch.transpose(x, -2, -1)
        x = self.norm_2(x)
        x = torch.transpose(x, -2, -1)
        x = self.activation(x)
        x = self.conv1d_2(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, scale, **args):
        super(UpConvBlock, self).__init__()
        self.conv_block_up = ConvBlock_up(**args)
        # self.upsample = nn.Upsample(scale_factor=scale, mode='linear', align_corners=False)

    def forward(self, x, target_tensor=None):
        x = self.conv_block_up(x)
        if target_tensor is not None:
            x = F.interpolate(x, size=target_tensor.size(-1), mode='linear', align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        return x

class DeepFingerNet(nn.Module):
    """the backbone：1D UNet++"""

    def __init__(self,input_size, num_classes=5, deep_supervision=False):
        super(DeepFingerNet, self).__init__()
        self.deep_supervision = deep_supervision

        self.spatial_reduce = ConvBlock(input_size, 32, kernel_size=3, stride=1)

        self.stage_1 = ConvBlock(32, 64, stride=2, kernel_size=7)
        self.stage_2 = ConvBlock(64, 128, stride=2, kernel_size=7)
        self.stage_3 = ConvBlock(128, 256, stride=2, kernel_size=7)

        self.upsample_2_1 = UpConvBlock(scale=2, in_channels=256, out_channels=128, kernel_size=7)
        self.upsample_1_1 = UpConvBlock(scale=2, in_channels=128, out_channels=64, kernel_size=7)
        self.upsample_1_2 = UpConvBlock(scale=2, in_channels=128, out_channels=64, kernel_size=7)
        self.upsample_0_1 = UpConvBlock(scale=2, in_channels=64, out_channels=32, kernel_size=7)
        self.upsample_0_2 = UpConvBlock(scale=2, in_channels=64, out_channels=32, kernel_size=7)
        self.upsample_0_3 = UpConvBlock(scale=2, in_channels=64, out_channels=32, kernel_size=7)

        self.CONV0_3 = ConvBlock_up(in_channels=128, out_channels=32, stride=1, kernel_size=7)
        self.final_super_0_3 = nn.Conv1d(32, num_classes, kernel_size=1, padding='same')

    def forward(self, x):
        # x: (batch, C, T, F)
        B, C, T, F = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B, -1, T)  # flatten the input

        x_0_0 = self.spatial_reduce(x)
        x_1_0 = self.stage_1(x_0_0)
        x_2_0 = self.stage_2(x_1_0)
        x_3_0 = self.stage_3(x_2_0)

        x_0_1 = self.upsample_0_1(x_1_0)
        x_1_1 = self.upsample_1_1(x_2_0)
        # x_2_1 = self.upsample_2_1(x_3_0)
        x_2_1 = self.upsample_2_1(x_3_0, target_tensor=x_2_0)

        # x_1_2 = self.upsample_1_2(x_2_1)
        x_1_2 = self.upsample_1_2(x_2_1, target_tensor=x_1_1)
        x_0_2 = self.upsample_0_2(x_1_1)
        # x_0_3 = self.upsample_0_3(x_1_2)
        x_0_3 = self.upsample_0_3(x_1_2, target_tensor=x_0_0)

        x_Out_3 = self.CONV0_3(torch.cat([x_0_0, x_0_1, x_0_2, x_0_3], dim=1))
        x_output = self.final_super_0_3(x_Out_3)

        return x_output[:,:, -1]

# %% CNN RNN hybrid
class CNN_LSTM(nn.Module):
    def __init__(self, input_size, output_size = 1, dropout_prob=0.5):
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
        x = self.cnn(x)
        x = torch.mean(x, dim=3)
        x = x.permute(0, 2, 1)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.mlp(x)
        return x.squeeze(-1)

# %% Transformers
class BaseTransformer(nn.Module):
    def __init__(self, patch_len, input_dim, d_model, num_heads, num_layers):
        super().__init__()
        # linear projection layer
        self.projection = nn.Linear(input_dim, d_model)
        # Class Token, similar to ViT, BERT
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # learnable positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, patch_len + 1, d_model)) # patch_len + 1 cls_token
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: [batch, sequence, feature]
        B, T, _ = x.shape
        x = self.projection(x) # -> [B, T, D]
        
        # add Class Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # -> [B, T + 1, D]
        
        # add positional embedding
        x = x + self.pos_embed
        
        # Transformer encoder
        x = self.transformer(x) # -> [B, T + 1, D]

        return x[:, 0] # output the CLS token

class WaT(nn.Module):
    def __init__(self, nseq, nchan, nfreq, nClass):
        super().__init__()
        self.D = 256
        self.nhead = 4
        self.nlayer = 2
        self.encoder = BaseTransformer(patch_len=nseq, input_dim=nchan*nfreq, d_model=self.D, num_heads=self.nhead, num_layers=self.nlayer)
        self.MLPhead = nn.Linear(self.D, nClass)

    def forward(self, x):
        # x: (batch, C, T, F)
        B, C, T, F = x.shape
        
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, T, F * C)
        
        cls_output = self.encoder(x) # -> [Batch, d_model]
        
        return self.MLPhead(cls_output)
        
class WaTFi(nn.Module):
    def __init__(self, nseq, nchan, nfreq, nClass):
        super().__init__()
        self.D = 128
        self.nhead = 4
        self.nlayer = 2
        # frequency independent, refer to PatchTST
        self.encoder = BaseTransformer(patch_len=nseq, input_dim=nchan, d_model=self.D, num_heads=self.nhead, num_layers=self.nlayer)
        self.MLPhead = nn.Linear(self.D * nfreq, nClass)

    def forward(self, x):
        # x: (batch, C, T, F)
        B, C, T, F = x.shape
        
        x = x.permute(0, 3, 2, 1) # -> [Batch, Freq, Time, Chan]
        
        # Batch Folding
        x = x.reshape(B * F, T, C)
        
        x = self.encoder(x) # -> [B * F, D]
        
        # Unfold
        x = x.reshape(B, F, -1) # -> [B, F, D]
        
        # Concatenate
        x = x.reshape(B, -1) # -> [B, F*D]
        
        return self.MLPhead(x)

class WaTEi(nn.Module):
    def __init__(self, nseq, nchan, nfreq, nClass):
        super().__init__()
        self.D = 128
        self.nhead = 4
        self.nlayer = 2
        # channel independent, refer to PatchTST
        self.encoder = BaseTransformer(patch_len=nseq, input_dim=nfreq, d_model=self.D, num_heads=self.nhead, num_layers=self.nlayer)
        self.MLPhead = nn.Linear(self.D * nchan, nClass)

    def forward(self, x):
        # x: (batch, C, T, F)
        B, C, T, F = x.shape
        
        # Batch Folding
        x = x.reshape(B * C, T, F)
        
        x = self.encoder(x) # -> [B * C, D]
        
        # Unfold
        x = x.reshape(B, C, -1) # -> [B, C, D]
        
        # Concatenate
        x = x.reshape(B, -1) # -> [B, C*D]
        
        return self.MLPhead(x)

# %% the proposed model
class HiLoFuseNet(nn.Module):
    def __init__(self, C, F, lstm_hidden=256, output_size = 1, D = 16, dropout_prob=0.5):
        super().__init__()
        self.D = D
        self.spatialConv = nn.Sequential(
            Conv2dWithConstraint(F, self.D*F, (C, 1),
                                 padding=0, bias=False, max_norm=1,
                                 groups=F),
            nn.BatchNorm2d(self.D*F),
            nn.ELU(),
            nn.Dropout(dropout_prob),

            nn.Conv2d(self.D*F, self.D*F, (1, 20),
                      padding=(0, 20 // 2), bias=False,
                      groups=self.D*F),
            nn.Conv2d(self.D*F, self.D*F, (1, 1), bias=False),
            nn.BatchNorm2d(self.D*F),
            nn.ELU(),
            nn.AvgPool2d((1, 10), stride=10),
            nn.Dropout(dropout_prob),
        )

        self.lstm1 = nn.LSTM(self.D*F, lstm_hidden, batch_first=True, bidirectional=False)

        mlp_in = lstm_hidden
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_in//2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_in//2, output_size)
        )

    def forward(self, x):
        # x: (batch, C, T, F)
        in_cnn = x
        in_cnn = in_cnn.permute(0, 3, 1, 2)
        out_cnn = self.spatialConv(in_cnn)

        in_lstm = out_cnn.squeeze(2)
        in_lstm = in_lstm.transpose(1, 2)
        x_lstm, _ = self.lstm1(in_lstm)
        out_lstm = x_lstm[:, -1, :] 

        out = self.mlp(out_lstm) 

        return out.squeeze(-1)

# %% ablated versions
class HiLoFuseNet_woDSConv(nn.Module):
    def __init__(self, input_size = 100, hidden_size = 256, output_size = 1, num_layers = 1, dropout_prob = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers,  batch_first = True)

        mlp_in = hidden_size 
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_in//2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_in//2, output_size)
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
    def __init__(self, C, F, output_size = 1, D = 16, dropout_prob=0.5):
        super().__init__()
        self.D = D
        self.spatialConv = nn.Sequential(
            Conv2dWithConstraint(F, self.D*F, (C, 1),
                                 padding=0, bias=False, max_norm=1,
                                 groups=F),
            nn.BatchNorm2d(self.D*F),
            nn.ELU(),
            nn.Dropout(dropout_prob),

            nn.Conv2d(self.D*F, self.D*F, (1, 20),
                      padding=(0, 20 // 2), bias=False,
                      groups=self.D*F),
            nn.Conv2d(self.D*F, self.D*F, (1, 1), bias=False),
            nn.BatchNorm2d(self.D*F),
            nn.ELU(),
            nn.AvgPool2d((1, 10), stride=10),
            nn.Dropout(dropout_prob),
            
            # do GAP to adapt MLP
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        mlp_in = self.D*F
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_in//2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_in//2, output_size)
        )

    def forward(self, x):
        # x: (batch, C, T, F)
        in_cnn = x # self.pool(x)
        in_cnn = in_cnn.permute(0, 3, 1, 2)
        out_cnn = self.spatialConv(in_cnn)
        out_cnn = out_cnn.reshape(out_cnn.size(0), -1)

        out = self.mlp(out_cnn)  # (B, num_classes)

        return out.squeeze(-1)
