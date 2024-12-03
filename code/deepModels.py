import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class cnnModel(nn.Module):
    def __init__(self, inputDim, args):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (10, 1), padding='same')
        self.conv2 = nn.Conv2d(1, 1, (10, 1), padding='same')
        self.conv3 = nn.Conv2d(1, 1, (10, 1), padding='same')
        self.conv4 = nn.Conv2d(1, 1, (inputDim, 1))

        self.args = args

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(3)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        output = self.args.p_max * F.sigmoid(self.conv4(x))
        return output


class deepModel(nn.Module):
    def __init__(self, inputDim, outputDim, args):
        super(deepModel, self).__init__()
        self.args = args

        set_seed(args)

        self.config = np.zeros(2+args.network_hidden_config.shape[0], dtype=int)
        self.config[0] = inputDim
        self.config[1:-1] = args.network_hidden_config
        self.config[-1] = outputDim
        self.hidden_num = args.network_hidden_config.shape[0]

        self.fc = nn.ModuleList()
        for i in range(self.hidden_num+1):
            self.fc.append(nn.Linear(self.config[i], self.config[i+1]))

        self.history = []
        self.training_flag = False  # if the model is trained or not

        self.dropout = nn.Dropout(args.dropout_factor)
        self.activation_fun = torch.__dict__[args.activation_fun]

    def forward(self, x):
        for i in range(self.hidden_num):
            x = self.activation_fun(self.dropout(self.fc[i](x)))
        output = self.args.p_max * torch.sigmoid(self.fc[-1](x))
        return output