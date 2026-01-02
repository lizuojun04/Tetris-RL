import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.ff = nn.Linear(4, 1)

    def forward(self, x):
        return self.ff(x)

class TetrisRL(nn.Module):
    def __init__(self, 
                 num_featurs = 5,
                 board_height = 20,
                 board_width = 10,
                 grid_channel_in = 1):
        super(TetrisRL, self).__init__()
        self.fc1 = nn.Linear(num_featurs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, grid, feature):
        x = torch.relu(self.fc1(feature))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
    def get_action(self, next_steps):
        next_actions = list(next_steps.keys())
        next_grids = torch.stack([v[0] for v in next_steps.values()])
        next_feats = torch.stack([v[1] for v in next_steps.values()])
        if torch.cuda.is_available():
            next_grids = next_grids.cuda()
            next_feats = next_feats.cuda()
        predictions = self.forward(next_grids, next_feats)
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        return action


class TetrisRL_conv(nn.Module):
    def __init__(self, 
                 num_featurs = 5,
                 board_height = 20,
                 board_width = 10,
                 grid_channel_in = 1):
        super(TetrisRL_conv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = grid_channel_in,
                               out_channels = 32,
                               kernel_size = 3,
                               padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32,
                               out_channels = 64,
                               kernel_size = 3,
                               padding = 1)

        self.conv_out_size = 64 * board_height * board_width
        self.cnn_fc = nn.Linear(self.conv_out_size, 512)

        self.feat_bn = nn.BatchNorm1d(num_featurs)
        self.feat_fc = nn.Linear(num_featurs, 64)

        self.fusion_fc1 = nn.Linear(512 + 64, 256)
        self.fusion_fc2 = nn.Linear(256, 1)

    """
    在当前的 S 下，
    环境给出了所有的可能的动作和实行这个动作后的下一个 S'
    然后我们根据 S' 通过神经网络 Q 得到一个值，
    再根据这个值来选动作
    实际上也相当于计算了所有的 Q(S, a)，
    只不过中间的一些步骤由 env 完成了
    """
    def forward(self, grid, feature):
        # grid
        x = nn.functional.relu(self.conv1(grid))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.cnn_fc(x))

        # feature
        y = self.feat_bn(feature)
        y = nn.functional.relu(self.feat_fc(y))

        # combine grid and feature
        combined = torch.cat((x, y), dim = 1)
        combined = nn.functional.relu(self.fusion_fc1(combined))
        q_values = self.fusion_fc2(combined)

        return q_values

    def get_action(self, next_steps):
        next_actions = list(next_steps.keys())
        next_grids = torch.stack([v[0] for v in next_steps.values()])
        next_feats = torch.stack([v[1] for v in next_steps.values()])
        if torch.cuda.is_available():
            next_grids = next_grids.cuda()
            next_feats = next_feats.cuda()
        predictions = self.forward(next_grids, next_feats)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        return action
