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
                 num_featurs = 4,
                 board_height = 20,
                 board_width = 10,
                 grid_channel_in = 1):
        super(TetrisRL, self).__init__()

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

        self.feat_fc = nn.Linear(num_featurs, 64)

        self.fusion_fc1 = nn.Linear(512 + 64, 256)
        self.fusion_fc2 = nn.Linear(256, 1)

    def forward(self, grid, feature):
        # grid
        x = nn.functional.relu(self.conv1(grid))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.cnn_fc(x))

        # feature
        y = nn.functional.relu(self.feat_fc(feature))

        # combine grid and feature
        combined = torch.cat((x, y), dim = 1)
        combined = nn.functional.relu(self.fusion_fc1(combined))
        q_values = self.fusion_fc2(combined)

        return q_values
