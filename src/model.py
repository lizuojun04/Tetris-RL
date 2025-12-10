import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.ff = nn.Linear(4, 1)

    def forward(self, x):
        return self.ff(x)
