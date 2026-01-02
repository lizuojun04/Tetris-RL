import torch
import torch.nn as nn
from train.DQN_train import DQNTrain
from src.tetris import Tetris
from src.agents.model import TetrisRL, TetrisRL_conv

save_path = './checkpoints'
env = Tetris()
# train_model = TetrisRL()
# target_model = TetrisRL()
train_model = TetrisRL_conv()
target_model = TetrisRL_conv()
optimizer = torch.optim.Adam(train_model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

trainer = DQNTrain(save_path, env, train_model, target_model, optimizer, criterion)
trainer.train()
