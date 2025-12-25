import torch
import torch.nn as nn
from train.DQN_train import DQNTrain
from src.tetris import Tetris
from src.agents.model import TetrisRL

save_path = './checkpoints'
env = Tetris()
train_model = TetrisRL(hidden_dim=64)
target_model = TetrisRL(hidden_dim=64)
optimizer = torch.optim.Adam(train_model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

trainer = DQNTrain(save_path, env, train_model, target_model, optimizer, criterion)
trainer.train()
