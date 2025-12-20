import torch
import torch.nn as nn
from train.DQN_train import DQNTrain
from train.AC_train import A2CTrain
from train.Q_train import QTrain
from src.tetris import Tetris
from src.agents.model import TetrisRL
import argparse

save_path = './checkpoints'
env = Tetris()
train_model = TetrisRL()
target_model = TetrisRL()   
optimizer = torch.optim.Adam(train_model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

parser = argparse.ArgumentParser("Tetris RL Training Framework")
parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "ac", "q"], help="Choose the training agent: 'dqn' or 'a2c'")
args = parser.parse_args()
if args.agent == "ac":
    trainer = A2CTrain(save_path, env, lr=0.0005, epochs=150)
    trainer.train()
if args.agent == "dqn":
    trainer = DQNTrain(save_path, env, train_model, target_model, optimizer, criterion)
    trainer.train()

if args.agent == "q":
    trainer = QTrain(save_path, env, lr=1e-3, epochs=2000)
    trainer.train_online()
