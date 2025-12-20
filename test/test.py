"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from src.tetris import Tetris
from src.agents.model import TetrisRL
from src.agents.heuristic_agent import HeuristicAgent

# =============================================================================
# 1. 在这里补充定义训练时使用的线性模型结构
#    这样才能正确加载 checkpoints 中的 actor/critic 权重
# =============================================================================

class LinearACModel(nn.Module):
    def __init__(self, num_features=5):
        super(LinearACModel, self).__init__()
        self.actor = nn.Linear(num_features, 1, bias=False)
        self.critic = nn.Linear(num_features, 1, bias=False)

    def forward(self, x):
        value = self.critic(x)
        policy_logits = self.actor(x)
        return value, policy_logits

    # 增加 get_action 方法以适配 test 循环的调用接口
    def get_action(self, next_steps):
        next_actions = list(next_steps.keys())
        # 提取特征并堆叠: (batch_size, 5)
        next_feats = torch.stack([v[1] for v in next_steps.values()])
        
        if torch.cuda.is_available():
            next_feats = next_feats.cuda()

        # 预测
        with torch.no_grad():
            _, logits = self.forward(next_feats)
        
        # 测试时使用确定性策略 (Argmax)，选概率最大的动作
        probs = F.softmax(logits.squeeze(1), dim=0)
        index = torch.argmax(probs).item()
        
        return next_actions[index]


class LinearQModel(nn.Module):
    def __init__(self, num_features=5):
        super(LinearQModel, self).__init__()
        self.fc = nn.Linear(num_features, 1, bias=False)

    def forward(self, x):
        return self.fc(x)

    def get_action(self, next_steps):
        next_actions = list(next_steps.keys())
        next_feats = torch.stack([v[1] for v in next_steps.values()])
        
        if torch.cuda.is_available():
            next_feats = next_feats.cuda()

        with torch.no_grad():
            predictions = self.forward(next_feats).squeeze(1)
            
        # Q-learning 测试时直接选分数最高的 (Greedy)
        index = torch.argmax(predictions).item()
        return next_actions[index]

# =============================================================================
# End of Model Definitions
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="checkpoints")
    parser.add_argument("--output_path", type=str, default="output_video")
    parser.add_argument("--agent", type=str, default="dqn",
                        choices=["dqn", "ac", "q", "heuristic"],
                        help="Choose the agent")
    parser.add_argument("--test", type=int, default=10, help="test times")
    parser.add_argument("--render", action="store_true", help="Enable rendering to video.")

    args = parser.parse_args()
    return args


def test(opt):
    if not os.path.exists(opt.output_path):
        os.mkdir(opt.output_path)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    agent = None # 初始化 agent

    # ==========================
    # Agent 1: Heuristic (启发式)
    # ==========================
    if opt.agent == "heuristic":
        agent = HeuristicAgent()

    # ==========================
    # Agent 2: DQN (原始 MLP)
    # ==========================
    elif opt.agent == "dqn":
        agent = TetrisRL() # 使用原始模型结构
        model_path = "{}/best/DQN_best.pt".format(opt.saved_path)
        
        print(f"Loading DQN model from {model_path}")
        if torch.cuda.is_available():
            agent_dict = torch.load(model_path, weights_only=False)
        else:
            agent_dict = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
        
        agent.load_state_dict(agent_dict)
        agent.eval()
        if torch.cuda.is_available():
            agent.cuda()

    # ==========================
    # Agent 3: Actor-Critic (AC)
    # ==========================
    elif opt.agent == "ac":
        agent = LinearACModel() # 【关键修改】使用 LinearACModel 而不是 TetrisRL
        model_path = "{}/best/A2C_best.pt".format(opt.saved_path)
        
        print(f"Loading AC model from {model_path}")
        if torch.cuda.is_available():
            agent_dict = torch.load(model_path, weights_only=False)
        else:
            agent_dict = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
        
        agent.load_state_dict(agent_dict)
        agent.eval()
        if torch.cuda.is_available():
            agent.cuda()

    # ==========================
    # Agent 4: Linear Q-learning
    # ==========================
    elif opt.agent == "q":
        agent = LinearQModel() # 【关键修改】使用 LinearQModel 而不是 TetrisRL
        model_path = "{}/best/Q_best.pt".format(opt.saved_path)
        # 如果你之前保存的是 TD_best.pt 或 LinearQ_best.pt，请修改上面的文件名
        
        print(f"Loading Q model from {model_path}")
        if os.path.exists(model_path):
            if torch.cuda.is_available():
                agent_dict = torch.load(model_path, weights_only=False)
            else:
                agent_dict = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
            agent.load_state_dict(agent_dict)
            agent.eval()
            if torch.cuda.is_available():
                agent.cuda()
        else:
            print(f"Error: Model file not found at {model_path}")
            return

    # 开始游戏环境
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)

    test_times = opt.test
    total_score = 0
    
    for i in range(test_times):
        if opt.render:
            video_name = os.path.join(opt.output_path, f"{opt.agent}_game_{i}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
            out = cv2.VideoWriter(video_name, fourcc, opt.fps,
                                  (int(1.5*opt.width*opt.block_size), opt.height*opt.block_size))
        else:
            out = None
            
        env.reset()
        while True:
            next_steps = env.get_next_states()
            if not next_steps: # 游戏结束或无路可走
                if out is not None: out.release()
                break
            
            # 这里调用 agent.get_action
            # 我们在上面的类定义中添加了 get_action 方法，所以这里可以直接调用
            action = agent.get_action(next_steps)
            
            _, done = env.step(action, render=opt.render, video=out)

            if done:
                if out is not None:
                    out.release()
                break
                
        print(f"Game {i+1} Score: {env.score}")
        total_score += env.score

    print(f"Average Score: {total_score/test_times}")

if __name__ == "__main__":
    opt = get_args()
    test(opt)