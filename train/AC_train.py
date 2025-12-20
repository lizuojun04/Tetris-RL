import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
from collections import deque
from tqdm import tqdm

# 定义一个线性的 Actor-Critic 模型
# 既然启发式是线性的，我们用线性模型就能最快逼近它
class LinearACModel(nn.Module):
    def __init__(self, num_features=5):
        super(LinearACModel, self).__init__()
        # Actor: 给每个状态打分，决定动作概率 (策略)
        self.actor = nn.Linear(num_features, 1, bias=False)
        # Critic: 预测当前状态的价值 (Value)
        self.critic = nn.Linear(num_features, 1, bias=False)
        
        # 初始化权重：给一个不错的初始值可以加速收敛（仿照启发式）
        # 顺序: lines, holes, bumpiness, total_height, max_height
        with torch.no_grad():
            self.actor.weight.data = torch.tensor([[1.0, -1.0, -0.5, -0.5, -0.5]])
            self.critic.weight.data = torch.tensor([[1.0, -1.0, -0.5, -0.5, -0.5]])

    def forward(self, x):
        value = self.critic(x)
        policy_logits = self.actor(x)
        return value, policy_logits

class A2CTrain:
    def __init__(self,
                 save_path,
                 env,
                 gamma=0.99,
                 epochs=2000,
                 lr=1e-3,
                 save_epoch=50):
        self.save_path = save_path
        self.env = env
        self.gamma = gamma
        self.epochs = epochs
        self.save_epoch = save_epoch
        
        # 初始化模型
        self.model = LinearACModel()
        if torch.cuda.is_available():
            self.model.cuda()
            
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # 确保保存路径存在
        self.best_save_path = os.path.join(save_path, "best")
        if not os.path.exists(self.best_save_path):
            os.makedirs(self.best_save_path)

    def select_action(self, next_steps):
        """
        Actor-Critic 的核心：基于概率选择动作
        """
        next_actions = list(next_steps.keys())
        # (batch, features)
        next_feats = torch.stack([v[1] for v in next_steps.values()])
        
        if torch.cuda.is_available():
            next_feats = next_feats.cuda()

        # 1. 获取 Actor 的打分 (Logits) 和 Critic 的估值 (Values)
        values, logits = self.model(next_feats)
        
        # 2. 将 Actor 的打分转化为概率分布 (Softmax)
        # 注意：这里我们是对“所有可能的下一步”进行 Softmax
        probs = F.softmax(logits.squeeze(1), dim=0)
        
        # 3. 根据概率采样动作
        dist = torch.distributions.Categorical(probs)
        index = dist.sample()
        
        action = next_actions[index.item()]
        
        # 返回必要的信息用于计算 Loss
        # log_prob: 采取该动作的对数概率
        # value: 该动作对应状态的预期价值
        # entropy: 用于鼓励探索
        log_prob = dist.log_prob(index)
        entropy = dist.entropy()
        selected_value = values[index.item()]
        
        # 记录用于传给 step 的额外信息
        current_max_height = next_feats[index.item()][-1].item()
        
        return action, selected_value, log_prob, entropy, current_max_height

    def train(self):
        print("Start A2C (Linear) Training...")
        max_avg_score = 0
        recent_scores = deque(maxlen=50)

        with tqdm(total=self.epochs, desc="Training A2C") as pbar:
            for epoch in range(self.epochs):
                self.env.reset()
                done = False
                steps = 0
                total_loss = 0
                
                # A2C 是 N-step 或 1-step 更新。这里使用 1-step (TD) 
                # 因为是 Linear 模型，单步更新也足够稳定
                
                while not done:
                    # 1. 获取所有可能的下一步
                    next_steps = self.env.get_next_states()
                    if not next_steps: break # 极罕见情况
                    
                    # 2. Actor 选择动作
                    action, value, log_prob, entropy, current_max_height = self.select_action(next_steps)
                    
                    # 3. 执行
                    score, done = self.env.step(action, render=False)
                    
                    # --- 奖励函数 (Reward Shaping) ---
                    reward = score / 10.0
                    if not done:
                        # 如果堆太高，给惩罚
                        if current_max_height > self.env.height / 2:
                            reward -= (current_max_height - self.env.height / 2) * 0.1
                    else:
                        reward = -10.0 # 死亡惩罚
                    # -------------------------------

                    reward_tensor = torch.tensor([reward], dtype=torch.float32)
                    if torch.cuda.is_available(): reward_tensor = reward_tensor.cuda()

                    # 4. 计算 Target Value
                    if done:
                        target_value = reward_tensor
                    else:
                        # 获取下一步的状态信息，计算 V(s')
                        next_next_steps = self.env.get_next_states()
                        if next_next_steps: # 游戏未结束
                            next_feats = torch.stack([v[1] for v in next_next_steps.values()])
                            if torch.cuda.is_available(): next_feats = next_feats.cuda()
                            
                            with torch.no_grad():
                                # Critic 预测 V(s')
                                # 这里取 max 还是 mean? 标准 A2C 是 V(s')。
                                # 但由于我们是在选 next_state，这里简单用 max V 作为对未来的乐观估计
                                next_values, _ = self.model(next_feats)
                                next_value = torch.max(next_values) 
                            
                            target_value = reward + self.gamma * next_value
                        else:
                            target_value = reward_tensor

                    # 5. 计算 Advantage (优势函数)
                    # A = Target - V(s)
                    advantage = target_value - value
                    
                    # 6. 计算 Loss
                    # Critic Loss: 预测的 V 要接近 Target
                    critic_loss = advantage.pow(2)
                    
                    # Actor Loss: -log_prob * Advantage
                    # 如果 A > 0 (好动作)，增加该动作概率；反之减少
                    actor_loss = -log_prob * advantage.detach()
                    
                    # Entropy Loss: 鼓励熵越大越好 (减去 entropy)
                    entropy_loss = -0.01 * entropy
                    
                    loss = critic_loss + actor_loss + entropy_loss

                    # 7. 更新
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    steps += 1

                # 记录分数
                final_score = self.env.score
                recent_scores.append(final_score)
                avg_score = sum(recent_scores) / len(recent_scores)
                
                # 保存最佳
                if avg_score > max_avg_score and len(recent_scores) >= 10:
                    max_avg_score = avg_score
                    torch.save(self.model.state_dict(), os.path.join(self.best_save_path, "A2C_best.pt"))
                    pbar.write(f"Epoch {epoch}: New Max Avg: {max_avg_score:.2f} (Weights Saved)")

                pbar.set_postfix({
                    "score": final_score,
                    "avg": f"{avg_score:.2f}",
                    "loss": f"{total_loss/steps:.3f}" if steps > 0 else 0
                })
                pbar.update(1)