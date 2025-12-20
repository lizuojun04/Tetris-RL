import torch
import torch.nn as nn
import random
import os
from collections import deque
from tqdm import tqdm

# 定义线性模型：没有任何隐藏层，直接映射 特征 -> Q值
class QModel(nn.Module):
    def __init__(self, num_features=5):
        super(QModel, self).__init__()
        # 只有一层：输入5个特征 -> 输出1个Q值
        self.fc = nn.Linear(num_features, 1, bias=False)
        
        # 权重初始化：
        # 虽然可以随机初始化，但给一个类似启发式的初始方向有助于避免早期就死太快
        # 启发式参考: [0.76, -0.36, -0.18, -0.51, 0]
        # 我们这里用小的随机数或稍微引导一下
        with torch.no_grad():
            # 稍微引导一点：消除行是正的，其他是负的
            self.fc.weight.data = torch.tensor([[0.1, -0.1, -0.1, -0.1, -0.1]])

    def forward(self, x):
        return self.fc(x)

class QTrain:
    def __init__(self,
                 save_path,
                 env,
                 gamma=0.99,
                 epochs=2000,
                 lr=1e-3,             # 线性模型通常可以用大一点的学习率
                 batch_size=512,      # 批量更新更稳定
                 memory_size=30000,   # 记忆池
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 save_epoch=50):
        
        self.save_path = save_path
        self.env = env
        self.gamma = gamma
        self.epochs = epochs
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.save_epoch = save_epoch
        
        # 初始化模型
        self.model = QModel()
        if torch.cuda.is_available():
            self.model.cuda()
            
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # 确保保存路径
        self.best_save_path = os.path.join(save_path, "best")
        if not os.path.exists(self.best_save_path):
            os.makedirs(self.best_save_path)

    def get_action(self, next_steps):
        """Epsilon-Greedy 策略"""
        next_actions = list(next_steps.keys())
        next_feats = torch.stack([v[1] for v in next_steps.values()])
        if torch.cuda.is_available():
            next_feats = next_feats.cuda()

        if random.random() <= self.epsilon:
            index = random.randint(0, len(next_steps) - 1)
        else:
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(next_feats).squeeze(1)
                index = torch.argmax(predictions).item()
            self.model.train()

        return next_actions[index], next_feats[index].unsqueeze(0), next_feats[index][-1].item()

    def add_memory(self, state, reward, next_state, done):
        self.memory.append((state, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        batch = random.sample(self.memory, self.batch_size)
        
        # 整理 batch 数据
        state_batch = torch.cat([x[0] for x in batch])
        reward_batch = torch.tensor([x[1] for x in batch], dtype=torch.float32)
        # next_state 比较特殊，有的 step 死了就没有 next_state
        non_final_mask = torch.tensor([x[2] is not None for x in batch], dtype=torch.bool)
        # 过滤出非空状态
        non_final_next_states = torch.cat([x[2] for x in batch if x[2] is not None])
        
        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            non_final_mask = non_final_mask.cuda()
            non_final_next_states = non_final_next_states.cuda()
            
        # 1. 计算当前 Q(s, a)
        q_values = self.model(state_batch).squeeze(1)
        
        # 2. 计算目标 Q 值 (Target)
        # Q_target = r + gamma * max Q(s', a')
        # 初始化为 reward
        next_q_values = torch.zeros(self.batch_size)
        if torch.cuda.is_available(): next_q_values = next_q_values.cuda()
        
        if len(non_final_next_states) > 0:
            with torch.no_grad():
                # Q-learning 的核心：对 s' 选取最大值
                # 注意：这里的 LinearQModel 输入特征直接得到 Q，没有像 DQN 那样先 argmax
                # 我们存的 next_state 实际上是 "执行了动作后的状态特征"，所以直接过网络就是那个动作的 Q
                # 在 Tetris 的这个实现里，"state" 其实已经是 (s, a) 的特征了
                
                # 等等，这里需要注意：
                # 我们的 memory 里存的 next_state 应该是 "下一步所有可能的动作对应的特征列表" 吗？
                # 不，这样存太占内存。
                # 简化处理：我们在 step 里存的是 "选中的那个动作带来的状态特征"。
                # 但是 Q-learning 要求 max_a' Q(s', a')。
                # 如果只存了选中的 s'，那就是 SARSA 了。
                # 为了实现真正的 Q-learning，我们在 replay 时其实很难高效拿到 s' 的所有动作。
                # **修正策略**：
                # 鉴于 Tetris 环境的特殊性（get_next_states 返回列表），
                # 我们可以让 model 在 replay 阶段只做简单的 TD 更新，或者
                # 采用一种近似：我们在存储 memory 时，提前算好 max_q 并存进去？
                # 不行，因为网络在变。
                
                # **最佳方案 (针对 Tetris 环境)**：
                # 我们回到 "TD 训练" 的逻辑，但是加上 Replay Buffer。
                # 这种方法被称为 "Fitted Q Iteration" 的变体。
                pass

        # -------------------------------------------------------------
        # 重新思考：由于 get_next_states() 的特殊性，标准的 Replay Buffer 很难实现真正的 max_a' Q(s', a')
        # 因为在 replay 时我们没有 environment，无法调用 get_next_states() 生成 s' 的所有后续。
        # 除非把所有后续都存进内存（显存爆炸）。
        #
        # 因此，对于这个特殊的 Tetris 环境，如果要用 Replay Buffer，
        # 通常只能存 (state_feat, reward, done, max_next_q_prediction) -- 不，这样相当于 Target 固定了。
        # 
        # 所以，在这个环境里，最强的 "Linear Q-learning" 其实就是
        # **在线学习 (Online Learning) + 线性模型**。
        # 不需要 Replay Buffer。
        # -------------------------------------------------------------
        return 0 

    # 重写 train 方法，采用在线更新 (无 Replay Buffer)，但这正是 Linear Q 的威力所在
    def train_online(self):
        print("Start Linear Q-learning (Online)...")
        max_avg_score = 0
        recent_scores = deque(maxlen=50)

        with tqdm(total=self.epochs, desc="Training Linear Q") as pbar:
            for epoch in range(self.epochs):
                self.env.reset()
                done = False
                steps = 0
                total_loss = 0
                
                # 1. 获取初始候选状态
                next_steps = self.env.get_next_states()
                
                while not done:
                    # 2. 选动作 (Epsilon-Greedy)
                    action, state_feat, current_max_height = self.get_action(next_steps)
                    
                    # 3. 执行
                    score, done = self.env.step(action, render=False)
                    
                    # --- 奖励函数 ---
                    reward = score / 10.0
                    if not done:
                        if current_max_height > self.env.height / 2:
                            reward -= (current_max_height - self.env.height / 2) * 0.1
                    else:
                        reward = -10.0
                    # ---------------
                    
                    reward_tensor = torch.tensor([reward], dtype=torch.float32)
                    if torch.cuda.is_available(): reward_tensor = reward_tensor.cuda()

                    # 4. 计算 Target (Q-learning 核心)
                    if done:
                        target = reward_tensor
                    else:
                        # 获取 s' 的所有可能后续动作
                        next_next_steps = self.env.get_next_states()
                        
                        if next_next_steps:
                            next_feats = torch.stack([v[1] for v in next_next_steps.values()])
                            if torch.cuda.is_available(): next_feats = next_feats.cuda()
                            
                            with torch.no_grad():
                                # 预测所有可能的后续 Q 值
                                all_next_qs = self.model(next_feats)
                                # 取最大值 (Greedy) -> 这就是 Q-learning 的 max Q(s', a')
                                max_next_q = torch.max(all_next_qs)
                            
                            target = reward_tensor + self.gamma * max_next_q
                        else:
                            target = reward_tensor

                    # 5. 更新当前步
                    # 预测当前 Q(s, a)
                    pred_q = self.model(state_feat)
                    
                    loss = self.criterion(pred_q.view(-1), target.view(-1))
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    steps += 1
                    
                    # 更新 next_steps 给下一轮循环用
                    if not done:
                        next_steps = next_next_steps

                # --- Epoch 结束 ---
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                
                final_score = self.env.score
                recent_scores.append(final_score)
                avg_score = sum(recent_scores) / len(recent_scores)
                
                # 保存最佳模型
                if avg_score > max_avg_score and len(recent_scores) > 10:
                    max_avg_score = avg_score
                    torch.save(self.model.state_dict(), os.path.join(self.best_save_path, "Q_best.pt"))
                    pbar.write(f"Epoch {epoch}: New Max Avg: {max_avg_score:.2f} (Weights Saved)")

                pbar.set_postfix({
                    "score": final_score,
                    "avg": f"{avg_score:.2f}",
                    "loss": f"{total_loss/steps:.4f}" if steps else 0,
                    "eps": f"{self.epsilon:.2f}"
                })
                pbar.update(1)