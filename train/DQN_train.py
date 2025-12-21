import re
import torch
import torch.nn as nn
import random
from collections import deque

class DQNTrain:
    def __init__(
        self,
        save_path,
        env,
        train_model,
        target_model,
        optimizer,
        criterion,
        memory_size=30000,
        batch_size=512,
        window_size=50,
        gamma=0.99,
        failed_penalty = -10.0,
        height_penalty_scalar=0.1,
        safe_height_factor=0.5,
        steps_num=1,
        epochs=3000,
        fresh_epoch=10,
        save_epoch=50,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995):
        self.save_path = save_path
        self.env = env
        self.train_model = train_model
        self.target_model = target_model
        self.optimizer = optimizer
        self.criterion = criterion
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.window_size = window_size
        self.gamma = gamma
        self.failed_penalty = failed_penalty
        self.height_penalty_scalar = height_penalty_scalar
        self.safe_height_factor = safe_height_factor
        self.steps_num = steps_num
        self.steps_buffer = deque(maxlen=self.steps_num)
        self.epochs = epochs
        self.fresh_epoch = fresh_epoch
        self.save_epoch = save_epoch
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def add_memory(self, state, target_q):
        """
        state: (grid, feat)
        target: real q value
        """
        self.memory.append((state, target_q))

    def add_one_step_reward(self, next_q_value):
        state_old, _ = self.steps_buffer[0]
        discount_reward = 0.0
        for i in range(len(self.steps_buffer)):
            discount_reward += self.steps_buffer[i][1] * (self.gamma ** i)
        discount_reward += (self.gamma ** len(self.steps_buffer)) * next_q_value
        self.add_memory(state_old, torch.tensor(discount_reward, dtype=torch.float32))
        self.steps_buffer.popleft()


    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        batch_data = random.sample(self.memory, self.batch_size)

        batch_grids    = torch.stack([x[0][0] for x in batch_data])
        batch_feats    = torch.stack([x[0][1] for x in batch_data])
        batch_target_q = torch.stack([x[1]    for x in batch_data])

        if torch.cuda.is_available():
            batch_grids = batch_grids.cuda()
            batch_feats = batch_feats.cuda()
            batch_target_q = batch_target_q.cuda()

        # (batch_size, 1) -> (batch_size)
        predictions = self.train_model(batch_grids, batch_feats).squeeze(1)
        
        loss = self.criterion(predictions, batch_target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self):
        if torch.cuda.is_available():
            self.train_model.cuda()
            self.target_model.cuda()
        self.target_model.load_state_dict(self.train_model.state_dict())
        self.target_model.eval()

        recent_scores = deque(maxlen = self.window_size)
        max_avg_score = 0

        for epoch in range(self.epochs):
            self.env.reset()
            done = False
            steps = 0
            final_score = 0
            loss = 0
            total_loss = 0

            self.steps_buffer.clear()

            next_steps = self.env.get_next_states()

            while not done:
                # 采样
                next_actions = list(next_steps.keys())
                next_grids = torch.stack([v[0] for v in next_steps.values()])
                next_feats = torch.stack([v[1] for v in next_steps.values()])
                
                if torch.cuda.is_available():
                    next_grids = next_grids.cuda()
                    next_feats = next_feats.cuda()

                if random.random() <= self.epsilon:
                    index = random.randint(0, len(next_steps) - 1)
                else:
                    self.train_model.eval()
                    with torch.no_grad():
                        predictions = self.train_model(next_grids, next_feats).squeeze(1)
                        index = torch.argmax(predictions).item()
                    self.train_model.train()

                action = next_actions[index]
                current_max_height = next_feats[index][-1].item()
                current_state_save = (next_grids[index].cpu(), next_feats[index].cpu())

                score, done = self.env.step(action, render=False)

                reward = score / 10.0
                safe_threshold = self.safe_height_factor * self.env.height

                if not done:
                    if current_max_height > safe_threshold:
                        penalty = (current_max_height - safe_threshold) ** 2 * self.height_penalty_scalar
                        reward -= penalty
                else:
                    reward = -self.failed_penalty

                next_q_value = 0.0

                # 计算真实 q 值
                if done:
                    next_q_value = 0.0
                    next_steps = None # 游戏结束，没有下一步了
                    final_score = self.env.score
                else:
                    next_next_steps = self.env.get_next_states()
                    
                    n_grids = torch.stack([v[0] for v in next_next_steps.values()])
                    n_feats = torch.stack([v[1] for v in next_next_steps.values()])
                    
                    if torch.cuda.is_available():
                        n_grids = n_grids.cuda()
                        n_feats = n_feats.cuda()

                    # 实现 double DQN
                    # train_model 用来选 action
                    # target_model 用来评估 q value
                    self.train_model.eval()
                    with torch.no_grad():
                        next_preds_from_train = self.train_model(n_grids, n_feats).squeeze(1)
                        best_next_action_index = torch.argmax(next_preds_from_train).item()
                        next_preds_from_target = self.target_model(n_grids, n_feats).squeeze(1)
                        next_q_value = next_preds_from_target[best_next_action_index].item()
                    self.train_model.train()
                    
                    next_steps = next_next_steps

                self.steps_buffer.append((current_state_save, reward))

                if len(self.steps_buffer) == self.steps_num and not done:
                    self.add_one_step_reward(next_q_value)
                if done:
                    while len(self.steps_buffer) > 0:
                        self.add_one_step_reward(next_q_value)

                # 更新 train_model
                loss = self.train_step()
                if loss:
                    total_loss += loss
                steps += 1

            recent_scores.append(final_score)
            if len(recent_scores) >= self.window_size:
                avg_score = sum(recent_scores) / len(recent_scores)
                if avg_score > max_avg_score:
                    max_avg_score = avg_score
                    torch.save(self.train_model.state_dict(), f"{self.save_path}/DQN_best.pt")
                    print(f"Epoch {epoch}: New Max Avg Score: {max_avg_score:.2f} (Saved best_avg.pt)")

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            if epoch % self.fresh_epoch == 0:
                print(f'{epoch}/{self.epochs} {loss} {final_score} {sum(recent_scores) / len(recent_scores)} {len(self.memory)}/{self.memory_size}')
                if len(self.memory) > self.batch_size * 5:
                    self.memory.clear()
                # self.memory.clear()
                self.target_model.load_state_dict(self.train_model.state_dict())
            if epoch % self.save_epoch == 0:
                torch.save(self.train_model.state_dict(), f"{self.save_path}/DQN_{epoch}.pt")
