import torch
import torch.nn as nn
import random
from train.prioritized_replay import PrioritizedReplayBuffer
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
        height_penalty_scalar=0.1,
        epochs=5000,
        fresh_epoch=10,
        save_epoch=50,
        epsilon=1.0,
        epsilon_min=0.01, # 这里增大效果会更好吗
        epsilon_decay=0.995):
        self.save_path=save_path
        self.env = env
        self.train_model = train_model
        self.target_model = target_model
        self.optimizer = optimizer
        self.criterion = criterion
        self.memory = PrioritizedReplayBuffer(capacity=memory_size)
        self.batch_size = batch_size
        self.window_size = window_size
        self.gamma = gamma
        self.height_penalty_scalar = height_penalty_scalar
        self.epochs = epochs
        self.fresh_epoch = fresh_epoch
        self.save_epoch = save_epoch
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    def add_memory(self, state, target_q):
        """
        state: (grid, feat)
        target: real q value
        """
        sample = (state, target_q)
        self.memory.add(error=1.0, sample=sample)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        batch_data, idxs, is_weights = self.memory.sample(self.batch_size)

        is_weights = torch.FloatTensor(is_weights)

        batch_grids    = torch.stack([x[0][0] for x in batch_data])
        batch_feats    = torch.stack([x[0][1] for x in batch_data])
        batch_target_q = torch.stack([x[1]    for x in batch_data])

        if torch.cuda.is_available():
            batch_grids = batch_grids.cuda()
            batch_feats = batch_feats.cuda()
            batch_target_q = batch_target_q.cuda()
            is_weights = is_weights.cuda()

        # (batch_size, 1) -> (batch_size)
        predictions = self.train_model(batch_grids, batch_feats).squeeze(1)

        loss_elementwise = (predictions - batch_target_q)  ** 2
        loss = (loss_elementwise * is_weights).mean()

        # loss = self.criterion(predictions, batch_target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.train_model.parameters(), 1.0)
        self.optimizer.step()

        td_errors = torch.sqrt(loss_elementwise).detach().cpu().numpy()
        self.memory.update(idxs, td_errors)

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
                safe_threshold = 1 / 2 * self.env.height

                if not done:
                    if current_max_height > safe_threshold:
                        penalty = (current_max_height - safe_threshold) ** 2 * self.height_penalty_scalar
                        reward -= penalty
                else:
                    reward = -50.0

                # 计算真实 q 值
                if done:
                    target_q = reward
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
                    """
                    self.train_model.eval()
                    with torch.no_grad():
                        next_preds_from_train = self.train_model(n_grids, n_feats).squeeze(1)
                        best_next_action_index = torch.argmax(next_preds_from_train).item()
                    self.train_model.train()

                    with torch.no_grad():
                        next_preds_from_target = self.target_model(n_grids, n_feats).squeeze(1)
                        max_q = next_preds_from_target[best_next_action_index].item()
                    """
                    with torch.no_grad():
                        next_preds_from_target = self.target_model(n_grids, n_feats).squeeze(1)
                        max_q = torch.max(next_preds_from_target).item()

                    # 加上缩放
                    target_q = reward + self.gamma * max_q
                    next_steps = next_next_steps
                    
                # 储存 q 值
                target_q = torch.tensor(target_q, dtype=torch.float32)
                self.add_memory(current_state_save, target_q)

                # 更新 train_model
                loss = self.train_step()
                if loss:
                    total_loss += loss
                steps += 1

            if len(self.memory) >= self.batch_size:
                self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            recent_scores.append(final_score)
            if len(recent_scores) >= self.window_size:
                avg_score = sum(recent_scores) / len(recent_scores)
                if avg_score > max_avg_score:
                    max_avg_score = avg_score
                    torch.save(self.train_model.state_dict(), f"{self.save_path}/DQN_best.pt")
                    # print(f"==>Epoch {epoch}: New Max Avg Score: {max_avg_score:.2f} (Saved best_avg.pt)")

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            if epoch % self.fresh_epoch == 0:
                avg_score = 0
                for num in range(self.fresh_epoch):
                    avg_score += recent_scores[-(num + 1)]
                avg_score /= self.fresh_epoch
                # print(f'Epoch: {epoch}/{self.epochs} | Loss: {loss} | Eps: {self.epsilon} | Avg score: {sum(recent_scores) / len(recent_scores)} | Final score: {final_score} | lr: {current_lr}')
                print(f'{epoch} {loss} {self.epsilon} {avg_score} {final_score} {current_lr}')
                if len(self.memory) > 5 * self.batch_size:
                    self.memory.clear()
                self.target_model.load_state_dict(self.train_model.state_dict())
            if epoch % self.save_epoch == 0:
                torch.save(self.train_model.state_dict(), f"{self.save_path}/DQN_{epoch}.pt")
