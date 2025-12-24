base DQN
```
hidden_dim=64
```
500 tests
41047.488

double DQN
```
hidden_dim=64
```
500 tests
61749.916

double DQN
```
height_penalty_scalar = 0.05
fresh_epoch = 20
epsilong_decay = 0.998
```
```
hidden_dim=64
```
500 tests
39094.24

double+PER DQN
```
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
epsilon_min=0.01,
epsilon_decay=0.995):
```
```
hidden_dim=256
```
500 tests
69752.622
420188
