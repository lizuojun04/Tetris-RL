base DQN
500 tests
41047.488

double DQN
500 tests
about 59000

double DQN
```
height_penalty_scalar = 0.05
fresh_epoch = 20
epsilong_decay = 0.998
```
500 tests
39094.24

double+multi-steps DQN
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
failed_penalty = -10.0,
height_penalty_scalar=0.1,
safe_height_factor=0.5,
steps_num=1,
epochs=3000,
fresh_epoch=10,
save_epoch=50,
epsilon=1.0,
epsilon_min=0.01,
epsilon_decay=0.995
```
500 tests
58068.592
