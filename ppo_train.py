import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple

# --- 超参数 ---
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
LR = 1e-3
MEMORY_SIZE = 10000
TARGET_UPDATE = 10
NUM_EPISODES = 500

# --- DQN 网络 ---
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# --- 经验回放 ---
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
    def __init__(self, capacity): self.memory = deque([], maxlen=capacity)
    def push(self, *args): self.memory.append(Transition(*args))
    def sample(self, batch_size): return random.sample(self.memory, batch_size)
    def __len__(self): return len(self.memory)

# --- 动作选择 ---
def select_action(state, steps_done, policy_net, env):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad(): return policy_net(state).max(1)[1].view(1, 1)
    else: return torch.tensor([[env.action_space.sample()]], dtype=torch.long)

# --- 优化模型 ---
def optimize_model(policy_net, target_net, memory, optimizer, loss_fn):
    if len(memory) < BATCH_SIZE: return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad(): next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# --- 主训练函数 ---
def train():
    # 1. 创建环境（不渲染，保证速度）
    env = gym.make("CartPole-v1")
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = DQN(n_observations, n_actions)
    target_net = DQN(n_observations, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    loss_fn = nn.MSELoss()
    memory = ReplayMemory(MEMORY_SIZE)

    steps_done = 0
    best_score = 0 # 记录最佳分数

    print("开始快速训练，期间不会显示画面...")
    for i_episode in range(NUM_EPISODES):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        total_reward = 0
        while True:
            action = select_action(state, steps_done, policy_net, env)
            steps_done += 1
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward], dtype=torch.float32)
            done = terminated or truncated
            next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0) if not terminated else None
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model(policy_net, target_net, memory, optimizer, loss_fn)
            if done:
                print(f'Episode {i_episode+1}, Total Reward: {total_reward}')
                
                # 2. 关键修改：保存表现最好的模型
                if total_reward > best_score:
                    best_score = total_reward
                    torch.save(policy_net.state_dict(), "best_model.pth")
                    print(f"  -> 新的最佳模型已保存！分数: {best_score}")
                
                break
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    print("训练完成！")
    env.close()

if __name__ == "__main__":
    train()