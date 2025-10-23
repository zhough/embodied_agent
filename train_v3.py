import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random
from collections import deque, namedtuple
import gymnasium as gym
import minatar  # 导入MinAtar环境（关键新增）
import swanlab
import os

# --- 超参数调整：适配MinAtar（训练步数减少，其他保留原版逻辑）---
BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.15
EPS_DECAY = 150000  # 探索率衰减不变，MinAtar学习更快
LR = 3e-4  
MEMORY_SIZE = 200000  # 经验池容量不变
TARGET_UPDATE = 4000  # 目标网络更新频率不变
NUM_STEPS = 200000  # 关键：MinAtar训练10万步足够出结果（原版40万）
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
PRINT_INTERVAL = 5  
LOG_INTERVAL = 1  

# --- DQN模型修改：适配MinAtar的10x10x3输入（核心修改）---
class DQN(nn.Module):
    def __init__(self, n_act):
        super().__init__()
        # 关键1：输入通道从4（原版帧堆叠）→3（MinAtar单帧3通道：球/挡板/砖块）
        self.conv1 = nn.Conv2d(3, 64, 8, 4)
        self.conv2 = nn.Conv2d(64, 128, 4, 2)
        self.conv3 = nn.Conv2d(128, 64, 3, 1)
        
        # 关键2：全连接层输入维度重新计算（MinAtar输入10x10，原版84x84）
        # 10x10 → conv1(核8,步4)→1x1 → conv2(核4,步2)→1x1 → conv3(核3,步1)→1x1
        self.fc1 = nn.Linear(64 * 1 * 1, 512)  # 64（conv3输出通道）×1×1
        self.fc2 = nn.Linear(512, n_act)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # 展平为(batch_size, 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --- 经验回放：完全复用原版逻辑（无需修改）---
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
    def __init__(self, capacity): self.memory = deque([], maxlen=capacity)
    def push(self, *args): self.memory.append(Transition(*args))
    def sample(self, batch_size): return random.sample(self.memory, batch_size)
    def __len__(self): return len(self.memory)


# --- 动作选择：完全复用原版逻辑（MinAtar动作数自动适配）---
def select_action(state, steps_done, policy_net, env):
    state = state.to(DEVICE)
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad(): 
            return policy_net(state).max(1)[1].view(1, 1).cpu()
    else: 
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long)


# --- 优化模型：完全复用原版逻辑（输入维度自动适配新模型）---
def optimize_model(policy_net, target_net, memory, optimizer, loss_fn, total_steps):
    if len(memory) < BATCH_SIZE: 
        return 0.0
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(DEVICE)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(DEVICE)
    state_batch = torch.cat(batch.state).to(DEVICE)
    action_batch = torch.cat(batch.action).to(DEVICE)
    reward_batch = torch.cat(batch.reward).to(DEVICE)
    
    # 模型推理适配新输入维度
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE).to(DEVICE)
    with torch.no_grad(): 
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# --- 主训练函数：修改环境初始化和状态预处理（核心修改）---
def train():
    print(f'开始MinAtar-Breakout训练, 设备:{DEVICE}')
    # SwanLab日志：复用原版逻辑
    swanlab.login(api_key="Nj75sPpgjdzUONcpKxlg6")
    swanlab.init(
        project='Breakout_DQN',
        experiment_name="MinAtar_train",  # 区分原版实验
    )

    # --- 关键1：替换为MinAtar环境（10x10x3输入，3个有效动作）---
    env = gym.make("minatar/Breakout-v0")  # MinAtar-Breakout默认3动作：左/右/无操作
    n_actions = env.action_space.n  # 自动获取动作数（固定为3）
    print(f"MinAtar环境动作数：{n_actions}（左/右/无操作）")

    # --- 模型初始化：使用修改后的DQN（输入3通道）---
    policy_net = DQN(n_actions).to(DEVICE)
    target_net = DQN(n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEMORY_SIZE)

    # --- 统计变量：复用原版逻辑---
    episode_rewards = []
    episode_losses = []
    total_steps = 0
    episode_count = 0

    # --- 主训练循环：修改状态预处理（删除帧差/帧堆叠）---
    while total_steps < NUM_STEPS:
        episode_count += 1
        current_episode_reward = 0
        current_episode_loss = 0.0
        optimize_count = 0

        # --- 新回合初始化：MinAtar状态直接预处理（无需帧差/堆叠）---
        frame, info = env.reset()  # MinAtar输出：(10,10,3) numpy数组（HWC格式）
        # 关键2：MinAtar状态预处理（HWC→CHW，转tensor，补batch维度）
        # 原因：PyTorch卷积层要求输入格式为(batch_size, channels, height, width)
        state = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1,3,10,10)

        while True:
            # --- 动作选择：复用原版逻辑---
            action = select_action(state, total_steps, policy_net, env)
            eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * total_steps / EPS_DECAY)
            total_steps += 1

            # --- 执行动作：MinAtar环境step输出与原版一致---
            next_frame, reward, terminated, truncated, _ = env.step(action.item())
            current_episode_reward += reward
            reward = torch.tensor([reward], dtype=torch.float32)  # 存CPU，抽样时转GPU
            done = terminated or truncated

            # --- 下一状态预处理：与初始状态逻辑一致（无帧堆叠）---
            if not terminated:
                next_state = torch.tensor(next_frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            else:
                next_state = None  # 终止状态设为None，复用原版经验回放逻辑

            # --- 存储经验+优化模型：完全复用原版逻辑---
            memory.push(state, action, next_state, reward)
            loss = optimize_model(policy_net, target_net, memory, optimizer, nn.MSELoss(), total_steps)
            if loss > 0:
                current_episode_loss += loss
                optimize_count += 1

            # --- 更新状态---
            state = next_state

            # --- SwanLab日志：复用原版逻辑---
            swanlab.log({'train/Epsilon': eps_threshold}, step=total_steps)

            # --- 回合结束处理：复用原版逻辑---
            if done:
                avg_episode_loss = current_episode_loss / optimize_count if optimize_count > 0 else 0.0
                episode_rewards.append(current_episode_reward)
                episode_losses.append(avg_episode_loss)

                # 记录回合得分和损失
                if episode_count % LOG_INTERVAL == 0:
                    swanlab.log({"train/Episode_Reward": current_episode_reward}, step=episode_count)
                    swanlab.log({"train/Episode_Avg_Loss": avg_episode_loss}, step=episode_count)

                # 打印统计信息
                if episode_count % PRINT_INTERVAL == 0:
                    avg_reward = np.mean(episode_rewards[-PRINT_INTERVAL:])
                    avg_loss = np.mean(episode_losses[-PRINT_INTERVAL:])
                    print(f"=== Episode {episode_count:5d} | Total Steps {total_steps:7d} ===")
                    print(f"Epsilon: {eps_threshold:.4f} | Avg Reward: {avg_reward:.2f} | Avg Loss: {avg_loss:.4f}")
                    print(f"Memory Size: {len(memory):7d} | Target Update: {total_steps // TARGET_UPDATE:5d}\n")

                break

            # --- 目标网络更新+模型保存：完全复用原版逻辑---
            if total_steps % TARGET_UPDATE == 0:
                print('更新target_net')
                torch.cuda.empty_cache()
                target_net.load_state_dict(policy_net.state_dict())

                # 保存训练状态
                save_dir = "models_minatar"  # 单独目录，区分原版模型
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                checkpoint_save_path = os.path.join(save_dir, "training_checkpoint_minatar.pt")
                torch.save({
                    "policy_net_state_dict": policy_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "total_steps": total_steps,
                    "episode_count": episode_count,
                    "eps_threshold": eps_threshold
                }, checkpoint_save_path)
                print(f"MinAtar训练状态保存路径：{checkpoint_save_path}")

    env.close()
    print("MinAtar-Breakout训练结束！")


if __name__ == "__main__":
    train()