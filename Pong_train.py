import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import gymnasium as gym
import ale_py
import shimmy  # 导入以确保包装器注册
import swanlab
import os

gym.register_envs(ale_py)
# --- 超参数 ---
BATCH_SIZE = 8192
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 100000
LR = 3e-5  
MEMORY_SIZE = 50000  # 经验回放池
TARGET_UPDATE = 2000  # 通常按步数更新，而不是按回合
NUM_STEPS = 100000  # 训练总步数
FRAME_SIZE = 4
PRINT_INTERVAL = 5  
LOG_INTERVAL = 1  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DQN(nn.Module):
    def __init__(self,n_act):
        super().__init__()
        self.conv1 = nn.Conv2d(FRAME_SIZE,32,8,4)
        self.conv2 = nn.Conv2d(32,64,4,2)
        self.conv3 = nn.Conv2d(64,64,3,1)
        
        self.fc1 = nn.Linear(64*7*7,512)
        self.fc2 = nn.Linear(512,n_act)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray,(84,84),interpolation=cv2.INTER_AREA)
    normalized = resized/255.0
    return normalized[np.newaxis,...]

# 帧堆叠：将最近4帧合并为一个状态
class FrameStacker:
    def __init__(self, stack_size=FRAME_SIZE):
        self.stack_size = stack_size
        self.stack = []

    def reset(self, first_frame):
        # 初始帧：重复stack_size次（比如4次），组成初始状态
        self.stack = [first_frame] * self.stack_size
        return np.concatenate(self.stack, axis=0)  # 输出：(4,84,84)

    def step(self, new_frame):
        # 新帧入栈，最旧帧出栈
        self.stack.pop(0)
        self.stack.append(new_frame)
        return np.concatenate(self.stack, axis=0)  # 输出：(4,84,84)
    
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
        with torch.no_grad(): 
            return policy_net(state).max(1)[1].view(1, 1).to(DEVICE)
    else: 
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long).to(DEVICE)

# --- 优化模型 ---
def optimize_model(policy_net, target_net, memory, optimizer, loss_fn):
    if len(memory) < BATCH_SIZE: 
        return 0.0
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(DEVICE)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
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

def train():
    # 可视化新增：初始化TensorBoard写入器（日志保存在"runs/Pong_DQN"目录）
    #writer = SummaryWriter(log_dir="runs/Pong_DQN")
    print(f'开始训练,设备:{DEVICE}')
    swanlab.login(api_key="Nj75sPpgjdzUONcpKxlg6")
    swanlab.init(
        project='Breakout_DQN',
        experiment_name="train",
    )
    # --- 初始化 ---
    env = gym.make("ALE/Breakout-v5")
    n_actions = env.action_space.n
    policy_net = DQN(n_actions).to(DEVICE)
    target_net = DQN(n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEMORY_SIZE)
    frame_stacker = FrameStacker(stack_size=4)

    # checkpoint = torch.load("models/training_checkpoint_final.pt", map_location=DEVICE)
    # policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # total_steps = checkpoint["total_steps"]  # 恢复总步数
    # episode_count = checkpoint["episode_count"]  # 恢复回合数
    # eps_threshold = checkpoint["eps_threshold"]  # 恢复探索率
    # 可视化新增：初始化统计变量
    episode_rewards = []  # 记录每回合得分
    episode_losses = []   # 记录每回合平均损失
    total_steps = 0       # 记录总步数
    episode_count = 0     # 记录回合数
    #last_target_update = 0

    # --- 主训练循环（按总步数） ---
    while total_steps < NUM_STEPS:
        episode_count += 1
        current_episode_reward = 0  # 可视化新增：当前回合得分
        current_episode_loss = 0.0  # 可视化新增：当前回合总损失
        optimize_count = 0          # 可视化新增：当前回合优化次数

        # --- 新回合初始化 ---
        frame, info = env.reset()
        processed_frame = preprocess_frame(frame)
        state = frame_stacker.reset(processed_frame)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        while True:
            # --- 选择动作 & 计算探索率 ---
            action = select_action(state, total_steps, policy_net, env)
            eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * total_steps / EPS_DECAY)  # 可视化新增
            total_steps += 1

            # --- 执行动作 ---
            next_frame, reward, terminated, truncated, _ = env.step(action.item())
            current_episode_reward += reward  # 可视化新增：累加回合得分
            reward = torch.tensor([reward], dtype=torch.float32).to(DEVICE)
            done = terminated or truncated

            # --- 预处理下一帧 ---
            processed_next_frame = preprocess_frame(next_frame)
            next_state_stacked = frame_stacker.step(processed_next_frame)
            next_state = torch.tensor(next_state_stacked, dtype=torch.float32).unsqueeze(0).to(DEVICE) if not terminated else None

            # --- 存储经验 & 优化模型 ---
            memory.push(state, action, next_state, reward)
            loss = optimize_model(policy_net, target_net, memory, optimizer, nn.MSELoss())  # 可视化新增：获取损失值
            if loss > 0:  # 可视化新增：只累加有效损失（避免未优化时的0）
                current_episode_loss += loss
                optimize_count += 1

            # --- 更新状态 ---
            state = next_state

            # --- 可视化新增：每步记录探索率（TensorBoard） ---
            #writer.add_scalar('Training/Epsilon', eps_threshold, total_steps)
            swanlab.log({'train/Epsilon':eps_threshold},step=total_steps)

            # --- 检查回合结束 ---
            if done:
                # 可视化新增：计算当前回合平均损失
                avg_episode_loss = current_episode_loss / optimize_count if optimize_count > 0 else 0.0
                episode_rewards.append(current_episode_reward)
                episode_losses.append(avg_episode_loss)

                # 可视化新增：每回合记录得分和平均损失（TensorBoard）
                if episode_count % LOG_INTERVAL == 0:
                # 记录每回合得分（每回合调用一次，可不设置 steps，X 轴自动为回合数）
                    swanlab.log({"train/Episode_Reward":current_episode_reward}, step=episode_count)
                    # 记录每回合平均损失（同理，用 episode_count 作为 steps，与回合数对齐）
                    swanlab.log({"train/Episode_Avg_Loss":avg_episode_loss}, step=episode_count)
                    #writer.add_scalar('Training/Episode_Reward', current_episode_reward, episode_count)
                    #writer.add_scalar('Training/Episode_Avg_Loss', avg_episode_loss, episode_count)

                # 可视化新增：每PRINT_INTERVAL回合打印统计信息
                if episode_count % PRINT_INTERVAL == 0:
                    avg_reward = np.mean(episode_rewards[-PRINT_INTERVAL:])  # 最近10回合平均得分
                    avg_loss = np.mean(episode_losses[-PRINT_INTERVAL:])    # 最近10回合平均损失
                    print(f"=== Episode {episode_count:5d} | Total Steps {total_steps:7d} ===")
                    print(f"Epsilon: {eps_threshold:.4f} | Avg Reward: {avg_reward:.2f} | Avg Loss: {avg_loss:.4f}")
                    print(f"Memory Size: {len(memory):7d} | Target Update: {total_steps % TARGET_UPDATE:5d}\n")

                break

            # --- 更新目标网络 ---
            if total_steps % TARGET_UPDATE == 0:
                print('更新target_net')
                target_net.load_state_dict(policy_net.state_dict())
                # 可视化新增：记录目标网络更新（TensorBoard）
                #writer.add_scalar('Training/Target_Network_Update', total_steps, total_steps)
                #swanlab.log({"train/Target_Network_Update":total_steps}, step=total_steps)
                #保存模型参数
                
                save_dir = "models"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                # 保存模型权重（核心）
                # model_save_path = os.path.join(save_dir, "policy_net_final.pt")
                # torch.save(policy_net.state_dict(), model_save_path)
                # print(f"最终模型权重保存路径：{model_save_path}")
                
                # （可选）保存完整训练状态（断点续训用）
                checkpoint_save_path = os.path.join(save_dir, "training_checkpoint_final.pt")
                torch.save({
                    "policy_net_state_dict": policy_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "total_steps": total_steps,
                    "episode_count": episode_count,
                    "eps_threshold": eps_threshold
                }, checkpoint_save_path)
                print(f"完整训练状态保存路径：{checkpoint_save_path}")

    env.close()
    print("训练结束！")

if __name__ == "__main__":
    train()