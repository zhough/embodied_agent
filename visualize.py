import gymnasium as gym
import torch
import numpy as np

# --- 必须与训练时使用的网络结构完全相同 ---
class DQN(torch.nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(n_observations, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, n_actions)
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

def visualize():
    # 1. 创建环境，并开启渲染
    env = gym.make("CartPole-v1", render_mode="human")
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # 2. 初始化模型
    policy_net = DQN(n_observations, n_actions)

    # 3. 关键步骤：加载训练好的模型权重
    try:
        policy_net.load_state_dict(torch.load("best_model.pth"))
        print("成功加载模型 'best_model.pth'")
    except FileNotFoundError:
        print("错误：找不到模型文件 'best_model.pth'。请先运行训练脚本。")
        return

    # 4. 设置为评估模式
    policy_net.eval()

    # 5. 运行评估循环
    print("开始可视化评估，将弹出一个窗口...")
    episodes_to_play = 5 # 你想可视化几个回合
    for i in range(episodes_to_play):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        total_reward = 0
        while True:
            # 在评估时，我们总是选择最优动作，不进行探索
            with torch.no_grad():
                action = policy_net(state).max(1)[1].view(1, 1)
            
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            
            if terminated or truncated:
                print(f'评估回合 {i+1}, 最终奖励: {total_reward}')
                break
            
            state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

    print("可视化结束。")
    env.close()

if __name__ == "__main__":
    visualize()