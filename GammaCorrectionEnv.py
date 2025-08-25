import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
from collections import deque
import random


# 修改后的环境定义 - 支持多图像处理
class GlobalGammaCorrectionEnv:
    def __init__(self):
        self.original_image = None
        self.image_path = None
        self.reset()

    def load_image(self, image_path):
        """加载新图像到环境"""
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Image not found at {image_path}")
        return self.reset()

    def reset(self):
        """重置环境状态"""
        if self.original_image is None:
            raise ValueError("No image loaded in environment")
        self.processed_image = self.original_image.copy()
        return self._get_state()

    def _get_state(self):
        """提取整个图像特征"""
        if len(self.original_image.shape) == 3:  # 彩色图像转灰度
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.original_image

        # 全局图像特征
        brightness = np.mean(gray_image)
        contrast = np.std(gray_image)
        hist = cv2.calcHist([gray_image], [0], None, [8], [0, 256]).flatten()
        hist = hist / hist.sum()  # 归一化

        # 组合状态: 亮度 + 对比度 + 直方图
        state = np.concatenate([
            [brightness / 255.0, contrast / 128.0],
            hist
        ])
        return state.astype(np.float32)

    def _apply_gamma(self, gamma):
        """对整个图像应用gamma校正"""
        corrected = np.power(self.original_image / 255.0, gamma) * 255.0
        self.processed_image = np.clip(corrected, 0, 255).astype(np.uint8)
        return self.processed_image

    def _calculate_reward(self, corrected_image):
        """图像质量评估指标"""
        if len(corrected_image.shape) == 3:
            corrected_gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
        else:
            corrected_gray = corrected_image

        # 1. 亮度评估 (目标亮度范围[50, 200])
        brightness = np.mean(corrected_gray)
        brightness_reward = -np.abs(brightness - 125) / 125.0

        # 2. 对比度评估
        contrast = np.std(corrected_gray)
        contrast_reward = np.tanh(contrast / 50.0)  # 使用tanh归一化

        # 3. 细节保留 (使用拉普拉斯方差)
        laplacian = cv2.Laplacian(corrected_gray, cv2.CV_64F).var()
        detail_reward = np.tanh(laplacian / 1000.0)

        # 综合奖励
        reward = brightness_reward * 0.4 + contrast_reward * 0.4 + detail_reward * 0.2
        return float(reward)

    def step(self, action):
        """执行一步动作"""
        gamma = np.clip(action, 0.1, 3.0)[0]  # 限制gamma范围
        corrected_image = self._apply_gamma(gamma)
        reward = self._calculate_reward(corrected_image)

        # 单步完成整个图像处理
        done = True
        next_state = None

        return next_state, reward, done, self.processed_image, gamma  # 返回gamma值


# Actor 网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # 输出范围[-1, 1]
        )

    def forward(self, state):
        return self.net(state) * 1.45 + 1.55  # 映射到[0.1, 3.0]


# Critic 网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)


# TD3 Agent
class TD3Agent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = deque(maxlen=100000)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = 256
        self.gamma = 0.99  # 折扣因子
        self.tau = 0.0007  # 软更新系数
        self.policy_noise = 0.1
        self.noise_clip = 0.5
        self.policy_delay = 6
        self.update_count = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()
        # 添加探索噪声
        noise = np.random.normal(0, 0.1, size=self.action_dim)
        return np.clip(action + noise, 0.1, 3.0)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # 从回放缓冲区采样
        batch = random.sample(self.replay_buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state))
        action = torch.FloatTensor(np.array(action))
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(np.array([s for s in next_state if s is not None]))
        done = torch.FloatTensor(done).unsqueeze(1)

        # 计算目标Q值
        with torch.no_grad():
            noise = torch.randn_like(action) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)

            next_action = self.actor_target(next_state) + noise
            next_action = torch.clamp(next_action, 0.1, 3.0)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # 更新Critic
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 延迟策略更新
        self.update_count += 1
        if self.update_count % self.policy_delay == 0:
            # 更新Actor
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# 训练函数 - 支持多图像
def train_td3_global_multiple(image_paths, episodes=100, output_dir="results"):
    """训练函数，支持多个图像路径"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建环境和智能体
    env = GlobalGammaCorrectionEnv()
    state_dim = 10  # 状态维度 (2个统计量 + 8个直方图bin)
    action_dim = 1
    agent = TD3Agent(state_dim, action_dim)

    # 初始化最佳记录
    best_gamma = 1.0
    best_reward = -float('inf')
    best_image_path = None
    gamma_history = []
    reward_history = []

    # 训练循环
    for episode in range(episodes):
        # 随机选择一张图像
        image_path = random.choice(image_paths)
        image_name = os.path.basename(image_path)

        # 加载图像到环境
        state = env.load_image(image_path)

        # 选择并执行动作
        action = agent.select_action(state)
        _, reward, done, processed_img, gamma = env.step(action)

        # 存储经验并训练
        agent.store_transition(state, action, reward, None, done)
        agent.train()

        # 跟踪最佳gamma值
        gamma_history.append((image_name, gamma))
        reward_history.append(reward)

        if reward > best_reward:
            best_reward = reward
            best_gamma = gamma
            best_image_path = image_path
            # 保存最佳处理图像
            cv2.imwrite(os.path.join(output_dir, f"best_result_{image_name}"), processed_img)

        # 每10轮打印进度
        if episode % 10 == 0:
            print(f"Episode {episode}, Image: {image_name}, Gamma: {gamma:.3f}, Reward: {reward:.4f}")
            # 保存处理后的图像
            cv2.imwrite(os.path.join(output_dir, f"processed_{episode}_{image_name}"), processed_img)

    # 保存模型和最佳gamma值
    torch.save(agent.actor.state_dict(), os.path.join(output_dir, "global_gamma_correction_actor.pth"))

    # 保存gamma历史记录
    with open(os.path.join(output_dir, "gamma_history.txt"), "w") as f:
        for img, gamma_val in gamma_history:
            f.write(f"{img}: {gamma_val:.4f}\n")

    # 保存奖励历史记录
    np.save(os.path.join(output_dir, "reward_history.npy"), np.array(reward_history))

    print(f"Training complete. Best gamma: {best_gamma:.4f} from image: {os.path.basename(best_image_path)}")
    return agent, best_gamma, gamma_history


# 应用函数 - 输出全局gamma
def apply_global_gamma(image_path, model_path):
    """应用训练好的模型到新图像"""
    # 创建环境
    env = GlobalGammaCorrectionEnv()

    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")

    # 初始化环境
    state = env.load_image(image_path)

    # 加载模型
    state_dim = 10
    action_dim = 1
    actor = Actor(state_dim, action_dim)
    actor.load_state_dict(torch.load(model_path))
    actor.eval()

    # 预测Gamma值
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        action = actor(state_tensor).numpy()

    # 应用校正
    _, _, _, processed_img, gamma = env.step(action)

    print(f"Applied global gamma: {gamma[0]:.4f}")
    return processed_img, gamma[0]


# 使用示例
if __name__ == "__main__":
    # 设置图像文件夹路径
    image_folder = "/train/low/"
    output_dir = "training_results"

    # 获取所有图像路径
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_paths:
        raise ValueError("No images found in the specified folder")

    print(f"Found {len(image_paths)} images for training")

    # 训练模型并获取最佳gamma
    trained_agent, best_gamma, gamma_history = train_td3_global_multiple(
        image_paths,
        episodes=200,
        output_dir=output_dir
    )

    # 输出最佳全局gamma值
    print(f"Optimal global gamma value: {best_gamma:.4f}")

    # 应用训练好的模型到新图像
    test_image = "test_image.jpg"
    result_image, applied_gamma = apply_global_gamma(test_image,
                                                     os.path.join(output_dir, "global_gamma_correction_actor.pth"))

    # 保存结果
    cv2.imwrite(os.path.join(output_dir, "final_result.jpg"), result_image)

    # 保存gamma值到文件
    with open(os.path.join(output_dir, "gamma_value.txt"), "w") as f:
        f.write(f"{applied_gamma:.4f}")

    print("Processing complete. Results saved in", output_dir)

import torch
from torch import nn
from torchinfo import summary
from thop import profile, clever_format


# 定义网络结构（与原始代码相同）
class Actor(nn.Module):
    def __init__(self, state_dim=10, action_dim=1):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state) * 1.45 + 1.55  # 映射到[0.1, 3.0]


class Critic(nn.Module):
    def __init__(self, state_dim=10, action_dim=1):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)


# 计算参数量 (Params) 和 FLOPs
def calculate_complexity():
    # 创建模型实例
    actor = Actor()
    critic = Critic()

    # 打印模型结构
    print("=" * 50)
    print("Actor Network Structure:")
    print(actor)

    print("\n" + "=" * 50)
    print("Critic Network Structure:")
    print(critic)

    # 计算Actor的参数量和FLOPs
    print("\n" + "=" * 50)
    print("Actor Network Analysis:")
    input_size = (1, 10)  # (batch_size, state_dim)
    summary(actor, input_size=input_size)

    # 计算Critic的参数量和FLOPs
    print("\n" + "=" * 50)
    print("Critic Network Analysis:")
    state_input = (1, 10)  # (batch_size, state_dim)
    action_input = (1, 1)  # (batch_size, action_dim)

    # 使用thop计算FLOPs
    state_tensor = torch.randn(*state_input)
    action_tensor = torch.randn(*action_input)
    flops, params = profile(critic, inputs=(state_tensor, action_tensor))
    flops, params = clever_format([flops, params])

    # 打印结果
    print(f"Total Parameters (Params): {params}")
    print(f"Total FLOPs: {flops}")


if __name__ == "__main__":

    calculate_complexity()
