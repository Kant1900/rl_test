import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 确保使用交互式后端
import math
from stable_baselines3.common.callbacks import EvalCallback
import os


# 全局变量设置
min_size = 1
upper_size = 10
start = np.array([1, 1], dtype=np.float32)  # 起点
goal = np.array([10, 10], dtype=np.float32)  # 目标点


obstacles = np.array([
    [5, 5, 1.5],
    [6, 8, 0.5],
    [8, 10, 0.5],
    [8, 2, 0.5],
    [10, 6, 1]
], dtype=np.float32)


obs_centers = obstacles[:, :2]
obs_radius = obstacles[:, 2]
move_step = 0.2

# 定义动作空间
actions = np.array([
    [0, 1],   # 上
    [0, -1],  # 下
    [1, 0],   # 右
    [-1, 0],  # 左
    [math.sqrt(2)/2, math.sqrt(2)/2],   # 右上
    [math.sqrt(2)/2, -math.sqrt(2)/2],  # 右下
    [-math.sqrt(2)/2, math.sqrt(2)/2],  # 左上
    [-math.sqrt(2)/2, -math.sqrt(2)/2]  # 左下
])    
n_actions = actions.shape[0]   #动作数量

# 自定义 Gymnasium 环境
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # 定义状态空间 [x, y]
        self.observation_space = spaces.Box(
            low=np.array([min_size, min_size], dtype=np.float32),
            high=np.array([upper_size, upper_size], dtype=np.float32),
            dtype=np.float32
        )
        # 定义动作空间（动作索引）
        self.action_space = spaces.Discrete(n_actions)

        self.step_count = 0
        self.max_steps = math.ceil(2*10*math.sqrt(2)/move_step)  # 合理的最大步数
        self.edge_threshold = 0.1  # 靠近边界的阈值
        self.eval_trajectory = []
        self.render_mode = "human"
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": 20 # 渲染帧率
        }

        # 初始化状态
        self.state = None  #(未初始化)
        # 初始化图形
        self.fig = None

    def reset(self, seed=None, options=None):
        """重置环境，返回初始状态和额外信息"""
        super().reset(seed=seed)
        self.state = start.copy()
        self.step_count = 0
        self.eval_trajectory = [self.state.copy()]
        return self.state, {}

    def step(self, action):
    ##
        global actions, move_step, goal, obs_centers, obs_radius, obstacles

        self.step_count += 1
        # 当前状态
        x, y = self.state
        dx, dy = actions[action] * move_step
        new_x = np.clip(x + dx, min_size, upper_size)
        new_y = np.clip(y + dy, min_size, upper_size)
        next_state = np.array([new_x, new_y], dtype=np.float32)

        # 累积轨迹数据
        self.eval_trajectory.append(next_state.copy())  # 添加新状态到轨迹中

        # 先检查所有终止条件
        done = False
        reward = -1  # 默认奖励

        # 1. 检查障碍物碰撞
        for i in range(len(obstacles)):
            if np.linalg.norm(next_state - obs_centers[i]) <= obs_radius[i]:
                reward = -100
                done = True
                self.state = next_state
                return next_state, reward, done, False, {}

        # 2. 检查是否到达目标
        if np.linalg.norm(next_state - goal) < 0.2:
            base_reward = 250
            step_penalty = self.step_count * 0.2
            reward = base_reward - step_penalty
            done = True
            self.state = next_state
            return next_state, reward, done, False, {}

        # 3. 检查是否超时
        if self.step_count >= self.max_steps:
            reward = -50
            done = True
            self.state = next_state
            return next_state, reward, done, False, {}

        # 4. 普通移动情况
        base_penalty = -1
        # step_penalty = self.step_count * 0.02
        distance_to_goal = np.linalg.norm(next_state - goal)

        # 找到最近边界距离
        dist_to_left = next_state[0] - min_size
        dist_to_right = upper_size - next_state[0]
        dist_to_bottom = next_state[1] - min_size
        dist_to_top = upper_size - next_state[1]
        min_edge_distance = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
        
        # 如果太靠近边界，给予惩罚
        if min_edge_distance < self.edge_threshold and np.linalg.norm(next_state - goal) > 1.0:
            edge_reward = float(-3.0 )
        else:
            # 适当鼓励在中心区域移动
            edge_reward = float(0.1)

        distance_reward = float(0.15 *(np.linalg.norm(start - goal)-distance_to_goal))   # 距离目标越近奖励越高
        reward = base_penalty + distance_reward +edge_reward
        done = False

        # 更新状态
        self.state = next_state
        return next_state, reward, done, False, {}
    

    def render(self):
        if self.render_mode is None:
            return None
        
        # 确保图形存在且活跃
        if not hasattr(self, '_fig') or not plt.fignum_exists(1):
            self._fig = plt.figure(1, figsize=(8, 8))
            plt.ion()  # 开启交互模式
            self._fig.show()
        
        plt.figure(1)
        plt.clf()
        
        # 绘制边界
        plt.gca().add_patch(plt.Rectangle(
            (min_size, min_size),
            upper_size - min_size,
            upper_size - min_size, 
            fill=False, edgecolor='k', linewidth=2
        ))

        # 障碍物
        theta = np.linspace(0, 2 * np.pi, 100)
        for i in range(len(obstacles)):
            x_obs = obstacles[i, 0] + obstacles[i, 2] * np.cos(theta)
            y_obs = obstacles[i, 1] + obstacles[i, 2] * np.sin(theta)
            plt.fill(x_obs, y_obs, 'r', alpha=0.5, label='Obstacles' if i == 0 else "")

        # 绘制起点、终点
        plt.plot(start[0], start[1], 'go', markersize=12, label='Start', markeredgecolor='black')
        plt.plot(goal[0], goal[1], 'b*', markersize=15, label='Goal', markeredgecolor='black')
        
        # 绘制智能体当前位置
        if self.state is not None:
            plt.plot(self.state[0], self.state[1], 'ro', markersize=10, label='Agent', markeredgecolor='black')
        
        # 绘制历史轨迹
        if hasattr(self, 'eval_trajectory') and len(self.eval_trajectory) > 1:
            state_array = np.array(self.eval_trajectory)
            plt.plot(state_array[:, 0], state_array[:, 1], 'b-', linewidth=2, alpha=0.7, label='Path')
            plt.plot(state_array[:, 0], state_array[:, 1], 'bo', markersize=3, alpha=0.5)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title(f"评估模式 (Step: {self.step_count})")
        plt.axis('equal')
        
        # 强制刷新显示
        plt.draw()
        plt.pause(0.001)
        if hasattr(self, '_fig'):
            self._fig.canvas.flush_events()
        
        if self.render_mode == "rgb_array":
            self._fig.canvas.draw()
            buf = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            return buf
        return None

# 初始化环境
env = CustomEnv()

# 检查环境是否符合 Gymnasium 规范（如果需要）
from stable_baselines3.common.env_checker import check_env
check_env(env)

# 创建评估环境（用于自动保存最佳模型）
eval_env = CustomEnv()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  
model_path = os.path.join(parent_dir, 'best_model')      # 最佳模型保存路径  改这里

# 设置评估回调，自动保存最佳模型
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path= model_path,     # 最佳模型保存路径
    log_path=os.path.join(parent_dir, 'logs'),                   # 评估日志路径
    eval_freq=20000,                      # 每2000步评估一次
    n_eval_episodes=1,                    # 每次评估5个episode
    deterministic=True,                   # 使用确定性动作
    render=True,                          # 评估时不渲染
    verbose=1                             # 显示评估信息
)

# 定义 DQN 模型
model = DQN(
    "MlpPolicy",  # 使用多层感知器网络
    env,
    learning_rate=1e-4,
    buffer_size=5_000_0,
    learning_starts=5000,
    batch_size=256,
    tau=1e-3,  # 目标网络平滑系数
    gamma=0.99,  # 折扣因子
    verbose=1,
    tensorboard_log=os.path.join(parent_dir, 'dqn_tensorboard1'),  # TensorBoard 日志路径
)


# 训练模型
max_episodes = 2000
model.learn(total_timesteps=max_episodes * 500 ,callback=eval_callback ) # 添加回调函数
model.save(os.path.join(parent_dir, 'dqn_path_agent')) # 保存最终模型


# 测试最佳模型（而不是最终模型）
print("\n测试最佳模型...")
try:
    # 加载自动保存的最佳模型
    model_path_save = os.path.join(model_path, 'best_model') 
    best_model = DQN.load(model_path_save)
    print(" 使用自动保存的最佳模型进行测试")
    test_model = best_model
except Exception as e:
    # 如果最佳模型不存在，使用最终模型
    print(f"无法加载最佳模型: {e}")
    print("使用最终模型进行测试")
    test_model = model
# 测试模型
state, _ = env.reset()
state_list = [state.copy()]
for _ in range(math.ceil(2*10*math.sqrt(2)/move_step)):
    action, _ = test_model.predict(state)
    state, _, done, _, _ = env.step(action)
    state_list.append(state.copy())
    if done:
        break

# 转换状态轨迹数组
state_list = np.array(state_list)

# 绘制路径
plt.figure(figsize=(8, 8))
plt.gca().add_patch(plt.Rectangle(
    (min_size, min_size),
    upper_size - min_size,
    upper_size - min_size, 
    fill=False, edgecolor='k', linewidth=2
))

# 障碍物
theta = np.linspace(0, 2 * np.pi, 100)
for i in range(len(obstacles)):
    x_obs = obs_centers[i, 0] + obs_radius[i] * np.cos(theta)
    y_obs = obs_centers[i, 1] + obs_radius[i] * np.sin(theta)
    plt.fill(x_obs, y_obs, 'r', alpha=0.5)

# 绘制起点、终点及路径
plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
plt.plot(goal[0], goal[1], 'bo', markersize=10, label='Goal')
plt.plot(state_list[:, 0], state_list[:, 1], 'b-', linewidth=2, label='Path')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()
plt.title("Path Traversed by Agent")
plt.show()