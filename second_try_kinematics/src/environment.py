import matplotlib.pyplot as plt
import matplotlib
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class PathPlanningEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # 环境参数
        self.min_size = 1
        self.upper_size = 10
        self.start = np.array([1.5, 1.5], dtype=np.float32)
        self.goal = np.array([10, 10], dtype=np.float32)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.obstacles = np.array([
            [5, 5, 1.5], 
            [6, 8, 0.5], 
            [8, 10, 0.5],
            [8, 2, 0.5], 
            [10, 6, 1]
        ], dtype=np.float32)
        self.obs_centers = self.obstacles[:, :2]
        self.obs_radius = self.obstacles[:, 2]

        
        # 动作空间  规定加速度
        self.actions = np.array([
            [0,0],
            [0, 1], 
            [0, -1], 
            [1, 0], 
            [-1, 0],
        ])
        self.n_actions = self.actions.shape[0]
        
        # 物理参数
        self.dt = 0.1  # 时间步长
        self.max_speed = 2.0
        self.max_acceleration = 1 # 最大加速度

            # 定义状态空间 [x, y]
        self.observation_space = spaces.Box(
            low=np.array([self.min_size, self.min_size], dtype=np.float32),
            high=np.array([self.upper_size, self.upper_size], dtype=np.float32),
            dtype=np.float32
        )

        # 定义动作空间（动作索引）
        self.action_space = spaces.Discrete(self.n_actions)

        self.step_count = 0
        
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
        self.state = self.start.copy()
        self.step_count = 0
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.eval_trajectory = [self.state.copy()]
        return self.state, {}
    
    def _calculate_repulsive_punishment_ (self, position):
        """计算位置处的斥力"""
        influence_distance = 2.0  # 斥力影响距离
        max_repulsive = 2.0  # 最大斥力惩罚
        min_distance = 0.3  # 最小距离，防止除零
        repulsive_punishment = 0.0
        a = (influence_distance - min_distance) / (min_distance ** 2)  # 非常近时，给予最大斥力
        for i in range(len(self.obstacles)):
            obs_center = self.obs_centers[i]
            obs_radius = self.obs_radius[i]
            direction = position - obs_center
            distance = np.linalg.norm(direction) - obs_radius
            force_magnitude = 0.0

            if distance < influence_distance and distance > 0:
                
                if distance > min_distance:
                    force_magnitude = (influence_distance - distance) / (distance ** 2)
                else:
                    force_magnitude = a
                force_magnitude = max_repulsive *((distance)/(a-distance))
            repulsive_punishment += force_magnitude 
        
        return repulsive_punishment 


    def step(self, action):
        # 实现步进逻辑
        self.step_count += 1
        x, y = self.state
        acceleration = self.actions[action] * self.max_acceleration
        self.velocity += acceleration * self.dt
        speed = np.linalg.norm(self.velocity)

        # 限制最大速度
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed

        dx, dy =  self.velocity * self.dt   #x和y方向的位移
        new_x = np.clip(x + dx, self.min_size, self.upper_size)
        new_y = np.clip(y + dy, self.min_size, self.upper_size)
        next_state = np.array([new_x, new_y], dtype=np.float32)

        distance_to_goal = np.linalg.norm(next_state - self.goal)
        distance_improvement = np.linalg.norm(self.state - self.goal) - distance_to_goal

        self.eval_trajectory.append(next_state.copy())  # 添加新状态到轨迹中

        # 先检查所有终止条件
        done = False
        reward = -1  # 默认奖励

        # 1. 检查障碍物碰撞
        for i in range(len(self.obstacles)):
            if np.linalg.norm(next_state - self.obs_centers[i]) <= self.obs_radius[i]:
                reward = -100
                done = True
                self.state = next_state
                return next_state, reward, done, False, {}

        # 2. 检查是否到达目标
        if np.linalg.norm(next_state - self.goal) < 0.6:
            base_reward = 250
            step_penalty = self.step_count * 0.2
            reward = base_reward - step_penalty
            done = True
            self.state = next_state
            return next_state, reward, done, False, {}
        
        # 3. 检查是否超时
        if self.step_count >= 200:
            reward = -100
            done = True
            self.state = next_state
            return next_state, reward, done, False, {}

         # 4. 普通移动情况
        base_penalty = -0.5
        

        # 找到最近边界距离
        dist_to_left = next_state[0] - self.min_size
        dist_to_right = self.upper_size - next_state[0]
        dist_to_bottom = next_state[1] - self.min_size
        dist_to_top = self.upper_size - next_state[1]
        min_edge_distance = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
        
        # 如果太靠近边界，给予惩罚
        if min_edge_distance < self.edge_threshold and np.linalg.norm(next_state - self.goal) > 1.0:
            done = True
            self.state = next_state
            reward = -50
            return next_state, reward, done, False, {}
        
        # distance_reward = float(0.9 *(np.linalg.norm(self.start - self.goal)-distance_to_goal))   # 距离目标越近奖励越高
        distance_reward = float(distance_improvement * 5) # 奖励与距离改善成正比

        repulsive_punishment= self._calculate_repulsive_punishment_(next_state)

        reward = base_penalty + distance_reward - repulsive_punishment
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
            (self.min_size, self.min_size),
            self.upper_size - self.min_size,
            self.upper_size - self.min_size, 
            fill=False, edgecolor='k', linewidth=2
        ))

        # 障碍物
        theta = np.linspace(0, 2 * np.pi, 100)
        for i in range(len(self.obstacles)):
            x_obs = self.obstacles[i, 0] + self.obstacles[i, 2] * np.cos(theta)
            y_obs = self.obstacles[i, 1] + self.obstacles[i, 2] * np.sin(theta)
            plt.fill(x_obs, y_obs, 'r', alpha=0.5, label='Obstacles' if i == 0 else "")

        # 绘制起点、终点
        plt.plot(self.start[0], self.start[1], 'go', markersize=12, label='Start', markeredgecolor='black')
        plt.plot(self.goal[0], self.goal[1], 'b*', markersize=15, label='Goal', markeredgecolor='black')
        
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