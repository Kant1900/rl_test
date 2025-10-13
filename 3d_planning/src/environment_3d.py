import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PathPlanningEnv_3d(gym.Env):
    def __init__(self):
        super().__init__()

        # 环境参数
        self.min_size = 1
        self.upper_size = 10
        self.start = np.array([1.5, 1.5, 1.5], dtype=np.float32)
        self.goal = np.array([9.6, 9.6, 9.6], dtype=np.float32)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.obstacles = np.array([   #位置，x,y,z,半径
            [5, 5, 5, 1.5], 
            [6, 8, 8, 0.5], 
            [8, 10, 10, 0.5],
            [8, 2, 2, 0.5], 
            [10, 6, 6, 1]
        ], dtype=np.float32)
        self.obs_centers = self.obstacles[:, :3]
        self.obs_radius = self.obstacles[:, 3]
        self.ideal_goal_speed_min = 0.1  # 理想的到达目标的最小速度
        self.ideal_goal_speed_max = 0.3  # 理想的到达目标的最大速度
        self.speed_tolerance = 0.2  # 理想速度的容忍度
        # 动作空间  规定加速度
        self.actions = np.array([
            [0, 0, 0],
            [1, 0, 0], 
            [-1, 0, 0],
            [0, 1, 0], 
            [0, -1, 0], 
            [0, 0, 1],
            [0, 0, -1]
        ])
        self.n_actions = self.actions.shape[0]

                # 物理参数
        self.dt = 0.1  # 时间步长
        self.max_speed = 2.0
        self.max_acceleration = 1 # 最大加速度

            # 定义状态空间 [x, y, z]
        self.observation_space = spaces.Box(
            low=np.array([self.min_size, self.min_size, self.min_size], dtype=np.float32),
            high=np.array([self.upper_size, self.upper_size, self.upper_size], dtype=np.float32),
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
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float)
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
    

    def _calculate_speed_reward(self, final_speed):
        """计算到达目标时的速度奖励"""
        # 理想速度范围奖励
        if self.ideal_goal_speed_min <= final_speed <= self.ideal_goal_speed_max:
            # 在理想范围内，给予高奖励
            return 30
        elif final_speed < self.ideal_goal_speed_min:
            # 速度太慢
            speed_ratio = final_speed / self.ideal_goal_speed_min
            return 10 * speed_ratio  # 按比例给予奖励
        else:
            # 速度太快
            overspeed_ratio = min(1.0, (final_speed - self.ideal_goal_speed_max) / self.max_speed)
            return -20 * overspeed_ratio  # 超速惩罚

    def _get_ideal_approach_speed(self, distance_to_goal):
        """根据距离目标的距离计算理想接近速度"""
        # 距离目标越近，理想速度应该越小
        if distance_to_goal > 2.0:
            return self.ideal_goal_speed_max
        elif distance_to_goal > 1.0:
            # 线性减速
            return self.ideal_goal_speed_min + (self.ideal_goal_speed_max - self.ideal_goal_speed_min) * (distance_to_goal - 1.0)
        else:
            return self.ideal_goal_speed_min

    def step(self, action):
        # 实现步进逻辑
        self.step_count += 1
        x, y, z = self.state
        acceleration = self.actions[action] * self.max_acceleration
        self.velocity += acceleration * self.dt
        speed = np.linalg.norm(self.velocity)

        # 限制最大速度
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed

        # 更新位置
        dx, dy ,dz= self.velocity * self.dt
        new_x = np.clip(x + dx, self.min_size, self.upper_size)
        new_y = np.clip(y + dy, self.min_size, self.upper_size)
        new_z = np.clip(z + dz, self.min_size, self.upper_size)
        next_state = np.array([new_x, new_y, new_z], dtype=np.float32)
        
        distance_to_goal = np.linalg.norm(next_state - self.goal)
        distance_improvement = np.linalg.norm(self.state - self.goal) - distance_to_goal
        self.eval_trajectory.append(next_state.copy())  # 添加新状态到轨迹中

        # 先检查所有终止条件
        done = False
        reward = -1  # 默认奖励

        # 1. 检查障碍物碰撞
        for i in range(len(self.obstacles)):
            obs_center = self.obs_centers[i]
            obs_radius = self.obs_radius[i]
            if np.linalg.norm(next_state - obs_center) <= obs_radius:
                reward = -100  # 碰撞惩罚
                done = True
                return next_state, reward, done, False, {}      
            
        # 2. 检查是否到达目标
        if distance_to_goal < 0.2:
            base_reward = 250
            step_penalty = self.step_count * 0.2
            # 速度约束检查
            speed_reward = self._calculate_speed_reward(speed)
            reward = base_reward - step_penalty + speed_reward
            done = True
            self.state = next_state
            return next_state, reward, done, False, {}
        
        # 3. 检查是否超出最大步数
        if self.step_count >= 200:
            reward = -100
            done = True
            self.state = next_state
            return next_state, reward, done, False, {}
        
        # 4. 普通移动情况
        base_penalty = -0.5

        dist_to_left = next_state[0] - self.min_size
        dist_to_right = self.upper_size - next_state[0]
        dist_to_behind = next_state[1] - self.min_size
        dist_to_front = self.upper_size - next_state[1]
        dist_to_bottom = next_state[2] - self.min_size
        dist_to_top = self.upper_size - next_state[2]
        dist_to_behind = next_state[1] - self.min_size
        min_edge_distance = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top, dist_to_behind, dist_to_front )

        # 如果太靠近边界，给予惩罚
        if min_edge_distance < self.edge_threshold :
            done = True
            self.state = next_state
            reward = -50
            return next_state, reward, done, False, {}
        
        distance_reward = float(distance_improvement * 6) # 奖励与距离改善成正比
        repulsive_punishment= self._calculate_repulsive_punishment_(next_state)

        approach_speed_reward = 0
        if distance_to_goal < 3.0:  # 当接近目标时
            # 鼓励减速到理想速度范围
            ideal_speed = self._get_ideal_approach_speed(distance_to_goal)
            speed_diff = abs(speed - ideal_speed)
            if speed_diff < self.speed_tolerance:
                approach_speed_reward = 1.0
            else:
                approach_speed_reward = -0.5 * (speed_diff / self.max_speed)

        reward = base_penalty + distance_reward + approach_speed_reward - repulsive_punishment
        done = False

        # 更新状态
        self.state = next_state
        return next_state, reward, done, False, {}    
    
    def render(self):
     pass