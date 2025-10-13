import os
import math
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 确保使用交互式后端
from stable_baselines3.common.callbacks import EvalCallback
# 导入或定义 PathPlanningEnv
from environment import PathPlanningEnv

# 初始化环境
env = PathPlanningEnv()

# 检查环境是否符合 Gymnasium 规范（如果需要）
from stable_baselines3.common.env_checker import check_env
check_env(env)

# 创建评估环境（用于自动保存最佳模型）
eval_env = PathPlanningEnv()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  
model_path = os.path.join(parent_dir, 'best_model1')      # 最佳模型保存路径  改这里

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
# model.learn(total_timesteps=max_episodes * 500 ,callback=eval_callback ) # 添加回调函数
# model.save(os.path.join(parent_dir, 'dqn_path_agent')) # 保存最终模型


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
for _ in range(500):
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
    (env.min_size, env.min_size),
    env.upper_size - env.min_size,
    env.upper_size - env.min_size, 
    fill=False, edgecolor='k', linewidth=2
))

# 障碍物
theta = np.linspace(0, 2 * np.pi, 100)
for i in range(len(env.obstacles)):
    x_obs = env.obs_centers[i, 0] + env.obs_radius[i] * np.cos(theta)
    y_obs = env.obs_centers[i, 1] + env.obs_radius[i] * np.sin(theta)
    plt.fill(x_obs, y_obs, 'r', alpha=0.5)

# 绘制起点、终点及路径
plt.plot(env.start[0], env.start[1], 'go', markersize=10, label='Start')
plt.plot(env.goal[0], env.goal[1], 'bo', markersize=10, label='Goal')
plt.plot(state_list[:, 0], state_list[:, 1], 'b-', linewidth=2, label='Path')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()
plt.title("Path Traversed by Agent")
plt.show()