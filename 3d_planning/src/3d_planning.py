import os
import math
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 确保使用交互式后端
from mpl_toolkits.mplot3d import art3d
from stable_baselines3.common.callbacks import EvalCallback
# 导入或定义 PathPlanningEnv
from environment_3d import PathPlanningEnv_3d as PathPlanningEnv

# 初始化环境
env = PathPlanningEnv()

# 检查环境是否符合 Gymnasium 规范（如果需要）
from stable_baselines3.common.env_checker import check_env
check_env(env)

# 创建评估环境（用于自动保存最佳模型）
eval_env = PathPlanningEnv()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  
model_path = os.path.join(parent_dir, 'best_model1')        # 最佳模型

# 设置评估回调，自动保存最佳模型
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path= model_path,     # 最佳模型保存路径
    log_path=os.path.join(parent_dir, 'logs'),                   # 评估日志路径
    eval_freq=20000,                      # 每2000步评估一次
    n_eval_episodes=5,                    # 每次评估5个episode
    deterministic=True,                   # 使用确定性动作
    render=False,                          # 评估时不渲染
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
    print(f"无法加载最佳模型: {e}")
    # # 如果最佳模型不存在，使用最终模型
    # print(f"无法加载最佳模型: {e}")
    # print("使用最终模型进行测试")
    # test_model = model

# 测试模型
state, _ = env.reset()
state_list = [state.copy()]
# 假设state_list是2D的，添加z坐标
for _ in range(200):
    action, _ = test_model.predict(state)
    state, _, done, _, _ = env.step(action)
    state_list.append(state.copy())
    if done:
        break

state_list = np.array(state_list)

# 创建3D图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制边界（在z=0平面）
ax.add_patch(plt.Rectangle(
    (env.min_size, env.min_size),
    env.upper_size - env.min_size,
    env.upper_size - env.min_size, 
    fill=False, edgecolor='k', linewidth=2
))
art3d.pathpatch_2d_to_3d(ax.patches[0], z=0, zdir="z")

# 绘制障碍物
for i in range(len(env.obstacles)):
    center = env.obs_centers[i]
    radius = env.obs_radius[i]
    
    # 使用更少的点
    u = np.linspace(0, 2 * np.pi, 15)
    v = np.linspace(0, np.pi, 15)
    
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, color='red', alpha=0.5, edgecolor='darkred', linewidth=0.5)

# 绘制路径
ax.plot(state_list[:, 0], state_list[:, 1], state_list[:, 2],
        'b-', linewidth=3, label='Path')

# 绘制起点终点
ax.scatter(env.start[0], env.start[1], env.start[2], c='g', s=100, label='Start')
ax.scatter(env.goal[0], env.goal[1], env.goal[2], c='b', s=100, label='Goal')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.legend()
plt.title(" Visualization of 3D Path")
plt.show()