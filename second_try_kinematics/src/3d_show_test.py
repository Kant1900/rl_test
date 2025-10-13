import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import art3d
matplotlib.use('TkAgg')  # 确保使用交互式后端

from environment import PathPlanningEnv

# 初始化环境
env = PathPlanningEnv()

# 导入或定义 PathPlanningEnv
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  
model_path = os.path.join(parent_dir, 'best_model1')      # 最佳模型保存路径  改这里
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

# 绘制障碍物（圆柱体）
theta = np.linspace(0, 2 * np.pi, 100)
z_cylinder = np.linspace(0, 1, 10)  # 假设高度为1
for i in range(len(env.obstacles)):
    for z in z_cylinder:
        x_obs = env.obs_centers[i, 0] + env.obs_radius[i] * np.cos(theta)
        y_obs = env.obs_centers[i, 1] + env.obs_radius[i] * np.sin(theta)
        ax.plot(x_obs, y_obs, z, 'r-', alpha=0.3)

# 绘制路径（在z=0平面上的3D线）
z_path = np.zeros_like(state_list[:, 0])  # 所有点z=0
ax.plot(state_list[:, 0], state_list[:, 1], z_path, 
        'b-', linewidth=3, label='Path')

# 绘制起点终点
ax.scatter(env.start[0], env.start[1], 0, c='g', s=100, label='Start')
ax.scatter(env.goal[0], env.goal[1], 0, c='b', s=100, label='Goal')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.legend()
plt.title("3D Visualization of 2D Path")
plt.show()