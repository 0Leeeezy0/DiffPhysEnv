import torch
import cv2
import numpy as np
import random
import time
import math

import env.util as util
import env.geom as geom
import env.robot as robot
import env.sensor as sensor
import model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
""" 训练参数 """
checkpoint = {'state': False, 'path': 'outputs/checkpoint_episode_11000.pth'}
episodes = 50000
save_episode_period = 1000
steps = 300
batch_size = 12
gru_seq_len = 32
target_pos = torch.tensor([[[8.0, 0.0, 0.0]]], dtype=torch.float, device=device)
target_vel = torch.zeros((batch_size, 1, 1), dtype=torch.float, device=device)
target_vel[:, :, :] = 0.1
safty_distance = 0.3
best_mean_reward = -1e10
# 模型归一化参数
max_vel = 1.0
max_ang = 180.0
# 奖励
coef = {
    "coef_out_vel": -5.0,    # 惩罚超出最大速度误差
    "coef_distance_target": -1.0,   # 惩罚到目标点的距离    
    "coef_distance_no_safty": -15.0,  # 惩罚不安全距离
    "coef_crash": -20.0,    # 惩罚碰撞
    "coef_get_target": 20.0,    # 奖励到达目标位置
    "coef_alive": 1.0,  # 奖励存活
}
# act = torch.tensor(
#     [
#         [[0.0, 10.0, 0.0, 1.0]],
#         [[0.0, 10.0, 0.0, 1.0]],
#         [[0.0, 10.0, 0.0, 0.9]],
#         [[0.0, 10.0, 0.0, 1.0]],
#         [[0.0, 10.0, 0.0, 1.0]],
#         [[0.0, 10.0, 0.0, 1.0]],
#         [[0.0, 10.0, 0.0, 1.0]],
#         [[0.0, 10.0, 0.0, 1.2]],
#         [[0.0, 10.0, 0.0, 1.0]],
#         [[0.0, 10.0, 0.0, 1.0]],
#         [[0.0, 10.0, 0.0, 1.0]],
#         [[0.0, 10.0, 0.0, 1.0]]
#     ], 
#     dtype=torch.float, device=device, requires_grad=True)
""" GEOM设置 """
# 地形域随机化
sphere_dict = {'num':0, 'x_min':1.0, 'x_max':6.0, 'y_min':-3.0, 'y_max':3.0, 'z_min':1.5, 'z_max':3.0, 'R_min':0.2, 'R_max':0.4}
cylinder_dict = {'num':10, 'x_min':1.0, 'x_max':6.0, 'y_min':-3.0, 'y_max':3.0, 'z_min':0.3, 'z_max':0.3, 'R_min':0.2, 'R_max':0.3}
# 目标速度域随机化
target_vel_range = {"min":0.5, "max":2.5}  
""" 机器人 """
init_pos = torch.tensor([0.0, 0.0, 0.01+0.072], dtype=torch.float, device=device, requires_grad=True)
init_euler = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device, requires_grad=True)
mass = 0.33
collision_radius = 0.072
# 控制域随机化
alpha_1_range = {'min':0.6, 'max':0.8}
alpha_2_range = {'min':0.6, 'max':0.8}
control_freq_range = {"min":40.0, "max":60.0}
""" 2D激光雷达 """
pos_offset = torch.tensor([0.0, 0.0, 0.1], dtype=torch.float, device=device, requires_grad=True)
euler_offset = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device, requires_grad=True)
angle_range = {'start':0.0, 'end':360.0}
angular_res = 0.72
distance_range = {'min': 0.03, 'max': 12.0}
# 传感器域随机化
noise_range = {'min':0.0, 'max':0.002}

""" GEOM/机器人/传感器设置 """
geom = geom.geom(
    batch_size=batch_size,
    device=device,
)
rigid_robot = robot.rigid(
    device = device,
    init_pos = init_pos, 
    init_euler = init_euler, 
    mass = mass, 
    collision_radius = collision_radius
)
cloest_dist = sensor.cloest_dist(
    device = device
)
lidar_2D = sensor.lidar_2D(
    device = device,
    pos_offset = pos_offset, 
    euler_offset = euler_offset,  
    angle_range = angle_range,
    angular_res = angular_res,
    distance_range = distance_range,
    noise_range = noise_range
)
rigid_robot.sensor_bind(cloest_dist)
rigid_robot.sensor_bind(lidar_2D)
geom.add_robot(rigid_robot)

"""
    @ 点云转换
"""
def draw_lidar_points(distance, angle_deg, img_size=300, max_distance=10):
    # 转换为 numpy
    if torch.is_tensor(distance):
        distance = distance.cpu().numpy()
    if torch.is_tensor(angle_deg):
        angle_deg = angle_deg.cpu().numpy()
    
    # 创建黑色背景
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    center = (img_size // 2, img_size // 2)
    
    # 极坐标转直角坐标并绘制
    for dist, ang in zip(distance, angle_deg):
        if dist <= 0 or dist > max_distance:
            continue
            
        rad = np.deg2rad(-ang-90.0)
        r = (dist / max_distance) * (img_size // 2 - 10)
        x = int(center[0] + r * np.cos(rad))
        y = int(center[1] + r * np.sin(rad))
        
        # 只画点（白色）
        cv2.circle(img, (x, y), 1, (255), -1)
    
    return img

"""
    @ 点云可视化
"""
def cloud_point_show(cloud_point, img_size, max_distance):
    img_list = []
    for geom_idx in range(cloud_point.size(0)):
        for robot_depth_idx in range(cloud_point.size(1)):
            img_norm_np = draw_lidar_points(cloud_point[geom_idx, robot_depth_idx, :, 0], 
                                           cloud_point[geom_idx, robot_depth_idx, :, 1], 
                                           img_size=img_size, 
                                           max_distance=max_distance)
            
            # 在图像中央画一个1像素的红点
            center_y = img_norm_np.shape[0] // 2
            center_x = img_norm_np.shape[1] // 2
            
            # 对于单通道灰度图，设置该像素为红色（需要转换为BGR或保持单通道但标记）
            if len(img_norm_np.shape) == 2:  # 灰度图
                # 方法1: 将单通道转换为3通道BGR图像
                img_color = cv2.cvtColor(img_norm_np, cv2.COLOR_GRAY2BGR)
                img_color[center_y, center_x] = [0, 0, 255]  # BGR格式的红色
                img_list.append(img_color)
            else:  # 已经是彩色图
                img_norm_np[center_y, center_x] = [0, 0, 255]  # 设置红点
                img_list.append(img_norm_np)
    
    # 计算网格布局
    num_images = len(img_list)
    cols = min(6, num_images)
    rows = math.ceil(num_images / cols)
    
    # 获取单张图像尺寸
    h, w = img_list[0].shape[:2]  # 处理彩色图
    line_width = 2
    
    # 创建空白画布
    canvas_h = h * rows + line_width * (rows - 1)
    canvas_w = w * cols + line_width * (cols - 1)
    
    # 根据图像类型创建画布
    if len(img_list[0].shape) == 3:  # 彩色图
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    else:  # 灰度图
        canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255
    
    # 填充图像
    for idx, img in enumerate(img_list):
        row = idx // cols
        col = idx % cols
        y_start = row * (h + line_width)
        y_end = y_start + h
        x_start = col * (w + line_width)
        x_end = x_start + w
        
        canvas[y_start:y_end, x_start:x_end] = img
    
    # 缩放画布
    scale = 3
    canvas_resized = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    
    cv2.imshow("All Depth Images", canvas_resized)
    cv2.waitKey(1)

"""
    @ 地形域随机化
"""
def geom_random(geom, batch_size, sphere_dict, cylinder_dict):
    geom.clear()
    for idx in range(batch_size):
        for sphere in range(sphere_dict['num']):
            geom.add_sphere(
                torch.tensor(
                    [
                        random.uniform(sphere_dict['x_min'], sphere_dict['x_max']), 
                        random.uniform(sphere_dict['y_min'], sphere_dict['y_max']), 
                        random.uniform(sphere_dict['z_min'], sphere_dict['z_max']), 
                        random.uniform(sphere_dict['R_min'], sphere_dict['R_max'])
                    ], 
                    dtype=torch.float, 
                    device=device
                ),
                idx = idx
            )
        for cylinder in range(cylinder_dict['num']):
            z = random.uniform(cylinder_dict['z_min'], cylinder_dict['z_max'])
            geom.add_cylinder(
                torch.tensor(
                    [
                        random.uniform(cylinder_dict['x_min'], cylinder_dict['x_max']), 
                        random.uniform(cylinder_dict['y_min'], cylinder_dict['y_max']), 
                        z, 
                        random.uniform(cylinder_dict['R_min'], cylinder_dict['R_max']),
                        z*2
                    ], 
                    dtype=torch.float, 
                    device=device
                ),
                idx = idx
            )

""" 模型初始化 """
model = model.Model().to(device)
optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
start_episode = 0
# 加载checkpoint
if checkpoint['state']:
    checkpoint = torch.load(checkpoint['path'])
    # 按顺序恢复状态
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    start_episode = checkpoint['epoch'] + 1  # 从下一轮开始
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
checkpoint_num = 0
last_checkpoint_episode = 0
for episode in range(start_episode, episodes):
    start = time.perf_counter()
    geom_random(geom, batch_size, sphere_dict, cylinder_dict)
    geom.reset()
    obs = geom.build()
    """ 时序化队列 """
    cloud_point_queue = [torch.zeros_like(obs['cloud_point'][:, :, :, 0])]*gru_seq_len
    vel_queue = [torch.zeros_like(obs['vel'])]*gru_seq_len
    ang_queue = [torch.zeros_like(obs['ang'])]*gru_seq_len
    target_vel_queue = [torch.zeros_like(target_vel)]*gru_seq_len
    """ 奖励队列 """
    # 奖励计算
    reward_vel_queue = []
    reward_H_dir_queue = []
    reward_list = []
    distance_prev = obs['distance'].clone()
    for step in range(steps):
        """ 模型前向传播 """
        # 归一化
        cloud_point_norm = (obs['cloud_point'][:, :, :, 0] / distance_range['max']).clamp_max(distance_range['max'])
        vel_norm = obs['vel'] / max_vel
        ang_norm = util.rad_to_deg(obs['ang']) / max_ang
        target_vel_norm = target_vel/max_vel
        # 时序化
        cloud_point_queue.append(cloud_point_norm)
        if len(cloud_point_queue) > gru_seq_len:
            cloud_point_queue.pop(0)
        vel_queue.append(vel_norm)
        if len(vel_queue) > gru_seq_len:
            vel_queue.pop(0)
        ang_queue.append(ang_norm)
        if len(ang_queue) > gru_seq_len:
            ang_queue.pop(0)
        target_vel_queue.append(target_vel_norm)
        if len(target_vel_queue) > gru_seq_len:
            target_vel_queue.pop(0)
        # 前向传播
        act_raw = model.forward(
            torch.stack(cloud_point_queue, dim=2), 
            torch.stack(vel_queue, dim=2), 
            torch.stack(ang_queue, dim=2), 
            torch.stack(target_vel_queue, dim=2)
        )
        """ 映射 """
        act = torch.zeros((batch_size, 1, 6), dtype=torch.float, device=device)
        act[:, 0, 0:2] = act_raw[:, 0, 0:2]
        act[:, 0, 2:5] = 0.0
        act[:, 0, 5] = act_raw[:, 0, 5]
        """ 仿真 """
        obs = geom.step(
            mode = 'vel+ang',
            T_att_range = {'min':0.0, 'max':0.0},
            act = act,
            alpha_1_range = alpha_1_range,   
            alpha_2_range = alpha_2_range,
            dt = 1.0/random.uniform(control_freq_range['min'], control_freq_range['max'])
        )

        """ 点云可视化 """
        cloud_point_show(obs['cloud_point'], 150, distance_range['max']/2)

        """ 奖励 """
        # 计算平均
        reward_vel_queue.append(obs['vel'])
        if len(reward_vel_queue) > 40:
            reward_vel_queue.pop(0)
        vel_avg = torch.stack(reward_vel_queue).mean(dim=0)   # 计算平均速度
        target_vel_vec = util.tensor_norm(target_pos-obs['pos'])*target_vel # 目标速度方向向量
        vel_delta = (torch.norm(vel_avg, dim=-1)-torch.norm(target_vel_vec, dim=-1)).clamp_min(0.0)
        vel_to_pt = ((distance_prev-obs['distance'])*135).clamp_min(1.0)
        is_robot_get_target = torch.norm((obs['pos']-target_pos), dim=-1) <= 0.2
        # 计算奖励
        reward = (
            coef["coef_out_vel"]*vel_delta + \
            coef["coef_distance_target"]*torch.norm((obs['pos']-target_pos), dim=-1) + \
            coef["coef_distance_no_safty"]*vel_to_pt*(safty_distance-obs['distance']).squeeze(1) + \
            coef["coef_crash"]*obs['is_collision'].float() + \
            coef["coef_get_target"]*is_robot_get_target.float() + \
            coef["coef_alive"]
        )
        reward_list.append(reward)
        is_collision = [item for sublist in obs['is_collision'] for item in sublist]
        if sum(is_collision) == batch_size:
            break
        distance_prev = obs['distance']

    # 计算折扣奖励
    discount_reward_list = []
    discount_reward = torch.zeros(batch_size, 1, device=device)
    for reward in reversed(reward_list):
        discount_reward = steps/(step+1)*reward+0.99*discount_reward
        discount_reward_list.insert(0, discount_reward)
    mean_reward_list = torch.stack(discount_reward_list).mean(dim=0)
    # 计算损失
    batch_loss = -mean_reward_list
    loss = torch.mean(batch_loss)
    # 反向传播
    optim.zero_grad()
    loss.backward()
    optim.step()
    """ 保存checkpoint """
    # 当平均奖励大于之前的最佳平均奖励 && 全部没有碰撞 || 每save_episode_period轮
    is_collision = [item for sublist in obs['is_collision'] for item in sublist]
    if (torch.mean(mean_reward_list) > best_mean_reward and sum(is_collision) == 0):
        best_mean_reward = torch.mean(mean_reward_list)
        checkpoint_num += 1
        last_checkpoint_episode = episode
        checkpoint = {
            'epoch': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss if 'loss' in locals() else None,
            'best_score': best_mean_reward,
            'model_mode': 'train' if model.training else 'eval',
        }
        torch.save(checkpoint, f'outputs/checkpoint_{checkpoint_num}.pth')
        print("Save Checkpoint")
    elif (episode+1)%save_episode_period == 0:
        checkpoint = {
            'epoch': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss if 'loss' in locals() else None,
            'best_score': best_mean_reward,
            'model_mode': 'train' if model.training else 'eval',
        }
        torch.save(checkpoint, f'outputs/checkpoint_episode_{episode+1}.pth')
        print("Save Checkpoint")

    end = time.perf_counter()
    elapsed = end - start
    sep = "=" * 50
    print(f"""
    {sep}
    @ Episode: {episode:3d}/{episodes}
    @ Non Collision: {batch_size-sum(is_collision).item()}/{batch_size}
    @ Mean Reward: {torch.mean(mean_reward_list)}
    @ Min Reward: {torch.min(mean_reward_list)}
    @ Max Reward: {torch.max(mean_reward_list)}
    @ Best Mean Reward: {best_mean_reward}
    @ Last Checkpoint Episode: {last_checkpoint_episode}
    @ Duration Time: {elapsed}s
    {sep}
    """)    