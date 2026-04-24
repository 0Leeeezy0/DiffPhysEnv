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
steps = 150
batch_size = 12
gru_seq_len = 32
target_pos = torch.tensor([[[8.0, 0.0, 2.0]]], dtype=torch.float, device=device)
target_vel = torch.zeros((batch_size, 1, 1), dtype=torch.float, device=device)
target_vel[:, :, :] = 0.5
safty_distance = 0.3
best_mean_reward = -1e10
# 模型归一化参数
max_acc = 16.0
max_vel = 20.0
max_roll_pitch = 40.0
max_yaw = 30.0
ang_vel_max = [50, 50, 20]
# 奖励
coef = {
    "coef_vel": -0.5,    # 惩罚速度误差
    "coef_H_dir": -0.0,    # 惩罚水平方向误差
    "coef_distance_target": -1.0,   # 惩罚到目标点的距离    
    "coef_distance_no_safty": -15.0,  # 惩罚不安全距离
    "coef_crash": -20.0,    # 惩罚碰撞
    "coef_T": -10.0,    # 惩罚模型推力输出为负
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
sphere_dict = {'num':10, 'x_min':1.0, 'x_max':6.0, 'y_min':-3.0, 'y_max':3.0, 'z_min':1.5, 'z_max':3.0, 'R_min':0.2, 'R_max':0.4}
cylinder_dict = {'num':10, 'x_min':1.0, 'x_max':6.0, 'y_min':-3.0, 'y_max':3.0, 'z_min':10.0, 'z_max':10.0, 'R_min':0.2, 'R_max':0.3}
# 目标速度域随机化
target_vel_range = {"min":0.5, "max":2.5}  
""" 机器人 """
init_pos = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float, device=device, requires_grad=True)
init_euler = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device, requires_grad=True)
mass = 0.33
T_max = 4*0.40*9.81  # 0.33*9.81 
collision_radius = 0.072
# 控制域随机化
T_att_range = {'min':0.0, 'max':0.5}
alpha_1_range = {'min':0.6, 'max':0.8}
control_freq_range = {"min":40.0, "max":60.0}
""" 深度相机 """
pos_offset = torch.tensor([0.0425, 0.0, 0.0345], dtype=torch.float, device=device, requires_grad=True)
euler_offset = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device, requires_grad=True)
res_W = 40
res_H = 25
fov_H = 67.9
fov_V = 45.3
depth_distance_range = {"min":0.25, "max":2.5}
# 传感器域随机化
noise_range = {'min':0.0, 'max':0.05}
black_hole_prob = 0.0

""" GEOM/机器人/传感器设置 """
geom = geom.geom(
    batch_size=batch_size,
    device=device,
)
drone_robot = robot.drone(
    device = device,
    init_pos = init_pos, 
    init_euler = init_euler, 
    mass = mass, 
    T_max = T_max, 
    collision_radius = collision_radius
)
cloest_dist = sensor.cloest_dist(
    device = device
)
depth = sensor.depth(
    device = device,
    pos_offset = pos_offset, 
    euler_offset = euler_offset,  
    res_W = res_W, 
    res_H = res_H, 
    fov_H = fov_H,
    fov_V = fov_V,
    distance_range = depth_distance_range,
    noise_range = noise_range,
    black_hole_prob = black_hole_prob
)
drone_robot.sensor_bind(cloest_dist)
drone_robot.sensor_bind(depth)
geom.add_robot(drone_robot)

"""
    @ 深度图可视化
"""
def depth_show(depth):
    img_list = []
    for geom_idx in range(depth.size(0)):
        for robot_depth_idx in range(depth.size(1)):
            img = 255 * depth[geom_idx, robot_depth_idx, ...].detach().cpu().numpy() / depth_distance_range['max']
            img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            img_norm_np = img_norm.astype(np.uint8)
            img_list.append(img_norm_np)
    # 计算网格布局
    num_images = len(img_list)
    cols = min(4, num_images)
    rows = math.ceil(num_images / cols)
    # 获取单张图像尺寸
    h, w = img_list[0].shape
    line_width = 2  # 分割线宽度
    # 创建空白画布（考虑分割线）
    canvas_h = h * rows + line_width * (rows - 1)
    canvas_w = w * cols + line_width * (cols - 1)
    canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255
    # 填充图像
    for idx, img in enumerate(img_list):
        row = idx // cols
        col = idx % cols
        # 计算图像位置（考虑分割线）
        y_start = row * (h + line_width)
        y_end = y_start + h
        x_start = col * (w + line_width)
        x_end = x_start + w
        canvas[y_start:y_end, x_start:x_end] = img
    # 缩放画布到合适大小（可选：放大显示）
    scale = 3  # 放大1.5倍
    canvas_resized = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.imshow("All Depth Images", canvas_resized)
    cv2.waitKey(1)

"""
    @ 地形域随机化
"""
def geom_random(geom, batch_size, sphere_dict, cylinder_dict):
    geom.clear()
    for idx in range(batch_size):
        # geom.add_cylinder(
        #     torch.tensor(
        #         [
        #             0.0, 
        #             0.0, 
        #             0.0, 
        #             1.0,
        #             1.0
        #         ], 
        #         dtype=torch.float, 
        #         device=device
        #     ),
        #     idx = idx
        # )
        # geom.add_cylinder(
        #     torch.tensor(
        #         [
        #             2.5, 
        #             0.0, 
        #             0.0, 
        #             1.0,
        #             5.0
        #         ], 
        #         dtype=torch.float, 
        #         device=device
        #     ),
        #     idx = idx
        # )
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
model = model.Model_Depth_GRU().to(device)
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
    depth_queue = [torch.zeros_like(obs['depth'])]*gru_seq_len
    acc_queue = [torch.zeros_like(obs['acc'])]*gru_seq_len
    ang_queue = [torch.zeros_like(obs['ang'])]*gru_seq_len
    ang_vel_queue = [torch.zeros_like(obs['ang_vel'])]*gru_seq_len
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
        depth_norm = obs['depth'] / depth_distance_range['max']
        acc_norm = obs['acc'] / max_acc
        ang_norm = util.rad_to_deg(obs['ang']) / 180.0
        ang_vel_norm = obs['ang_vel'] / torch.tensor(ang_vel_max, device=device, dtype=torch.float).unsqueeze(0)
        target_vel_norm = target_vel/max_vel
        # 时序化
        depth_queue.append(depth_norm)
        if len(depth_queue) > gru_seq_len:
            depth_queue.pop(0)
        acc_queue.append(acc_norm)
        if len(acc_queue) > gru_seq_len:
            acc_queue.pop(0)
        ang_queue.append(ang_norm)
        if len(ang_queue) > gru_seq_len:
            ang_queue.pop(0)
        ang_vel_queue.append(ang_vel_norm)
        if len(ang_vel_queue) > gru_seq_len:
            ang_vel_queue.pop(0)
        target_vel_queue.append(target_vel_norm)
        if len(target_vel_queue) > gru_seq_len:
            target_vel_queue.pop(0)
        # 前向传播
        act_raw, _ = model.forward(
            torch.stack(depth_queue, dim=2), 
            torch.stack(acc_queue, dim=2), 
            torch.stack(ang_queue, dim=2), 
            torch.stack(ang_vel_queue, dim=2),
            torch.stack(target_vel_queue, dim=2)
        )
        """ 映射 """
        act = torch.zeros((batch_size, 1, 4), dtype=torch.float, device=device)
        act[:, 0, 0:2] = torch.tanh(act_raw[:, 0, 0:2])*max_roll_pitch
        act[:, 0, 2] = 0.0
        act[:, 0, 3] = torch.sigmoid(act_raw[:, 0, 2])
        """ 仿真 """
        obs = geom.step(
            mode = 'euler+T_rate',
            T_att_range = T_att_range,
            act = act,
            alpha_1_range = alpha_1_range,   
            alpha_2_range = alpha_1_range,
            dt = 1.0/random.uniform(control_freq_range['min'], control_freq_range['max'])
        )

        """ 深度图可视化 """
        depth_show(obs['depth'])

        """ 奖励 """
        # 计算平均
        reward_vel_queue.append(obs['vel'])
        if len(reward_vel_queue) > 40:
            reward_vel_queue.pop(0)
        vel_avg = torch.stack(reward_vel_queue).mean(dim=0)   # 计算平均速度
        reward_H_dir_queue.append(obs['vel'])
        if len(reward_H_dir_queue) > 40:
            reward_H_dir_queue.pop(0)
        H_dir_avg = torch.stack(reward_H_dir_queue).mean(dim=0)   # 计算平均水平方向
        target_vel_vec = util.tensor_norm(target_pos-obs['pos'])*target_vel # 目标速度方向向量
        vel_to_pt = ((distance_prev-obs['distance'])*135).clamp_min(1)
        T_rate = torch.abs(act_raw[:, 0, 2].clamp_max(0))
        # 计算奖励
        reward = (
            coef["coef_vel"]*torch.abs(torch.norm(vel_avg, dim=-1)-torch.norm(target_vel_vec, dim=-1)) + \
            coef["coef_H_dir"]*torch.norm(util.tensor_norm(H_dir_avg)[:, :, 0:2]-util.euler_to_R(util.deg_to_rad(obs['ang']))[:, :, 0:2, 0], dim=-1) + \
            coef["coef_distance_target"]*(torch.norm((obs['pos']-target_pos), dim=-1)**2) + \
            coef["coef_distance_no_safty"]*vel_to_pt*(safty_distance-obs['distance']).squeeze(1) + \
            coef["coef_crash"]*obs['is_collision'].float() + \
            coef["coef_T"]*T_rate + \
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