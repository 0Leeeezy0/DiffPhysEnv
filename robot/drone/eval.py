import torch
import cv2
import numpy as np
import random
import time
import math
import genesis

import env.util as util
import env.geom as geom
import env.robot as robot
import env.sensor as sensor
import model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
""" 模型参数 """
model_path = 'outputs/checkpoint_episode_8000.pth'
steps = 1000
batch_size = 1
gru_seq_len = 32
target_vel = torch.zeros((batch_size, 1, 1), dtype=torch.float, device=device)
target_vel[:, :, :] = 0.5
safty_distance = 0.3
# 模型归一化参数
max_acc = 16.0
max_vel = 20.0
max_roll_pitch = 40.0
max_yaw = 30.0
ang_vel_max = [50, 50, 20]
""" GEOM设置 """
# 地形域随机化
sphere_dict = {'num':10, 'x_min':1.0, 'x_max':6.0, 'y_min':-3.0, 'y_max':3.0, 'z_min':1.5, 'z_max':3.0, 'R_min':0.2, 'R_max':0.4}
cylinder_dict = {'num':10, 'x_min':1.0, 'x_max':6.0, 'y_min':-3.0, 'y_max':3.0, 'z_min':10.0, 'z_max':10.0, 'R_min':0.2, 'R_max':0.3}
# sphere_dict = {'num':5, 'x_min':1.0, 'x_max':6.0, 'y_min':-5.0, 'y_max':5.0, 'z_min':1.5, 'z_max':3.0, 'R_min':0.3, 'R_max':1.0}
# cylinder_dict = {'num':10, 'x_min':1.0, 'x_max':6.0, 'y_min':-5.0, 'y_max':5.0, 'z_min':1.5, 'z_max':3.0, 'R_min':0.3, 'R_max':1.0}
# 目标速度域随机化
target_vel_range = {"min":0.5, "max":2.5}  
""" 机器人 """
init_pos = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float, device=device, requires_grad=False)
init_euler = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device, requires_grad=False)
mass = 0.33
T_max = 4*0.40*9.81  # 0.33*9.81 
collision_radius = 0.072
# 控制域随机化
T_att_range = {'min':0.0, 'max':0.5}
alpha_1_range = {'min':0.6, 'max':0.8}
control_freq_range = {"min":40.0, "max":60.0}
""" 深度相机 """
pos_offset = torch.tensor([0.0425, 0.0, 0.0345], dtype=torch.float, device=device, requires_grad=False)
euler_offset = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device, requires_grad=False)
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
""" GENESIS设置 """
if device == 'cuda':
    genesis.init(
        seed                = None,
        precision           = '32',
        debug               = False,
        eps                 = 1e-12,
        logging_level       = 'warning',
        backend             = genesis.cuda,
        theme               = 'dark',
        logger_verbose_time = False
    )
else:
    genesis.init(
        seed                = None,
        precision           = '32',
        debug               = False,
        eps                 = 1e-12,
        logging_level       = 'warning',
        backend             = genesis.cpu,
        theme               = 'dark',
        logger_verbose_time = False
    )
viewer_options = genesis.options.ViewerOptions(
    camera_pos=(1.0, 1.0, 1.0),
    camera_lookat=(0.0, 0.0, 0.0),
    camera_fov=90,
    max_FPS=120,
)
scene = genesis.Scene(
    sim_options=genesis.options.SimOptions(
        dt=0.01,
    ),
    viewer_options=viewer_options,
    show_viewer=True,
)
plane = scene.add_entity(
    genesis.morphs.Plane(
        visualization=True,   # 显示地面
        collision=False        # 有碰撞效果
    ),
)
drone = scene.add_entity(
    morph=genesis.morphs.Drone(
        file="urdf/ge_fpv.urdf",
        pos=init_pos.clone().detach().to('cpu').numpy(),
        euler=init_euler.clone().detach().to('cpu').numpy(),
    ),
)

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
    for sphere in range(sphere_dict['num']):
        sphere_x = random.uniform(sphere_dict['x_min'], sphere_dict['x_max'])
        sphere_y = random.uniform(sphere_dict['y_min'], sphere_dict['y_max'])
        sphere_z = random.uniform(sphere_dict['z_min'], sphere_dict['z_max'])
        sphere_R = random.uniform(sphere_dict['R_min'], sphere_dict['R_max'])
        geom.add_sphere(
            torch.tensor(
                [
                    sphere_x, 
                    sphere_y, 
                    sphere_z, 
                    sphere_R
                ], 
                dtype=torch.float, 
                device=device
            ),
            idx = 0
        )
        scene.add_entity(
            genesis.morphs.Sphere(
                pos=(sphere_x, sphere_y, sphere_z),
                radius=sphere_R,
                fixed=True,
            )
        )
    for cylinder in range(cylinder_dict['num']):
        cylinder_x = random.uniform(cylinder_dict['x_min'], cylinder_dict['x_max'])
        cylinder_y = random.uniform(cylinder_dict['y_min'], cylinder_dict['y_max'])
        cylinder_z = random.uniform(cylinder_dict['z_min'], cylinder_dict['z_max'])
        cylinder_R = random.uniform(cylinder_dict['R_min'], cylinder_dict['R_max'])
        cylinder_H = 2*cylinder_z
        geom.add_cylinder(
            torch.tensor(
                [
                    cylinder_x, 
                    cylinder_y, 
                    cylinder_z, 
                    cylinder_R,
                    cylinder_H
                ], 
                dtype=torch.float, 
                device=device
            ),
            idx = 0
        )
        scene.add_entity(
        genesis.morphs.Cylinder(
                height=cylinder_H,
                radius=cylinder_R,
                pos=(cylinder_x, cylinder_y, cylinder_z),
                fixed=True,
            )
        )
geom_random(geom, batch_size, sphere_dict, cylinder_dict)
obs = geom.build()
scene.build()
geom.reset()

""" 模型初始化 """
model = model.Model_Depth_GRU()  # 先创建模型实例
# model.load_state_dict(torch.load(model_path))  # 再加载参数
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])  # 从字典中提取模型参数
model.eval()

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
        dt = 1.0/random.uniform(control_freq_range['min'], control_freq_range['max'])
    )
    drone.set_pos(obs['pos'][0, 0, ...].clone().detach())
    drone.set_quat(util.euler_to_quat(util.deg_to_rad(obs['ang'][0, 0, ...]).clone().detach()))
    R = util.euler_to_R(util.deg_to_rad(obs['ang'][0, 0, ...]))
    scene.draw_debug_sphere(pos=obs['pos'][0, 0, ...].clone().detach().cpu(), radius=collision_radius, color=(1, 0, 0))
    scene.draw_debug_arrow(pos=obs['pos'][0, 0, ...].clone().detach().cpu(), vec=(R[:, 2]*act[0, 0, 3]).clone().detach().cpu(), color=(0, 1, 0), radius=0.01)
    scene.step()

    """ 深度图可视化 """
    depth_show(obs['depth'])

    time.sleep(0.5)