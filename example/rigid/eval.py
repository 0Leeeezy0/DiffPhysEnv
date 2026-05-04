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
model_path = 'outputs/checkpoint_46.pth'
steps = 10000
batch_size = 1
gru_seq_len = 24
target_pos = torch.zeros((batch_size, 1, 3), dtype=torch.float, device=device)
target_vel = torch.zeros((batch_size, 1, 1), dtype=torch.float, device=device)
r = 5.5
theta = random.uniform(0, 2*math.pi)
target_pos[:, :, 0] = r*math.cos(theta)
target_pos[:, :, 1] = r*math.sin(theta)
target_pos[:, :, 2] = 0.01+0.072
target_vel[:, :, :] = 0.5
safty_distance = 0.3
# 模型归一化参数
max_pos = 15.0
max_vel = 1.0
max_ang = 180.0
""" GEOM设置 """
# 地形域随机化
sphere_dict = {'num':0, 'r_min':1.0, 'r_max':5.0, 'z_min':1.5, 'z_max':3.0, 'R_min':0.2, 'R_max':0.4}
cylinder_dict = {'num':10, 'r_min':1.0, 'r_max':5.0, 'z_min':0.3, 'z_max':0.3, 'R_min':0.1, 'R_max':0.3}
# sphere_dict = {'num':0, 'x_min':1.0, 'x_max':6.0, 'y_min':-3.0, 'y_max':3.0, 'z_min':1.5, 'z_max':3.0, 'R_min':0.2, 'R_max':0.4}
# cylinder_dict = {'num':15, 'x_min':1.0, 'x_max':5.5, 'y_min':-3.0, 'y_max':3.0, 'z_min':0.3, 'z_max':0.3, 'R_min':0.1, 'R_max':0.3}
# 目标速度域随机化
target_vel_range = {"min":0.5, "max":2.5}  
""" 机器人 """
init_pos = torch.tensor([0.0, 0.0, 0.01+0.072], dtype=torch.float, device=device, requires_grad=False)
init_euler = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device, requires_grad=False)
mass = 0.33
collision_radius = 0.072
# 控制域随机化
alpha_1_range = {'min':0.6, 'max':0.8}
alpha_2_range = {'min':0.6, 'max':0.8}
control_freq_range = {"min":7.0, "max":10.0}
""" 2D激光雷达 """
pos_offset = torch.tensor([0.0, 0.0, 0.1], dtype=torch.float, device=device, requires_grad=False)
euler_offset = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device, requires_grad=False)
angle_range = {'start':0.0, 'end':360.0}
angular_res = 0.72
distance_range = {'min': 0.03, 'max': 12.0}
# 传感器域随机化
noise_range = {'min':0.0, 'max':0.002}

""" GEOM/机器人/传感器设置 """
geom = geom.geom(
    batch_size=batch_size,
    device=device,
    requires_grad=False
)
rigid_robot = robot.rigid(
    device = device,
    requires_grad=False,
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
            r = random.uniform(sphere_dict['r_min'], sphere_dict['r_max'])
            theta = random.uniform(0, 2*math.pi)
            sphere_x = r*math.cos(theta)
            sphere_y = r*math.sin(theta)
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
            r = random.uniform(cylinder_dict['r_min'], cylinder_dict['r_max'])
            theta = random.uniform(0, 2*math.pi)
            z = random.uniform(cylinder_dict['z_min'], cylinder_dict['z_max'])
            cylinder_x = r*math.cos(theta)
            cylinder_y = r*math.sin(theta)
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
# def geom_random(geom, batch_size, sphere_dict, cylinder_dict):
#     geom.clear()
#     for sphere in range(sphere_dict['num']):
#         sphere_x = random.uniform(sphere_dict['x_min'], sphere_dict['x_max'])
#         sphere_y = random.uniform(sphere_dict['y_min'], sphere_dict['y_max'])
#         sphere_z = random.uniform(sphere_dict['z_min'], sphere_dict['z_max'])
#         sphere_R = random.uniform(sphere_dict['R_min'], sphere_dict['R_max'])
#         geom.add_sphere(
#             torch.tensor(
#                 [
#                     sphere_x, 
#                     sphere_y, 
#                     sphere_z, 
#                     sphere_R
#                 ], 
#                 dtype=torch.float, 
#                 device=device
#             ),
#             idx = 0
#         )
#         scene.add_entity(
#             genesis.morphs.Sphere(
#                 pos=(sphere_x, sphere_y, sphere_z),
#                 radius=sphere_R,
#                 fixed=True,
#             )
#         )
#     for cylinder in range(cylinder_dict['num']):
#         cylinder_x = random.uniform(cylinder_dict['x_min'], cylinder_dict['x_max'])
#         cylinder_y = random.uniform(cylinder_dict['y_min'], cylinder_dict['y_max'])
#         cylinder_z = random.uniform(cylinder_dict['z_min'], cylinder_dict['z_max'])
#         cylinder_R = random.uniform(cylinder_dict['R_min'], cylinder_dict['R_max'])
#         cylinder_H = 2*cylinder_z
#         geom.add_cylinder(
#             torch.tensor(
#                 [
#                     cylinder_x, 
#                     cylinder_y, 
#                     cylinder_z, 
#                     cylinder_R,
#                     cylinder_H
#                 ], 
#                 dtype=torch.float, 
#                 device=device
#             ),
#             idx = 0
#         )
#         scene.add_entity(
#         genesis.morphs.Cylinder(
#                 height=cylinder_H,
#                 radius=cylinder_R,
#                 pos=(cylinder_x, cylinder_y, cylinder_z),
#                 fixed=True,
#             )
#         )

geom_random(geom, batch_size, sphere_dict, cylinder_dict)
geom.reset()
obs = geom.build()
scene.build()

""" 模型初始化 """
model = model.Model_pos()  # 先创建模型实例
# model.load_state_dict(torch.load(model_path))  # 再加载参数
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])  # 从字典中提取模型参数
model.eval()

""" 时序化队列 """
cloud_point_queue = [torch.zeros_like(obs['cloud_point'][:, :, :, 0])]*gru_seq_len
pos_queue = [torch.zeros_like(obs['pos'])]*gru_seq_len
# vel_queue = [torch.zeros_like(obs['vel'])]*gru_seq_len
ang_queue = [torch.zeros_like(obs['ang'])]*gru_seq_len
target_vel_queue = [torch.zeros_like(target_vel)]*gru_seq_len
for step in range(steps):
    # 随机目标位置
    for i in range(batch_size):
        mask = torch.norm(obs['pos'][i,0,:]-target_pos, dim=-1) < 1
        if mask[i,:]:
            model.reset()
            theta = random.uniform(0, 2*math.pi)
            target_pos[i, :, 0] = r*math.cos(theta)
            target_pos[i, :, 1] = r*math.sin(theta)
    """ 模型前向传播 """
    # 归一化
    cloud_point_norm = (obs['cloud_point'][:, :, :, 0] / distance_range['max']).clamp_max(distance_range['max'])
    pos_norm = obs['pos'] / max_pos
    vel_norm = obs['vel'] / max_vel
    ang_norm = util.rad_to_deg(obs['ang']) / max_ang
    target_pos_norm = target_pos/max_pos
    target_vel_norm = target_vel/max_vel
    # 时序化
    cloud_point_queue.append(cloud_point_norm)
    if len(cloud_point_queue) > gru_seq_len:
        cloud_point_queue.pop(0)
    pos_queue.append(pos_norm)
    if len(pos_queue) > gru_seq_len:
        pos_queue.pop(0)
    ang_queue.append(ang_norm)
    if len(ang_queue) > gru_seq_len:
        ang_queue.pop(0)
    print(pos_norm)
    print(target_pos_norm)
    print("")
    # 前向传播
    act_raw = model.forward(
        torch.stack(cloud_point_queue, dim=2), 
        torch.stack(pos_queue, dim=2), 
        torch.stack(ang_queue, dim=2), 
        target_pos_norm,
        target_vel_norm
    ).detach()
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
    R = util.euler_to_R(util.deg_to_rad(obs['ang'][0, 0, ...]))
    scene.draw_debug_sphere(pos=target_pos[0, 0, ...].clone().detach().cpu(), radius=collision_radius, color=(0, 1, 0))
    scene.draw_debug_sphere(pos=obs['pos'][0, 0, ...].clone().detach().cpu(), radius=collision_radius, color=(1, 0, 0))
    scene.draw_debug_arrow(pos=obs['pos'][0, 0, ...].clone().detach().cpu(), vec=(R[:, 2]*act[0, 0, 3]).clone().detach().cpu(), color=(0, 1, 0), radius=0.01)
    scene.step()

    # time.sleep(0.5)