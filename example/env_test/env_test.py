import torch
import genesis
import cv2
import numpy as np
import time
import math

import env.util as util
import env.geom as geom
import env.robot as robot
import env.sensor as sensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
init_pos = torch.tensor([0.0, 0.0, 0.05], dtype=torch.float, device=device, requires_grad=True)
init_euler = torch.tensor([0.0, 0.0, 45.0], dtype=torch.float, device=device, requires_grad=True)
act = torch.tensor(
    [[[0.0, 1.0, 0.0, 0.0, 0.0, 45.0]],
     [[0.0, 1.0, 0.0, 0.0, 0.0, 45.0]],
     [[0.0, 1.0, 0.0, 0.0, 0.0, 45.0]],
     [[0.0, 1.0, 0.0, 0.0, 0.0, 45.0]]
    ], 
    dtype=torch.float, device=device, requires_grad=True)
# act = torch.tensor(
#     [[[0.0, 30.0, 0.0, 0.90], [0.0, 30.0, 0.0, 0.90]],
#      [[0.0, 30.0, 0.0, 0.90], [0.0, 30.0, 0.0, 0.90]]], 
#     dtype=torch.float, device=device, requires_grad=True)
pos_offset = torch.tensor([0.0425, 0.0, 0.0345], dtype=torch.float, device=device, requires_grad=True)
euler_offset = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device, requires_grad=True)

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
# drone = scene.add_entity(
#     morph=genesis.morphs.Drone(
#         file="urdf/ge_fpv.urdf",
#         pos=init_pos.clone().detach().to('cpu').numpy(),
#         euler=init_euler.clone().detach().to('cpu').numpy(),
#     ),
# )
# scene.add_entity(
#     genesis.morphs.Sphere(
#         pos=(2.0, 0.0, 1.0),
#         radius=0.2,
#         fixed=True,
#     )
# )
scene.add_entity(
    genesis.morphs.Cylinder(
        height=2.0,
        radius=0.3,
        pos=(-5.0, 0.0, 1.0),
        fixed=True,
    )
)
scene.add_entity(
    genesis.morphs.Cylinder(
        height=2.0,
        radius=0.4,
        pos=(-2.0, 1.0, 1.0),
        fixed=True,
    )
)
scene.add_entity(
    genesis.morphs.Cylinder(
        height=2.0,
        radius=0.2,
        pos=(2.0, 3.0, 1.0),
        fixed=True,
    )
)
scene.add_entity(
    genesis.morphs.Cylinder(
        height=2.0,
        radius=0.6,
        pos=(3.0, 2.0, 1.0),
        fixed=True,
    )
)
scene.add_entity(
    genesis.morphs.Cylinder(
        height=2.0,
        radius=0.2,
        pos=(-3.0, -2.0, 1.0),
        fixed=True,
    )
)
scene.add_entity(
    genesis.morphs.Cylinder(
        height=2.0,
        radius=0.6,
        pos=(5.0, 0.0, 1.0),
        fixed=True,
    )
)
scene.add_entity(
    genesis.morphs.Cylinder(
        height=2.0,
        radius=0.5,
        pos=(5.0, 2.0, 1.0),
        fixed=True,
    )
)
scene.build()

geom = geom.geom(
    batch_size=4,
    device=device,
)
for i in range(1):
    rigid_robot = robot.rigid(
        device = device,
        init_pos = init_pos, 
        init_euler = init_euler, 
        mass = 0.33, 
        collision_radius = 0.01
    )
    cloest_dist = sensor.cloest_dist(
        device = device
    )
    depth = sensor.depth(
        device = device,
        pos_offset = pos_offset, 
        euler_offset = euler_offset,  
        res_W = 40, 
        res_H = 25, 
        fov_H = 67.9,
        fov_V = 45.3,
        distance_range = {'min': 0.25, 'max': 2.5},
        noise_range = {'min':0.0, 'max':0.0},
        black_hole_prob = 0.0
    )
    lidar_2D = sensor.lidar_2D(
        device = device,
        pos_offset = pos_offset, 
        euler_offset = euler_offset,  
        angle_range = {'start':0.0, 'end':360.0},
        angular_res = 1.0,
        distance_range = {'min': 0.25, 'max': 8.0},
        noise_range = {'min':0.0, 'max':0.0}
    )
    rigid_robot.sensor_bind(cloest_dist)
    rigid_robot.sensor_bind(depth)
    rigid_robot.sensor_bind(lidar_2D)
    geom.add_robot(rigid_robot)
for idx in range(17):
    # geom.add_sphere(
    #     torch.tensor([2.0, 0.0, 1.0, 0.2], dtype=torch.float, device=device),
    #     idx = idx
    # )
    # geom.add_sphere(
    #     torch.tensor([0.0, 0.0, 0.0, 0.8], dtype=torch.float, device=device),
    #     idx = idx
    # )
    geom.add_cylinder(
        torch.tensor([-5.0, 0.0, 1.0, 0.3, 2.0], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_cylinder(
        torch.tensor([-2.0, 1.0, 1.0, 0.4, 2.0], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_cylinder(
        torch.tensor([2.0, 3.0, 1.0, 0.2, 2.0], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_cylinder(
        torch.tensor([3.0, 2.0, 1.0, 0.6, 2.0], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_cylinder(
        torch.tensor([-3.0, -2.0, 1.0, 0.2, 2.0], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_cylinder(
        torch.tensor([5.0, 0.0, 1.0, 0.6, 2.0], dtype=torch.float, device=device),
        idx = idx
    )
    geom.add_cylinder(
        torch.tensor([5.0, 2.0, 1.0, 0.5, 2.0], dtype=torch.float, device=device),
        idx = idx
    )
geom.build()
count = 0

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

while True:
    start = time.perf_counter()
    obs = geom.step(
        mode = 'vel+ang',
        T_att_range = {'min':0.0, 'max':0.0},
        act = act,
        alpha_1_range = {'min':0.8, 'max':0.8},
        alpha_2_range = {'min':0.8, 'max':0.8},
        dt = 0.01
    )

    """ 深度图 """
    # img = 255*obs['depth'][0, 0, ...].detach().cpu().numpy()/2.5
    # # 2. 归一化到 0~255（深度图必须做这步）
    # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # cv2.imshow("DEPTH VIEWER", img.astype(np.uint8))
    # cv2.waitKey(1)
    """ 点云 """
    cloud_point_show(obs['cloud_point'], 150, 8.0)

    end = time.perf_counter()
    elapsed = end - start
    # print(obs['cloud_point'][0, 0, :, :])
    print(f"pos {obs['pos'][0, 0, ...]}")
    print(f"dist {obs['distance'][0, 0, ...]}")
    print(obs['pos'].shape)
    print(obs['distance'].shape)
    print(obs['depth'].shape)
    print(f"{1/elapsed}\n")

    scene.draw_debug_sphere(pos=obs['pos'][0, 0, ...].clone().detach().cpu(), radius=0.1, color=(1, 0, 0))
    # drone.set_pos(obs['pos'][0, 0, ...].clone().detach())
    # drone.set_quat(util.euler_to_quat(util.deg_to_rad(obs['ang'][0, 0, ...]).clone().detach()))
    scene.step()
    # break

# drone_robot.reset()
# while True:
#     drone_robot.solver(
#         'euler+T_rate', 
#         {'min':0.0, 'max':0.0}, 
#         act, 
#         {'min':0.0, 'max':0.0}, 
#         0.01
#     )
#     drone.set_pos(drone_robot.pos.clone().detach())
#     drone.set_quat(util.euler_to_quat(drone_robot.euler.clone().detach()))
#     scene.step()

    # drone_robot.sensor_list[0]['sensor_class'].render(init_euler, drone_robot.euler)
    # print(drone_robot.sensor_list[0]['sensor_class'].pos)