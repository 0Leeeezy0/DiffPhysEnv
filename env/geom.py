import torch
from typing import Union
import copy

import env.util as util
import env.geom as geom
import env.robot as robot
import env.sensor as sensor

"""
    将机器人作为质点,深度相机对无人机有一个pos_offset和euler_offset
"""
class geom:
    """
        @ GEOM场景初始化
        batch_size:并行数量
        device:运行设备:cpu/cuda
        # domain_randomization:域随机化
    """
    def __init__(
        self, 
        batch_size: int, 
        device: str
    ):
        self._batch_size = batch_size
        self._device = device

        self.geom_list = []
        self.robot_list = []
        self.sphere_list = []
        self.cylinder_list = []
    
    """
        @ GEOM场景添加机器人
        robot:机器人类(drone)
    """
    def add_robot(
        self,
        robot: Union[robot.drone],
    ):
        for idx in range(self._batch_size):
            robot_copy = copy.deepcopy(robot)
            self.robot_list.append(dict(type=robot_copy.type, robot=robot_copy, idx=idx))

    """
        @ 向GEOM的一个并行场景中加入球体
        param:球体参数:torch.tensor([x,y,z,R], dtype=torch.float, device=device, requires_grad=False)
        idx:并行场景编号:0-batch_size-1
        x,y,z:球体中心位置:m
        R:球体形状:m
        idx:并行环境索引
    """
    def add_sphere(
        self, 
        param: torch.Tensor,
        idx: int
    ):
        self.sphere_list.append(dict(param=param, idx=idx))

    """
        @ 向GEOM的一个并行场景中加入圆柱
        param:圆柱参数:torch.tensor([x,y,z,R,H], dtype=torch.float, device=device, requires_grad=False)
        idx:并行场景编号:0-batch_size-1
        x,y,z:圆柱中心位置:m
        R,H:圆柱形状:m
        idx:并行环境索引
    """
    def add_cylinder(
        self, 
        param: torch.Tensor,
        idx: int
    ):
        self.cylinder_list.append(dict(param=param, idx=idx))
    
    """
        @ GEOM复位
    """
    def reset(self):
        """ 遍历所有机器人，复位机器人 """
        for robot_dict in self.robot_list:
            robot_dict['robot'].reset()
    
    """
        @ GEOM清除场景
    """
    def clear(self):
        self.sphere_list.clear()
        self.cylinder_list.clear()

    """
        @ GEOM构建
        必须要有sphere和cylinder,最近不想修这个了 @。@
        @ 返回值
        geom_obs字典:
            geom_obs['acc'] = [batch_size, robot_num, 3]:Tensor
            geom_obs['vel'] = [batch_size, robot_num, 3]:Tensor
            geom_obs['pos'] = [batch_size, robot_num, 3]:Tensor
            geom_obs['ang_vel'] = [batch_size, robot_num, 3]:Tensor
            geom_obs['ang'] = [batch_size, robot_num, 3]:Tensor:度
            geom_obs['distance'] = [batch_size, closest_dist_num*robot_num, 1]:Tensor
            geom_obs['is_collision'] = [batch_size, closest_dist_num*robot_num, 1]:Tensor-Bool
            geom_obs['depth'] = [batch_size, closest_dist_num*robot_num, res_H, res_W]:Tensor
            geom_obs['cloud_point'] = [batch_size, closest_dist_num*robot_num, distance, ang]:Tensor
    """
    def build(self):
        self.geom_list.clear()
        """ 遍历各个并行场景 """
        for idx in range(self._batch_size):
            single_geom_robot_list = []
            single_geom_sphere_list = []
            single_geom_cylinder_list = []

            # 遍历所有机器人，匹配对应场景
            for robot_dict in self.robot_list:
                if robot_dict['idx'] == idx:
                    single_geom_robot_list.append(robot_dict['robot'])
            # 遍历所有球体，匹配对应场景
            for sphere_dict in self.sphere_list:
                if sphere_dict['idx'] == idx:
                    single_geom_sphere_list.append(sphere_dict['param'])
            # 遍历所有圆柱，匹配对应场景
            for cylinder_dict in self.cylinder_list:
                if cylinder_dict['idx'] == idx:
                    single_geom_cylinder_list.append(cylinder_dict['param'])
            """ 添加到场景列表 """ 
            self.geom_list.append(
                dict(
                    robot_list = single_geom_robot_list, 
                    sphere_list = util.tensor_stack(single_geom_sphere_list, dim=0, size=(1,4), dtype=torch.float, device=self._device, requires_grad=True),
                    cylinder_list = util.tensor_stack(single_geom_cylinder_list, dim=0, size=(1,5), dtype=torch.float, device=self._device, requires_grad=True)
                )
            )
        """ 计算初始传感器值 """
        geom_obs = {
            'acc': [],
            'vel': [],
            'pos': [],
            'ang_vel': [],
            'ang': [],
            'distance': [],
            'is_collision': [],
            'depth': [],
            'cloud_point': [],
        }
        """ 遍历各个并行场景 """
        for geom_dict in self.geom_list:
            single_geom_obs = {
                'acc': [],
                'vel': [],
                'pos': [],
                'ang_vel': [],
                'ang': [],
                'distance': [],
                'is_collision': [],
                'depth': [],
                'cloud_point': [],
            }
            """ 遍历场景下的机器人 """
            for robot in geom_dict['robot_list']:
                """ 传感器 """
                for sensor_dict in robot.sensor_list:
                    """ 最近距离计算 """
                    if sensor_dict['type'] == 'closest_dist':
                        sensor_dict['sensor'].distance_calc(
                            is_ground_exist = True,
                            collision_radius = robot.collision_radius,
                            sphere_list = geom_dict['sphere_list'],
                            cylinder_list = geom_dict['cylinder_list']
                        )
                        single_geom_obs['distance'].append(sensor_dict['sensor'].distance)
                        single_geom_obs['is_collision'].append(sensor_dict['sensor'].is_collision)
                    """ 深度相机渲染 """
                    if sensor_dict['type'] == 'depth':
                        sensor_dict['sensor'].render(
                            sphere_list = geom_dict['sphere_list'],
                            cylinder_list = geom_dict['cylinder_list'],
                        ) 
                        single_geom_obs['depth'].append(sensor_dict['sensor'].distance)
                    """ 2D激光雷达渲染 """
                    if sensor_dict['type'] == 'lidar_2D':
                        sensor_dict['sensor'].render(
                            sphere_list = geom_dict['sphere_list'],
                            cylinder_list = geom_dict['cylinder_list'],
                        ) 
                        single_geom_obs['cloud_point'].append(torch.stack([sensor_dict['sensor'].distance, sensor_dict['sensor'].ang], dim=1))
                """ 添加各个状态 """
                single_geom_obs['acc'].append(robot.acc)
                single_geom_obs['vel'].append(robot.vel)
                single_geom_obs['pos'].append(robot.pos)
                single_geom_obs['ang_vel'].append(robot.ang_vel)
                single_geom_obs['ang'].append(util.rad_to_deg(robot.euler))

            geom_obs['acc'].append(util.tensor_stack(single_geom_obs['acc'], dim=0, size=(3), dtype=torch.float, device=self._device, requires_grad=True))
            geom_obs['vel'].append(util.tensor_stack(single_geom_obs['vel'], dim=0, size=(3), dtype=torch.float, device=self._device, requires_grad=True))
            geom_obs['pos'].append(util.tensor_stack(single_geom_obs['pos'], dim=0, size=(3), dtype=torch.float, device=self._device, requires_grad=True))
            geom_obs['ang_vel'].append(util.tensor_stack(single_geom_obs['ang_vel'], dim=0, size=(3), dtype=torch.float, device=self._device, requires_grad=True))
            geom_obs['ang'].append(util.tensor_stack(single_geom_obs['ang'], dim=0, size=(3), dtype=torch.float, device=self._device, requires_grad=True))
            geom_obs['distance'].append(util.tensor_stack(single_geom_obs['distance'], dim=0, size=(1), dtype=torch.float, device=self._device, requires_grad=True))
            geom_obs['is_collision'].append(util.tensor_stack(single_geom_obs['is_collision'], dim=0, size=(1), dtype=torch.bool, device=self._device, requires_grad=False))
            geom_obs['depth'].append(util.tensor_stack(single_geom_obs['depth'], dim=0, size=(1,1), dtype=torch.float, device=self._device, requires_grad=True))
            geom_obs['cloud_point'].append(util.tensor_stack(single_geom_obs['cloud_point'], dim=0, size=(1,1), dtype=torch.float, device=self._device, requires_grad=True))
                            
        return {
            'acc': torch.stack(geom_obs['acc'], dim=0),
            'vel': torch.stack(geom_obs['vel'], dim=0),
            'pos': torch.stack(geom_obs['pos'], dim=0),
            'ang_vel': torch.stack(geom_obs['ang_vel'], dim=0),
            'ang': torch.stack(geom_obs['ang'], dim=0),
            'distance': torch.stack(geom_obs['distance'], dim=0),
            'is_collision': torch.stack(geom_obs['is_collision'], dim=0),
            'depth': torch.stack(geom_obs['depth'], dim=0),
            'cloud_point': torch.stack(geom_obs['cloud_point'], dim=0)
        }

    """
        @ GEOM仿真
        mode:解算模式:'euler+T_rate'/acc+yaw'/'ang_vel+T_rate':姿态+推力率/加速度+yaw/角速度+推力率
        T_att_range:推力衰减率范围:{'min':min, 'max':max}:0-1.0
        act:所有场景的所有机器人的动作:按照场景顺序和机器人添加顺序:torch.tensor([[[a,b,c,d], ...], ...], dtype=torch.float, device=device, requires_grad=True):[batch_size,num,4]
        alpha_1_range:延迟范围:{'min':min, 'max':max}:0-1.0
        alpha_2_range:延迟范围:{'min':min, 'max':max}:0-1.0
        dt:步长:s
        @ 返回值
        geom_obs字典:
            geom_obs['acc'] = [batch_size, robot_num, 3]:Tensor
            geom_obs['vel'] = [batch_size, robot_num, 3]:Tensor
            geom_obs['pos'] = [batch_size, robot_num, 3]:Tensor
            geom_obs['ang_vel'] = [batch_size, robot_num, 3]:Tensor
            geom_obs['ang'] = [batch_size, robot_num, 3]:Tensor:度
            geom_obs['distance'] = [batch_size, closest_dist_num*robot_num, 1]:Tensor
            geom_obs['is_collision'] = [batch_size, closest_dist_num*robot_num, 1]:Tensor-Bool
            geom_obs['depth'] = [batch_size, closest_dist_num*robot_num, res_H, res_W]:Tensor
            geom_obs['cloud_point'] = [batch_size, closest_dist_num*robot_num, distance, ang]:Tensor
    """
    def step(self, 
        mode: str,
        T_att_range: dict, 
        act: torch.Tensor, 
        alpha_1_range: dict, 
        alpha_2_range: dict, 
        dt: int
    ):
        idx_geom = 0
        geom_obs = {
            'acc': [],
            'vel': [],
            'pos': [],
            'ang_vel': [],
            'ang': [],
            'distance': [],
            'is_collision': [],
            'depth': [],
            'cloud_point': [],
        }
        """ 遍历各个并行场景 """
        for geom_dict in self.geom_list:
            idx_robot = 0
            single_geom_obs = {
                'acc': [],
                'vel': [],
                'pos': [],
                'ang_vel': [],
                'ang': [],
                'distance': [],
                'is_collision': [],
                'depth': [],
                'cloud_point': [],
            }
            """ 遍历场景下的机器人 """
            for robot in geom_dict['robot_list']:
                """ 机器人动力学解算 """
                if robot.type == 'rigid':
                    robot.solver(
                        mode = mode,
                        act = act[idx_geom, idx_robot, ...],
                        alpha_1_range = alpha_1_range,
                        alpha_2_range = alpha_2_range,
                        dt = dt
                    )
                if robot.type == 'drone':
                    robot.solver(
                        mode = mode,
                        T_att_range = T_att_range,
                        act = act[idx_geom, idx_robot, ...],
                        alpha_1_range = alpha_1_range,
                        dt = dt
                    )
                """ 传感器 """
                for sensor_dict in robot.sensor_list:
                    """ 最近距离计算 """
                    if sensor_dict['type'] == 'closest_dist':
                        sensor_dict['sensor'].distance_calc(
                            is_ground_exist = True,
                            collision_radius = robot.collision_radius,
                            sphere_list = geom_dict['sphere_list'],
                            cylinder_list = geom_dict['cylinder_list']
                        )
                        single_geom_obs['distance'].append(sensor_dict['sensor'].distance)
                        single_geom_obs['is_collision'].append(sensor_dict['sensor'].is_collision)
                    """ 深度相机渲染 """
                    if sensor_dict['type'] == 'depth':
                        sensor_dict['sensor'].render(
                            sphere_list = geom_dict['sphere_list'],
                            cylinder_list = geom_dict['cylinder_list'],
                        )  
                        single_geom_obs['depth'].append(sensor_dict['sensor'].distance)
                    """ 2D激光雷达渲染 """
                    if sensor_dict['type'] == 'lidar_2D':
                        sensor_dict['sensor'].render(
                            sphere_list = geom_dict['sphere_list'],
                            cylinder_list = geom_dict['cylinder_list'],
                        ) 
                        single_geom_obs['cloud_point'].append(torch.stack([sensor_dict['sensor'].distance, sensor_dict['sensor'].ang], dim=1))
                """ 添加各个状态 """
                single_geom_obs['acc'].append(robot.acc)
                single_geom_obs['vel'].append(robot.vel)
                single_geom_obs['pos'].append(robot.pos)
                single_geom_obs['ang_vel'].append(robot.ang_vel)
                single_geom_obs['ang'].append(util.rad_to_deg(robot.euler))

                idx_robot += 1

            geom_obs['acc'].append(util.tensor_stack(single_geom_obs['acc'], dim=0, size=(3), dtype=torch.float, device=self._device, requires_grad=True))
            geom_obs['vel'].append(util.tensor_stack(single_geom_obs['vel'], dim=0, size=(3), dtype=torch.float, device=self._device, requires_grad=True))
            geom_obs['pos'].append(util.tensor_stack(single_geom_obs['pos'], dim=0, size=(3), dtype=torch.float, device=self._device, requires_grad=True))
            geom_obs['ang_vel'].append(util.tensor_stack(single_geom_obs['ang_vel'], dim=0, size=(3), dtype=torch.float, device=self._device, requires_grad=True))
            geom_obs['ang'].append(util.tensor_stack(single_geom_obs['ang'], dim=0, size=(3), dtype=torch.float, device=self._device, requires_grad=True))
            geom_obs['distance'].append(util.tensor_stack(single_geom_obs['distance'], dim=0, size=(1), dtype=torch.float, device=self._device, requires_grad=True))
            geom_obs['is_collision'].append(util.tensor_stack(single_geom_obs['is_collision'], dim=0, size=(1), dtype=torch.bool, device=self._device, requires_grad=False))
            geom_obs['depth'].append(util.tensor_stack(single_geom_obs['depth'], dim=0, size=(1,1), dtype=torch.float, device=self._device, requires_grad=True))
            geom_obs['cloud_point'].append(util.tensor_stack(single_geom_obs['cloud_point'], dim=0, size=(1,1), dtype=torch.float, device=self._device, requires_grad=True))
                
            idx_geom += 1
        
        return {
            'acc': torch.stack(geom_obs['acc'], dim=0),
            'vel': torch.stack(geom_obs['vel'], dim=0),
            'pos': torch.stack(geom_obs['pos'], dim=0),
            'ang_vel': torch.stack(geom_obs['ang_vel'], dim=0),
            'ang': torch.stack(geom_obs['ang'], dim=0),
            'distance': torch.stack(geom_obs['distance'], dim=0),
            'is_collision': torch.stack(geom_obs['is_collision'], dim=0),
            'depth': torch.stack(geom_obs['depth'], dim=0),
            'cloud_point': torch.stack(geom_obs['cloud_point'], dim=0)
        }

