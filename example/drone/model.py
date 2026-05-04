import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    输入:
        depth:深度图像:[batch_size, robot_num, seq_len, 25, 40]
        acc:加速度:[batch_size, robot_num, seq_len, 3]
        ang:角度(欧拉角):[batch_size, robot_num, seq_len, 3]
        ang_vel:角速度:[batch_size, robot_num, seq_len, 3]
        vel:目标ENU速度:[batch_size, robot_num, seq_len, 1]
    输出:
        act:动作输出:[batch_size, robot_num, 3]
"""
class Model_Depth_GRU(nn.Module):
    def __init__(self, dim_obs=9, dim_action=3, hidden_size=192):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 深度图像处理CNN (处理单帧)
        # 输入: [batch, 1, 25, 40]
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2), nn.LeakyReLU(0.05), nn.MaxPool2d(2),  # [batch, 32, 6, 10]
            nn.Conv2d(32, 64, 3, 2, 1), nn.LeakyReLU(0.05), nn.MaxPool2d(2),  # [batch, 64, 1, 2]
            nn.Conv2d(64, 128, 3, 1, 1), nn.LeakyReLU(0.05),                  # [batch, 128, 1, 2]
            nn.Conv2d(128, 192, 3, 1, 1), nn.LeakyReLU(0.05),                 # [batch, 192, 1, 2]
            nn.AdaptiveAvgPool2d(1), nn.Flatten()                             # [batch, 192]
        )
        
        # 观测投影 (acc, ang, ang_vel, vel 统一处理)
        # 输入: [batch, 10] (3+3+3+1=10维)
        self.obs_proj = nn.Linear(dim_obs + 1, hidden_size, bias=False)
        self.obs_proj.weight.data.mul_(0.5)
        
        # GRU Cell
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        
        # 输出层
        self.fc = nn.Linear(hidden_size, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)
        
    def forward(self, depth, acc, ang, ang_vel, vel, hx=None):
        """
        输入维度:
            depth: [batch_size, robot_num, seq_len, 25, 40]
            acc: [batch_size, robot_num, seq_len, 3]
            ang: [batch_size, robot_num, seq_len, 3]
            ang_vel: [batch_size, robot_num, seq_len, 3]
            vel: [batch_size, robot_num, seq_len, 1]
        输出:
            act: [batch_size, robot_num, 3]
            hx: 隐藏状态
        """
        batch_size, robot_num, seq_len = depth.shape[:3]
        
        # 拼接观测: [batch, robot_num, seq_len, 10]
        obs = torch.cat([acc, ang, ang_vel, vel], dim=-1)
        
        # 合并batch和robot_num维度，以便统一处理
        depth_reshaped = depth.view(batch_size * robot_num, seq_len, 25, 40)
        obs_reshaped = obs.view(batch_size * robot_num, seq_len, -1)
        
        # 提取每帧特征: [batch*robot_num, seq_len, 192]
        frame_features = self.extract_frame_features(depth_reshaped, obs_reshaped)
        
        # 逐帧通过GRUCell
        if hx is None:
            hx = torch.zeros(batch_size * robot_num, self.hidden_size, device=depth.device)
        
        for t in range(seq_len):
            hx = self.gru(frame_features[:, t, :], hx)
        
        # hx: [batch*robot_num, 192]
        # 恢复robot_num维度
        last_hidden = hx.view(batch_size, robot_num, self.hidden_size)  # [batch, robot_num, 192]
        
        # 输出动作: [batch, robot_num, 3]
        act = self.fc(self.act(last_hidden))
        
        return act, hx
    
    def extract_frame_features(self, depth, obs):
        """
        depth: [batch*robot_num, seq_len, 25, 40]
        obs: [batch*robot_num, seq_len, 10]
        返回: [batch*robot_num, seq_len, 192]
        """
        total_batch, seq_len = depth.shape[:2]
        
        # 处理深度图像: 将序列维度展平
        depth_flat = depth.view(-1, depth.shape[2], depth.shape[3])  # [total_batch*seq_len, 25, 40]
        depth_flat = depth_flat.unsqueeze(1)  # [total_batch*seq_len, 1, 25, 40]
        depth_features = self.cnn(depth_flat)  # [total_batch*seq_len, 192]
        depth_features = depth_features.view(total_batch, seq_len, -1)  # [total_batch, seq_len, 192]
        
        # 处理观测
        obs_flat = obs.view(-1, obs.shape[-1])  # [total_batch*seq_len, 10]
        obs_features = self.obs_proj(obs_flat)  # [total_batch*seq_len, 192]
        obs_features = obs_features.view(total_batch, seq_len, -1)  # [total_batch, seq_len, 192]
        
        # 特征相加融合
        frame_features = self.act(depth_features + obs_features)
        
        return frame_features  # [total_batch, seq_len, 192]
    
    def reset(self):
        """重置隐藏状态（用于新episode）"""
        pass

"""
    @ 模型(旧) - 带GRU时序建模
    输入:
        depth:深度图像:[batch_size, seq_len, 25, 40] 或 [batch_size, 25, 40]（单帧）
        acc:加速度:[batch_size, seq_len, 3] 或 [batch_size, 3]
        ang:角度(欧拉角):[batch_size, seq_len, 3] 或 [batch_size, 3]
        ang_vel:角速度:[batch_size, seq_len, 3] 或 [batch_size, 3]
        vel:目标ENU速度:[batch_size, seq_len, 1] 或 [batch_size, 1]
    输出:
        mean:姿态欧拉角(roll\pitch)/推力均值:[batch_size, 3]
        std:姿态欧拉角/推力标准差:[batch_size, 3]
"""
class Model_Depth_GRU_Prob(nn.Module):
    def __init__(self, seq_len=5, hidden_size=128, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        
        # 深度图像处理CNN (处理单帧)
        # 输入: [batch, 1, 25, 40]
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2), nn.LeakyReLU(), nn.MaxPool2d(2),  # [batch, 32, 6, 10]
            nn.Conv2d(32, 64, 3, 2, 1), nn.LeakyReLU(), nn.MaxPool2d(2),  # [batch, 64, 2, 3]
            nn.Conv2d(64, 128, 3, 2, 1), nn.LeakyReLU(),                  # [batch, 128, 1, 2]
            nn.Conv2d(128, 256, 3, 2, 1), nn.LeakyReLU(),                 # [batch, 256, 1, 1]
            nn.AdaptiveAvgPool2d(1), nn.Flatten()                         # [batch, 256]
        )
        
        # 姿态传感器MLP (acc, ang, ang_vel) - 处理单帧
        # 输入: [batch, 9] (3+3+3)
        self.pose_mlp = nn.Sequential(
            nn.Linear(9, 128), nn.LeakyReLU(),
            nn.Linear(128, 128), nn.LeakyReLU()
        )
        
        # 速度处理MLP - 处理单帧
        # 输入: [batch, 1]
        self.vel_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.LeakyReLU(),
            nn.Linear(64, 64), nn.LeakyReLU()
        )
        
        # 单帧特征维度: 深度256 + 姿态128 + 速度64 = 448
        self.frame_feature_dim = 256 + 128 + 64
        
        # GRU模块用于时序建模
        # 输入: [batch, seq_len, frame_feature_dim]
        self.gru = nn.GRU(
            input_size=self.frame_feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 特征融合 (GRU输出的hidden_state)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.LeakyReLU(),
            nn.Linear(512, 512), nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        
        # 输出均值 (3个值: roll, pitch, thrust)
        self.mean = nn.Linear(512, 3)
        
        # 输出标准差 (3个值: 对应每个输出的标准差)
        self.std_head = nn.Sequential(
            nn.Linear(512, 32), nn.LeakyReLU(),
            nn.Linear(32, 3)
        )
        
    def extract_frame_features(self, depth, acc, ang, ang_vel, vel):
        """
        提取单帧特征
        输入可以是单帧 [batch, ...] 或序列 [batch, seq_len, ...]
        """
        batch_size = depth.shape[0]
        
        # 判断是否有序列维度
        if depth.dim() == 4:  # [batch, seq_len, H, W]
            seq_len = depth.shape[1]
            # 重塑为 [batch*seq_len, 1, H, W]
            depth_flat = depth.view(-1, depth.shape[2], depth.shape[3])
            depth_flat = depth_flat.unsqueeze(1)
            depth_features = self.cnn(depth_flat)  # [batch*seq_len, 256]
            depth_features = depth_features.view(batch_size, seq_len, -1)
            
            # 处理序列姿态数据
            pose_input = torch.cat([acc, ang, ang_vel], dim=-1)  # [batch, seq_len, 9]
            pose_flat = pose_input.view(-1, 9)
            pose_features = self.pose_mlp(pose_flat)  # [batch*seq_len, 128]
            pose_features = pose_features.view(batch_size, seq_len, -1)
            
            # 处理序列速度数据
            if vel.dim() == 2:
                vel = vel.unsqueeze(-1)  # [batch, seq_len] -> [batch, seq_len, 1]
            vel_flat = vel.view(-1, 1)
            vel_features = self.vel_mlp(vel_flat)  # [batch*seq_len, 64]
            vel_features = vel_features.view(batch_size, seq_len, -1)
            
        elif depth.dim() == 3:  # 单帧输入 [batch, H, W]
            # 处理深度图像
            depth = depth.unsqueeze(1)  # [batch, 1, H, W]
            depth_features = self.cnn(depth)  # [batch, 256]
            
            # 处理姿态数据
            pose_input = torch.cat([acc, ang, ang_vel], dim=-1)  # [batch, 9]
            pose_features = self.pose_mlp(pose_input)  # [batch, 128]
            
            # 处理速度数据
            if vel.dim() == 1:
                vel = vel.unsqueeze(-1)  # [batch] -> [batch, 1]
            vel_features = self.vel_mlp(vel)  # [batch, 64]
            
            # 拼接并添加序列维度
            frame_features = torch.cat([depth_features, pose_features, vel_features], dim=-1)
            return frame_features.unsqueeze(1)  # [batch, 1, 448]
        else:
            raise ValueError(f"Unsupported depth dimension: {depth.dim()}")
        
        # 拼接所有特征
        frame_features = torch.cat([depth_features, pose_features, vel_features], dim=-1)  # [batch, seq_len, 448]
        return frame_features
    
    def forward(self, depth, acc, ang, ang_vel, vel):
        """
        前向传播
        支持单帧和序列输入
        """
        # 提取每帧特征 [batch, seq_len, frame_feature_dim]
        frame_features = self.extract_frame_features(depth, acc, ang, ang_vel, vel)
        
        # GRU时序建模
        # output: [batch, seq_len, hidden_size]
        # hidden: [num_layers, batch, hidden_size]
        output, hidden = self.gru(frame_features)
        
        # 使用最后一层的最后一个时间步的隐藏状态
        # 取最后一层 (num_layers-1) 的最后一个时间步
        last_hidden = hidden[-1]  # [batch, hidden_size]
        
        # 特征融合
        fused_features = self.fusion(last_hidden)  # [batch, 512]
        
        # 输出均值 (roll, pitch, thrust)
        mean = self.mean(fused_features)  # [batch, 3]
        
        # 输出标准差
        std = self.std_head(fused_features)  # [batch, 3]
        std = F.softplus(std) + 0.01  # 确保标准差为正
        
        return mean, std
    
    """
输入:
    acc:加速度:[batch_size, 3] 或 [batch_size, seq_len, 3]
    ang:角度(欧拉角):[batch_size, 3] 或 [batch_size, seq_len, 3]
    ang_vel:角速度:[batch_size, 3] 或 [batch_size, seq_len, 3]
    vel:目标ENU速度:[batch_size, 1] 或 [batch_size, seq_len, 1]
输出:
    act:动作输出:[batch_size, 3]
"""
class Model_GRU(nn.Module):
    def __init__(self, dim_obs=9, dim_action=3, hidden_size=192):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 观测投影 (acc, ang, ang_vel, vel 统一处理)
        # 输入: [batch, 10] (3+3+3+1=10维)
        self.obs_proj = nn.Linear(dim_obs + 1, hidden_size, bias=False)
        self.obs_proj.weight.data.mul_(0.5)
        
        # GRU Cell
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        
        # 输出层 (修改为3维)
        self.fc = nn.Linear(hidden_size, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)
        
    def forward(self, acc, ang, ang_vel, vel, hx=None):
        """
        前向传播（支持单步和序列）
        输入:
            acc: [batch, 3] 或 [batch, seq_len, 3]
            ang: [batch, 3] 或 [batch, seq_len, 3]
            ang_vel: [batch, 3] 或 [batch, seq_len, 3]
            vel: [batch, 1] 或 [batch, seq_len, 1]
            hx: [batch, hidden_size] 初始隐藏状态，可选
        输出:
            act: [batch, 3] 动作输出
            hx: [batch, hidden_size] 隐藏状态
        """
        # 拼接观测
        if acc.dim() == 3:  # 序列输入
            obs = torch.cat([acc, ang, ang_vel, vel], dim=-1)  # [batch, seq_len, 10]
            batch_size, seq_len = obs.shape[:2]
            
            # 投影观测特征
            obs_flat = obs.view(-1, obs.shape[-1])
            obs_features = self.obs_proj(obs_flat)  # [batch*seq_len, hidden_size]
            obs_features = obs_features.view(batch_size, seq_len, -1)
            obs_features = self.act(obs_features)  # [batch, seq_len, hidden_size]
            
            # 序列处理：逐帧通过GRUCell
            if hx is None:
                hx = torch.zeros(batch_size, self.hidden_size, device=acc.device)
            
            outputs = []
            for t in range(seq_len):
                hx = self.gru(obs_features[:, t, :], hx)
                outputs.append(hx)
            
            # 返回最后一帧的输出
            last_hidden = outputs[-1]
            act = self.fc(self.act(last_hidden))
            return act, hx
            
        else:  # 单帧输入
            obs = torch.cat([acc, ang, ang_vel, vel], dim=-1)  # [batch, 10]
            obs_features = self.obs_proj(obs)  # [batch, hidden_size]
            obs_features = self.act(obs_features)  # [batch, hidden_size]
            
            if hx is None:
                hx = torch.zeros(obs_features.shape[0], self.hidden_size, device=acc.device)
            
            hx = self.gru(obs_features, hx)
            act = self.fc(self.act(hx))
            return act, hx
    
    def reset(self):
        """重置隐藏状态（用于新episode）"""
        pass