import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    输入:
        cloud_point:深度图像:[batch_size, robot_num, seq_len, 500]
        vel:速度:[batch_size, robot_num, seq_len, 3]
        ang:角度(欧拉角):[batch_size, robot_num, seq_len, 3]
        vel:目标ENU速度:[batch_size, robot_num, seq_len, 1]
    输出:
        act:动作输出(速度+角度):[batch_size, robot_num, 6]
"""
class Model_vel(nn.Module):
    def __init__(self, 
                 point_dim=500,
                 hidden_dim=256,
                 num_layers=3,
                 dropout=0.1):
        super(Model, self).__init__()
        
        # 1. 点云特征提取模块 (处理深度图像)
        self.point_cloud_encoder = nn.Sequential(
            nn.Linear(point_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 2. 速度特征提取模块
        self.vel_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 3. 角度特征提取模块
        self.ang_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 4. 目标速度特征提取
        self.target_vel_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # 5. 时序处理模块 (LSTM)
        self.lstm = nn.LSTM(
            input_size=64 + 32 + 32 + 16,  # 拼接后的特征维度
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 6. 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # 双向LSTM
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 7. 输出层 (动作输出)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 6)  # 输出: 速度(3) + 角度(3)
        )
        
    def forward(self, cloud_point, vel, ang, target_vel):
        batch_size, robot_num, seq_len, _ = cloud_point.shape
        
        # 重塑张量以并行处理所有机器人
        # [batch_size * robot_num, seq_len, feature_dim]
        cloud_point_flat = cloud_point.view(batch_size * robot_num, seq_len, -1)
        vel_flat = vel.view(batch_size * robot_num, seq_len, -1)
        ang_flat = ang.view(batch_size * robot_num, seq_len, -1)
        target_vel_flat = target_vel.view(batch_size * robot_num, seq_len, -1)
        
        # 编码各个模态的特征
        # 处理点云 (在每个时间步独立处理)
        point_features = []
        for t in range(seq_len):
            point_t = cloud_point_flat[:, t, :]  # [B*R, 500]
            point_feat_t = self.point_cloud_encoder(point_t)
            point_features.append(point_feat_t)
        point_features = torch.stack(point_features, dim=1)  # [B*R, seq_len, 64]
        
        # 编码速度
        vel_features = self.vel_encoder(vel_flat)  # [B*R, seq_len, 32]
        
        # 编码角度
        ang_features = self.ang_encoder(ang_flat)  # [B*R, seq_len, 32]
        
        # 编码目标速度
        target_features = self.target_vel_encoder(target_vel_flat)  # [B*R, seq_len, 16]
        
        # 融合所有特征
        combined_features = torch.cat([
            point_features,
            vel_features,
            ang_features,
            target_features
        ], dim=-1)  # [B*R, seq_len, 64+32+32+16=144]
        
        # LSTM时序处理
        lstm_out, (hidden, cell) = self.lstm(combined_features)
        # lstm_out: [B*R, seq_len, hidden_dim*2]
        
        # 注意力机制
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 聚合时序信息 (取最后一个时间步或平均池化)
        # 方案1: 取最后一个时间步
        final_features = attn_out[:, -1, :]  # [B*R, hidden_dim*2]
        
        # 方案2: 平均池化 (可选的更好方案)
        # final_features = attn_out.mean(dim=1)
        
        # 输出动作
        actions = self.action_head(final_features)  # [B*R, 6]
        
        # 重塑回原始形状
        actions = actions.view(batch_size, robot_num, 6)
        
        return actions
    
"""
    输入:
        cloud_point:深度图像:[batch_size, robot_num, seq_len, 500]
        pos:位置:[batch_size, robot_num, seq_len, 3]
        ang:角度(欧拉角):[batch_size, robot_num, seq_len, 3]
        pos_tar:目标位置:[batch_size, robot_num, 3]
        vel_tar:目标ENU速度:[batch_size, robot_num, 1]
    输出:
        act:动作输出(速度+角度):[batch_size, robot_num, 6]
"""
class Model_pos(nn.Module):
    def __init__(self, 
                 point_dim=500,
                 hidden_dim=256,
                 num_layers=3,
                 dropout=0.1):
        super(Model_pos, self).__init__()
        
        # 保存参数以便reset时使用
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 1. 点云特征提取模块 (处理深度图像)
        self.point_cloud_encoder = nn.Sequential(
            nn.Linear(point_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 2. 位置特征提取模块 (当前位置)
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 3. 角度特征提取模块
        self.ang_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 4. 目标位置特征提取模块 (无时间序列)
        self.target_pos_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 5. 目标速度特征提取模块 (无时间序列)
        self.target_vel_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # 6. 时序处理模块 (LSTM)
        # 输入特征: 64(point) + 32(pos) + 32(ang) = 128
        # 目标特征将在每个时间步连接
        self.lstm = nn.LSTM(
            input_size=64 + 32 + 32,  # 点云特征 + 位置特征 + 角度特征
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 7. 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # 双向LSTM
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 8. 特征融合层 (融合时序特征和目标特征)
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 32 + 16, 256),  # LSTM输出 + 目标位置特征 + 目标速度特征
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 9. 输出层 (动作输出)
        self.action_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 6)  # 输出: 线速度(x,y,z) + 角速度(roll,pitch,yaw)
        )
        
        # 初始化LSTM隐藏状态
        self.hidden_state = None
        self.cell_state = None
        
    def reset(self, batch_size=None, robot_num=None, device=None):
        """
        重置LSTM的隐藏状态和细胞状态
        
        参数:
            batch_size: 批次大小，如果为None则保持原有设置
            robot_num: 机器人数量，如果为None则保持原有设置
            device: 设备（cpu/cuda），如果为None则自动检测
        """
        if batch_size is not None and robot_num is not None:
            # 新的维度: batch_size * robot_num
            total_batch = batch_size * robot_num
            
            # 确定设备
            if device is None:
                device = next(self.parameters()).device
            
            # 重置隐藏状态和细胞状态
            # LSTM是双向的，所以隐藏状态维度为 [num_layers*2, total_batch, hidden_dim]
            self.hidden_state = torch.zeros(
                self.num_layers * 2, total_batch, self.hidden_dim, 
                device=device
            )
            self.cell_state = torch.zeros(
                self.num_layers * 2, total_batch, self.hidden_dim, 
                device=device
            )
        else:
            # 如果没有指定新的维度，只将状态置为None
            self.hidden_state = None
            self.cell_state = None
    
    def reset_parameters(self):
        """
        重置网络中所有可学习参数的初始化
        用于完全重新初始化模型
        """
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
        
        self.apply(init_weights)
        # 重置状态
        self.reset()
        
    def forward(self, cloud_point, pos, ang, pos_tar, vel_tar, reset_state=False):
        """
        前向传播
        
        参数:
            reset_state: 是否在前向传播前重置LSTM状态
        """
        if reset_state:
            self.reset()
            
        batch_size, robot_num, seq_len, _ = cloud_point.shape
        
        # 重塑张量以并行处理所有机器人
        cloud_point_flat = cloud_point.view(batch_size * robot_num, seq_len, -1)
        pos_flat = pos.view(batch_size * robot_num, seq_len, -1)
        ang_flat = ang.view(batch_size * robot_num, seq_len, -1)
        
        # 目标值没有时间维度，直接重塑
        pos_tar_flat = pos_tar.view(batch_size * robot_num, -1)  # [B*R, 3]
        vel_tar_flat = vel_tar.view(batch_size * robot_num, -1)  # [B*R, 1]
        
        # 编码各个模态的特征
        # 处理点云 (在每个时间步独立处理)
        point_features = []
        for t in range(seq_len):
            point_t = cloud_point_flat[:, t, :]  # [B*R, 500]
            point_feat_t = self.point_cloud_encoder(point_t)
            point_features.append(point_feat_t)
        point_features = torch.stack(point_features, dim=1)  # [B*R, seq_len, 64]
        
        # 编码当前位置 (每个时间步)
        pos_features = self.pos_encoder(pos_flat)  # [B*R, seq_len, 32]
        
        # 编码当前角度 (每个时间步)
        ang_features = self.ang_encoder(ang_flat)  # [B*R, seq_len, 32]
        
        # 编码目标位置 (全局，无时间维度)
        target_pos_features = self.target_pos_encoder(pos_tar_flat)  # [B*R, 32]
        
        # 编码目标速度 (全局，无时间维度)
        target_vel_features = self.target_vel_encoder(vel_tar_flat)  # [B*R, 16]
        
        # 融合时序特征 (点云+位置+角度)
        temporal_features = torch.cat([
            point_features,
            pos_features,
            ang_features
        ], dim=-1)  # [B*R, seq_len, 64+32+32=128]
        
        # LSTM时序处理（使用保存的状态或None）
        if self.hidden_state is not None and self.cell_state is not None:
            # 检查维度是否匹配
            expected_hidden_shape = (self.num_layers * 2, batch_size * robot_num, self.hidden_dim)
            if self.hidden_state.shape != expected_hidden_shape:
                # 如果不匹配，自动重置
                self.reset(batch_size, robot_num, cloud_point.device)
            
            lstm_out, (self.hidden_state, self.cell_state) = self.lstm(
                temporal_features, (self.hidden_state, self.cell_state)
            )
        else:
            # 如果没有保存的状态，使用默认（零状态）
            lstm_out, (self.hidden_state, self.cell_state) = self.lstm(temporal_features)
        
        # 注意力机制
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)  # [B*R, seq_len, hidden_dim*2]
        
        # 取最后一个时间步的时序特征
        last_temporal_features = attn_out[:, -1, :]  # [B*R, hidden_dim*2]
        
        # 将目标特征扩展到每个机器人
        # 扩展维度以便与时序特征拼接
        target_pos_features_expanded = target_pos_features.unsqueeze(1)  # [B*R, 1, 32]
        target_vel_features_expanded = target_vel_features.unsqueeze(1)  # [B*R, 1, 16]
        
        # 融合时序特征和目标特征
        combined_features = torch.cat([
            last_temporal_features.unsqueeze(1),  # [B*R, 1, hidden_dim*2]
            target_pos_features_expanded,
            target_vel_features_expanded
        ], dim=-1)  # [B*R, 1, hidden_dim*2 + 32 + 16]
        
        combined_features = combined_features.squeeze(1)  # [B*R, hidden_dim*2 + 32 + 16]
        
        # 特征融合
        fused_features = self.fusion_layer(combined_features)  # [B*R, 128]
        
        # 输出动作
        actions = self.action_head(fused_features)  # [B*R, 6]
        
        # 重塑回原始形状
        actions = actions.view(batch_size, robot_num, 6)
        
        return actions
    
    def detach_state(self):
        """从计算图中分离LSTM状态，用于防止梯度回溯太远"""
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.detach()
        if self.cell_state is not None:
            self.cell_state = self.cell_state.detach()