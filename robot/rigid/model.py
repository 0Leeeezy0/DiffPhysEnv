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
class Model(nn.Module):
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