import torch

class pid(torch.nn.Module):
    def __init__(self, batch_size, kp, ki, kd, sigma_limit, output_limit, device):
        super().__init__()
        self._batch_size = batch_size
        self.device = device

        # PID 参数（保持可微分，不 detach）
        self.kp = torch.full((batch_size, 1), kp, device=device)
        self.ki = torch.full((batch_size, 1), ki, device=device)
        self.kd = torch.full((batch_size, 1), kd, device=device)

        # 限幅
        self.sigma_limit = sigma_limit
        self.output_limit = output_limit

        # 状态量（注册为 buffer，不参与训练，但保持梯度流动）
        self.register_buffer('last_err', torch.zeros(batch_size, 1, device=device))
        self.register_buffer('sigma_err', torch.zeros(batch_size, 1, device=device))

    def position(self, current, target):
        # 误差（完全可微分）
        current_err = target - current

        # 积分（累积，不覆盖式破坏计算图）
        sigma_err = self.sigma_err + current_err
        sigma_err = torch.clamp(sigma_err, -self.sigma_limit, self.sigma_limit)

        # PID 输出
        output = (
            self.kp * current_err
            + self.ki * sigma_err
            + self.kd * (current_err - self.last_err)
        )

        # 输出限幅
        output = torch.clamp(output, -self.output_limit, self.output_limit)

        # 更新状态（用 buffer 保存，不破坏梯度）
        self.sigma_err = sigma_err.detach()  # 只阻断状态累积，不阻断输入梯度
        self.last_err = current_err.detach()

        return output

    def reset(self):
        self.last_err.zero_()
        self.sigma_err.zero_()

